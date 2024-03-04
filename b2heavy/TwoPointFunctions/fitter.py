import os
import itertools
import jax 
import jax.numpy         as jnp
import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import lsqfit


from .utils     import MPHYS, load_toml, NplusN2ptModel, correlation_diagnostics
from .types2pts import CorrelatorInfo, CorrelatorIO, Correlator



def PeriodicExpDecay_jax(Nt):
    return lambda t,E,Z: Z * ( jnp.exp(-E*t) + jnp.exp(-E*(Nt-t)) ) 

def NplusN2ptModel_jax(Nstates,Nt,sm,pol):
    sm1,sm2 = sm.split('-')
    mix = sm1!=sm2

    def aux(t,p):
        E0, E1 = p['E'][0], p['E'][0] + jnp.exp(p['E'][1])
        Z0 = jnp.exp(p[f'Z_{sm1}_{pol}'][0]) * jnp.exp(p[f'Z_{sm2}_{pol}'][0])
        Z1 = jnp.exp(p[f'Z_{sm1}_{pol}'][1]) * jnp.exp(p[f'Z_{sm2}_{pol}'][1])
        ans = PeriodicExpDecay_jax(Nt)(t,E0,Z0) + PeriodicExpDecay_jax(Nt)(t,E1,Z1) * (-1)**(t+1)

        Es = [E0,E1]
        for i in range(2,2*Nstates):
            Ei = Es[i-2] + jnp.exp(p['E'][i])
            Z = p[f'Z_{sm if mix else sm1}_{pol}'][i-2 if mix else i]**2
            ans += PeriodicExpDecay_jax(Nt)(t,Ei,Z) * (-1)**(i*(t+1))

            Es.append(Ei)
        return ans

    return aux

def set_priors_phys(corr, Nstates, Meff=None, Aeff=None, prior_policy=None):
    """
        This function returns a dictionary with priors, according to the following rules:

        Energies
        ---------
        - The prior for the **fundamental physical state at zero momentum** is set with the effective mass, while its width is set following the paper to $140\text{ MeV}$
        $$
            \tilde{E}_0(\mathbf{p}=0) = M_\text{eff} \pm 140\text{ MeV}
        $$ 
        - At **non-zero momentum** 
        $$
            \tilde{E}_0(\mathbf{p}\neq0) = \sqrt{M_\text{eff}^2+\mathbf{p}^2} \pm (140\text{ MeV} + \#\alpha_sa^2\mathbf{p}^2)
        $$ 
        - The energy priors for the energy differences are $\Delta \tilde E = 500(200) \text{ MeV}$

        Overlap factors
        ---------------
        We will use the following rules
        - For non-mixed smearing, physical fund. state, we use the plateaux of $\mathcal{C}(t)e^{tM_\text{eff}}=A_\text{eff}(t)$
        $$
        \tilde Z_0^{(\text{sm},\text{pol})} = \sqrt{A_\text{eff}} \, \pm (0.5 \,\mathtt{if\,[p=0]\,else}\,?)
        $$
        - For non-mixed smearing, oscillating fund. state, we use
        $$
        \tilde Z_1^{(\text{sm},\text{pol})} = \sqrt{A_\text{eff}} \, \pm (1.2 \,\mathtt{if\,[p=0]\,else}\,2.0)
        $$
        - For mixed smearing, excited states, we always use
        $$
        \tilde Z_{n\geq2}^{(\text{sm},\text{pol})} = 0.5 \pm 1.5
        $$
    """    
    at_rest = corr.info.momentum=='000'
    p2 = sum([(2*np.pi*float(px)/corr.io.mData['L'])**2 for px in corr.info.momentum])


    dScale = corr.io.mData['mDs']/MPHYS['Ds']
    bScale = corr.io.mData['mBs']/MPHYS['Bs']
    aGeV   = corr.io.mData['aSpc'].mean/corr.io.mData['hbarc']
    
    # SET ENERGIES ---------------------------------------------------------------------------
    if prior_policy==None:
        c = os.path.join(os.path.dirname(os.path.abspath(__file__)),'fitter_PriorPolicy.toml')
    else:
        c = prior_policy
    d = load_toml(c)[corr.info.meson]

    Scale    = dScale if d['scale']=='D' else bScale if d['scale']=='B' else 1.
    dE       = d['dE']      
    dG       = d['dG']       
    dE_E0err = d['dE_E0err']
    dE_E1err = d['dE_E1err']
    dE_Eierr = d['dE_Eierr'] if 'dE_Eierr' in d else 1.  # old was d['dE_E1err'] if 'dE_E1err'

    E = [None]*(2*Nstates)

    if Meff is None:
        E[0] = gv.gvar(MPHYS[corr.info.meson],dE/dE_E0err) * Scale * aGeV # fundamental physical state
    else:
        # E[0] = gv.gvar(Meff.mean,dE/dE_E0err*Scale*aGeV)
        E[0] = Meff

    E[1] = np.log(gv.gvar(dG, dE/dE_E1err)*Scale*aGeV)

    for n in range(2,2*Nstates):
        E[n]= np.log(gv.gvar(dE,dE/dE_Eierr)*Scale*aGeV) # excited states

        if corr.info.meson=='Dst' and n==2:
            E1 = (2.630-MPHYS['Dst'])*Scale
            E[2] = np.log(gv.gvar(
                E1,
                dE/dE_Eierr
            ) * aGeV )

    priors = {'E': E}

    if not at_rest and Meff is None: # non-zero mom
        priors['E'][0] = gv.gvar(
            np.sqrt(priors['E'][0].mean**2 + p2),
            np.sqrt(4*(priors['E'][0].mean**2)*(priors['E'][0].sdev**2) + (p2*corr.io.mData['alphaS'])**2)/(2*priors['E'][0].mean)
        )
    
    # SET OVERLAP FACTORS --------------------------------------------------------------------
    apEr = corr.io.mData["alphaS"]*p2

    lbl = [] # infer all smearing labels
    for smr,pol in corr.keys:
        sm1,sm2 = smr.split('-')
        lbl.append(f'{sm1}_{pol}' if sm1==sm2 else f'{smr}_{pol}')         
    lbl = np.unique(lbl)

    for smr in list(lbl):
        if len(corr.info.meson) > 2:
            if corr.info.meson[-2:] == 'st':
                if 'd' in smr:
                    val  = (np.log((corr.io.mData['aSpc']*5.3 + 0.54)*corr.io.mData['aSpc'])).mean
                    err  = (np.sqrt((np.exp(val)*val*0.2)**2 + apEr**2)/np.exp(val))
                    bVal = gv.gvar(val, err)      

                    if smr.split('_')[-1]=='Par':
                        oVal = gv.gvar(-5.5,2.0)
                    else:
                        oVal = gv.gvar(-3.,1.5)  

                else:
                    bVal = gv.gvar(-2.0, 2.0) - 0.5*np.log(priors['E'][0].mean)
                    oVal = gv.gvar(-9.5, 2.0) - 2.0*np.log(corr.io.mData['aSpc'].mean)                                                        
        else:
            bVal = gv.gvar(-1.5, 1.5)
            oVal = gv.gvar(-2.5, 1.5)  

        baseGv = gv.gvar( 0.0, 1.2) if '1S' in smr else gv.gvar(bVal.mean, bVal.sdev)
        osciGv = gv.gvar(-1.2, 1.2) if '1S' in smr else gv.gvar(oVal.mean, oVal.sdev)
        highGv = gv.gvar( 0.5, 1.5)

        priors[f'Z_{smr}'] = []

        if len(smr.split('-'))==1: # single smearing case
            priors[f'Z_{smr}'].append(gv.gvar(baseGv.mean, baseGv.sdev)),
            priors[f'Z_{smr}'].append(gv.gvar(osciGv.mean, osciGv.sdev))
        
        for n in range(2,2*Nstates):
            priors[f'Z_{smr}'].append(
                gv.gvar(highGv.mean, highGv.sdev*(n//2)) if len(smr.split('-'))==1 else gv.gvar(0.5,1.7)
            )
    
    if Aeff is not None:
        for (sm,pol),v in Aeff.items():
            sm1,sm2 = sm.split('-')
            if sm1==sm2:
                # priors[f'Z_{sm1}_{pol}'][0] = np.log(v)/2
                v = np.log(v)/2
                # priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(np.log(v.mean)/2,priors[f'Z_{sm1}_{pol}'][0].sdev)
                priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev*100)
                # priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev)

    return priors  


class StagFitter(Correlator):
    def __init__(self, io:CorrelatorIO, priors_policy=None, **kwargs):
        super().__init__(io,**kwargs)
        self.priors_policy = priors_policy

    def priors(self, Nstates, Meff=None, Aeff=None, prior_policy=None):
        prs = self.priors_policy if prior_policy is None else prior_policy
        return set_priors_phys(self,Nstates, Meff=Meff, Aeff=Aeff, prior_policy=prs)

    def diff_model(self, xdata, Nstates):
        '''
            Return a `jax`-differentiable function, its jacobian and its hessian.

            Parameters
            ----------
            - xdata: dict. Dictionary with x-data, keys have to be `self.keys` 
            - Nexc: int. Number of excited states

            Returns
            -------
            - _model: func. Takes as argument the usual dict of parameter and passes it to `NplusN2ptModel`
            - _jac  : func. `jax` jacobian of the function (same signature of `_model`)
            - _hes  : func. `jax` hessian of the function (same signature of `_model`)
        '''

        def _model(pdict):  
            return jnp.concatenate(
                [NplusN2ptModel_jax(Nstates,self.Nt,smr,pol)(xdata,pdict) for smr,pol in self.keys]
            )

        def _jac(pdict):
            return jax.jacfwd(_model)(pdict)

        def _hes(pdict):
            return jax.jacfwd(jax.jacrev(_model))(pdict)
            
        return _model, _jac, _hes

    def diff_cost(self, Nexc, W2=None, **format_args):
        '''
            Return a `jax`-differentiable cost function.

            Parameters
            ----------
                - Nexc: int. Number of excited states to be fitted.
                - W2: matrix, optional. Weight matrix of the fit (see notes below).
        '''
        xdata,ydata = self.format(flatten=False, **format_args)

        # Prepare model, vector with y-data and weight matrix
        model, jac, hess = self.diff_model(xdata,Nexc)

        if W2 is not None:
            yvec = np.concatenate([ydata[k] for k in self.keys])
            wmat2 = jnp.asarray(W2) if W2 is not None else jnp.linalg.inv(gv.evalcov(yvec))
        else:
            wmat2 = W2


        def _cost(y,pdict):
            res = y - model(pdict)
            return jnp.matmul(jnp.transpose(res),jnp.matmul(wmat2,res))

        return _cost

    def fit(self, Nstates, trange, verbose=False, priors=None, p0=None, maxit=50000, svdcut=0., **data_kwargs):
        xdata, ydata = self.format(trange=trange, flatten=True, **data_kwargs)

        def model(xdata,pdict):
            return np.concatenate(
                [NplusN2ptModel(Nstates,self.Nt,smr,pol)(xdata,pdict) for smr,pol in self.keys]
            )            

        if verbose:
            print(f'---------- {Nstates}+{Nstates} fit in {trange} for mes: {self.info.meson} of ens: {self.info.ensemble} for mom: {self.info.momentum} --------------')

        pr = priors if priors is not None else self.priors(Nstates)
        p0 = p0 if p0 is not None else gv.mean(pr)

        fit = lsqfit.nonlinear_fit(
            data   = (xdata,ydata),
            fcn    = model,
            prior  = pr,
            p0     = p0,
            maxit  = maxit,
            svdcut = svdcut
        )

        if verbose:
            print(fit)

        return fit

    def chi2exp(self, Nexc, trange, popt, fitcov, pvalue=True, Nmc=10000):
        # Format data and estimate covariance matrix
        xdata,ydata = self.format(trange=trange,flatten=False)
        yvec = np.concatenate([ydata[k] for k in self.keys])
        cov = gv.evalcov(yvec)

        # Estimate jacobian and hessian
        fun,jac,hes = self.diff_model(xdata,Nexc)
        j = jac(popt)
        Jac = np.hstack([j[k] for k in popt])

        # Calculate chi2 from fit
        w2 = jnp.linalg.inv(fitcov)
        res = gv.mean(yvec) - fun(popt)
        chi2 = res.T @ w2 @ res

        # Calculate expected chi2
        w = np.linalg.pinv(fitcov)
        Hmat = Jac.T @ w @ Jac
        Hinv = np.linalg.pinv(Hmat)
        wg = w @ Jac
        proj = w - wg @ Hinv @ wg.T

        chiexp = np.trace(proj @ cov)


        if not pvalue:
            return chiexp
        
        else:
            l,evec = np.linalg.eig(cov)

            if np.any(np.real(l))<0:
                mask = np.real(l)>1e-12
                Csqroot = evec[:,mask] @ np.diag(np.sqrt(l[mask])) @ evec[:,mask].T
            else:
                Csqroot= evec @ np.diag(np.sqrt(l)) @ evec.T

            numat = Csqroot @ proj @ Csqroot
            l,_ = np.linalg.eig(numat)
            ls = l[l.real>=1e-14].real

            p = 0
            for _ in range(Nmc):
                ri = np.random.normal(0.,1.,len(ls))
                p += 1. if (ls.T @ (ri**2) - chi2)>=0 else 0.
            p /= Nmc
            
            return chi2, chiexp, p
        

def main():
    jax.config.update("jax_enable_x64", True)
    x = jax.random.uniform(jax.random.PRNGKey(0), (1000,))#, dtype=jnp.float64)
    assert x.dtype == jnp.float64


    ens      = 'Coarse-1'
    mes      = 'Dst'
    mom      = '200'
    binsize  = 11
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    TLIM     = (15,20)
    NEXC     = 2

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    x,y,yall = stag.format(trange=TLIM, alljk=True, flatten=True)



    # ========================================= fit =============================================
    cut = correlation_diagnostics(yall)

    meff,aeff = stag.meff(trange=(15,20))
    pr = stag.priors(NEXC,Meff=meff,Aeff=aeff)
    cov_specs = dict(
        diag   = False,
        block  = False,
        scale  = True,
        shrink = True,
        cutsvd = None,
    )

    fit = stag.fit(
        Nstates = NEXC,
        trange  = TLIM,
        priors  = pr, 
        verbose = True,
        **cov_specs
    )
    # # ===========================================================================================

    xdata,ydata =  stag.format(trange=TLIM,flatten=True,**cov_specs)

    chi2, ce,p = stag.chi2exp(
        Nexc   = NEXC,
        trange = TLIM,
        popt   = dict(fit.pmean),
        fitcov = gv.evalcov(ydata),
        pvalue = True
    )
    print(chi2/ce, ce,p)
