import os
import itertools
import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)

import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import lsqfit
from   tqdm import tqdm

from .utils     import MPHYS, load_toml, NplusN2ptModel, correlation_diagnostics, p_value
from .types2pts import CorrelatorInfo, CorrelatorIO, Correlator, plot_effective_coeffs



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
        E[0] = gv.gvar(Meff.mean,dE/dE_E0err*Scale*aGeV)

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
                # priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev*1000/2)
                priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev*100)

    return priors  


def standard_p(io, fit:lsqfit.nonlinear_fit):
    chi2red = fit.chi2
    for k,pr in fit.prior.items():
        for i,p in enumerate(pr):
            chi2red -= ((fit.pmean[k][i]-p.mean)/p.sdev)**2
    
    ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 

    aux = Correlator(io=io,jkBin=0)
    nconf = list(aux.data.values())[0].shape[0]

    return p_value(chi2red,nconf,ndof)





class StagFitter(Correlator):
    def __init__(self, io:CorrelatorIO, priors_policy=None, **kwargs):
        super().__init__(io,**kwargs)
        self.priors_policy = priors_policy
        self.fits = {}


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


    def diff_cost(self, Nexc, xdata, W2):
        '''
            Return a `jax`-differentiable cost function.

            Parameters
            ----------
                - Nexc: int. Number of excited states to be fitted.
                - xdata: array. Vector of timeslices to be fitted.
                - W2: matrix. Weight matrix of the fit.
        '''

        model,_,_ = self.diff_model(xdata,Nexc)
        w2mat = jnp.asarray(W2)

        # Prepare model dependent on parameters and data
        def _cost(pars,ydata):
            res = model(pars) - ydata
            return jnp.dot(res,jnp.matmul(w2mat,res))

        def hess_par(popt,ydata):
            return jax.jacfwd(jax.jacrev(_cost))(popt,ydata)

        def hess_mix(popt,ydata):
            return jax.jacfwd(jax.jacrev(_cost,argnums=1))(popt,ydata)

        return _cost, hess_par, hess_mix


    def fit(self, Nstates, trange, verbose=False, priors=None, p0=None, maxit=50000, svdcut=0., jkfit=False, **data_kwargs):
        xdata, ydata = self.format(
            trange  = trange, 
            flatten = True, 
            **data_kwargs
        )

        def model(xdata,pdict):
            return np.concatenate([
                NplusN2ptModel(Nstates,self.Nt,smr,pol)(xdata,pdict) for smr,pol in self.keys
            ])            

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
        self.fits[Nstates,trange] = fit

        if verbose:
            print(fit)


        if jkfit:
            # get info for standard p-value
            ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 
            aux = Correlator(io=self.io,jkBin=0)
            nconf = len(aux.data.jkbin)


            xdata, ydata, ally = self.format(trange=trange, flatten=True, alljk=True, **data_kwargs)
            cov = gv.evalcov(ydata)

            pkeys = sorted(fit.p.keys())
            fitpjk = []
            pstds  = []
            for ijk in tqdm(range(ally.shape[0])):
                ydata = gv.gvar(ally[ijk,:],cov)

                jfit = lsqfit.nonlinear_fit(
                    data   = (xdata,ydata),
                    fcn    = model,
                    prior  = pr,
                    p0     = p0,
                    maxit  = maxit,
                    svdcut = svdcut
                )
                fitpjk.append(jfit.pmean)


                # calculate p-value
                chi_pr = 0.
                for k,_pr in jfit.prior.items():
                    dr = ((gv.mean(_pr) - jfit.pmean[k])/gv.sdev(_pr))**2
                    chi_pr += dr.sum()
                chi2red = jfit.chi2 - chi_pr
                pvalue = p_value(chi2red,nconf,ndof)
                pstds.append(pvalue)

            fitjk = {k: np.asarray([jf[k] for jf in fitpjk]) for k in pkeys}
            fitjk['pstd'] = pstds

        return fitjk if jkfit else fit


    def fit_error(self, Nexc, xdata, yvec, fitcov, popt):
        pkeys = sorted(popt.keys())

        # Evaluate hessian for error propagation
        cost,hpar,hmix = self.diff_cost(Nexc,xdata,W2=jnp.linalg.inv(fitcov))
        fity = jnp.asarray(gv.mean(yvec))
        chi2 = cost(popt,fity)
        
        _hp = hpar(popt,fity) # Parameters hessian
        Hess_par = np.vstack([np.hstack([_hp[kr][kc] for kc in pkeys]) for kr in pkeys])
        
        _hm = hmix(popt,fity) # Mixed hessian
        Hess_mix = np.hstack([_hm[k] for k in pkeys]).T

        dpdy = -1. * np.linalg.pinv(Hess_par) @ Hess_mix
        pcov = dpdy @ fitcov @ dpdy.T

        pmean = np.concatenate([popt[k] for k in pkeys])
        pars = gv.gvar(pmean,pcov)

        return pars, chi2


    def chi2exp(self, Nexc, trange, popt, fitcov, priors=None, pvalue=True, Nmc=10000, method='eigval', verbose=False):
        # Format data and estimate covariance matrix
        xdata,ydata = self.format(trange=trange) # we want original data
        yvec = np.concatenate([ydata[k] for k in self.keys])
        cov = gv.evalcov(yvec)

        pkeys = sorted(popt.keys())

        # Estimate jacobian
        fun,jac,_ = self.diff_model(xdata,Nexc)
        _jc  = jac(popt)
        Jac = np.hstack([_jc[k] for k in pkeys])

        # Check chi2 from fit
        w2 = np.linalg.inv(fitcov)
        res = gv.mean(yvec) - fun(popt)
        chi2 = float(res.T @ w2 @ res) # FIXME: when using rescaling of covariance matrix, the data are averaged from the jkbin=1 data and therefore central value can change a bit. This results in a O(.5) difference in calculated chi2. 


        # Add prior part if needed
        if priors is not None:
            # Enlarge jacobian
            Jac = np.vstack([Jac,-np.eye(Jac.shape[-1])])

            # Enlarge chi2
            prio_v = np.concatenate([priors[k] for k in pkeys])
            popt_v = np.concatenate([popt[k]   for k in pkeys])
            r      = gv.mean(prio_v) - popt_v

            dpr   = np.diag(1/gv.sdev(prio_v))**2
            chi2 += float(r.T @ dpr @ r)

            # Augment covariance matrices            
            O = np.zeros((fitcov.shape[0],dpr.shape[1]))
            fcov = np.block([[fitcov,O], [O.T, dpr]])
            cov = np.block([[cov,O], [O.T,dpr]])
        else:
            fcov = fitcov

        # Calculate expected chi2
        w = np.linalg.inv(fcov)
        Hmat = Jac.T @ w @ Jac
        Hinv = np.linalg.pinv(Hmat)
        wg = w @ Jac
        proj = w - wg @ Hinv @ wg.T

        chiexp  = np.trace(proj @ cov)
        da = Hinv @ wg.T

        # Calculate nu matrix
        l,evec = np.linalg.eig(cov)
        if np.any(np.real(l)<0):
            mask = np.real(l)>1e-12
            Csqroot = evec[:,mask] @ np.diag(np.sqrt(l[mask])) @ evec[:,mask].T
        else:
            Csqroot= evec @ np.diag(np.sqrt(l)) @ evec.T
        numat = Csqroot @ proj @ Csqroot
        dchiexp = np.sqrt(2.*np.trace(numat @ numat))

        pvalue = {}
        # Compute it with eigenvalue spectrum
        l,_ = np.linalg.eig(numat)
        ls = l[l.real>=1e-14].real
        p = 0
        for _ in range(Nmc):
            ri = np.random.normal(0.,1.,len(ls))
            p += 1. if (ls.T @ (ri**2) - chi2)>=0 else 0.
        p /= Nmc
        pvalue['eigval'] = p

        # Compute it with MC sampling
        _n = cov.shape[0]
        z = np.random.normal(0.,1.,Nmc*_n).reshape(Nmc,_n)
        cexp = np.einsum('ia,ab,bi->i',z,numat,z.T)
        pvalue['MC'] = 1. - np.mean(cexp<chi2)

        if verbose:
            print(f'# ---------- chi^2_exp analysis -------------')
            print(f'# chi2_exp = {chiexp} +/- {dchiexp} ')
            print(f'# p-value [eval] = {pvalue["eigval"]}')
            print(f'# p-value [MC]   = {pvalue["MC"]}')


        if pvalue:
            return chi2, chiexp, pvalue[method]
        else:
            return chi2, chiexp


    def chi2(self, Nexc, trange):
        fit = self.fits[Nexc,trange]

        # normal chi2
        res = fit.fcn(fit.x,fit.pmean)-gv.mean(fit.y)
        chi2 = res.T @ np.linalg.inv(gv.evalcov(fit.y)) @ res

        # chi2 priors
        chi_pr = 0.
        for k, pr in fit.prior.items():
            dr = ((gv.mean(pr) - fit.pmean[k])/gv.sdev(pr))**2
            chi_pr += dr.sum()

        # chi2 augmented
        chi2_aug = chi2+chi_pr

        # standard p-value
        ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 
        aux = Correlator(io=self.io,jkBin=0)
        nconf = len(aux.data.jkbin)
        pvalue = p_value(chi2,nconf,ndof)

        return dict(chi2=chi2, chi2_pr=chi_pr, chi2_aug=chi2_aug, pstd=pvalue)


    def fit_result(self, Nexc, trange, verbose=True, error=False, priors=None):
        fit = self.fits[Nexc,trange]
        xdata  = fit.x
        yvec   = gv.mean(fit.y)
        fitcov = gv.evalcov(fit.y)
        popt   = dict(fit.pmean)

        chi2, chiexp, p = self.chi2exp(
            Nexc, 
            trange, 
            popt, 
            fitcov, 
            pvalue = True, 
            priors = priors
        )

        chi2 = self.chi2(Nexc,trange)

        if verbose:
            print(f'# ---------- {Nexc}+{Nexc} fit in {trange} for mes: {self.info.meson} of ens: {self.info.ensemble} for mom: {self.info.momentum} --------------')
            print(fit)

            print(f'# red chi2       = {chi2["chi2"]:.2f}')
            print(f'# aug chi2       = {chi2["chi2_aug"]:.2f}')
            print(f'# chi2_exp       = {chiexp:.2f}')
            print(f'# chi2/chi_exp   = {chi2["chi2"]/chiexp:.2f}')
            print(f'# p-value (exp)  = {p:.2f}')
            print(f'# p-value (std)  = {chi2["pstd"]:.2f}')

        res = dict(
            fit     = fit,
            chi2red = chi2['chi2'],
            chi2aug = chi2['chi2_aug'],
            chiexp  = chiexp,
            pexp    = p,
            pstd    = chi2['pstd'],
        )

        if error:
            pars, chi2cost  = self.fit_error(Nexc,xdata,yvec,fitcov, popt)
            # assert np.isclose(chi2cost,chi2,atol=1e-14)
            res['pars']   = pars


        return res


    def plot_fit(self,ax,Nexc,trange):
        fit = self.fits[Nexc,trange]
        xdata,ydata = self.format()

        # for i,k in enumerate(self.keys):
        for i,sm in enumerate(self.smr):
            for j,pl in enumerate(self.pol):
                model = NplusN2ptModel(Nexc,self.Nt,sm,pl)

                axi = ax[i,j] if len(self.pol)>1 else ax[i]

                iin = np.array([min(trange)<=x<=max(trange) for x in xdata])

                xin = xdata[iin]
                yin = ydata[sm,pl][iin]/model(xin,fit.pmean)

                axi.errorbar(xin,gv.mean(yin),gv.sdev(yin),fmt='o', ecolor='C0', mfc='w', capsize=2.5, label=(sm,pl))
                
                ye = gv.mean(ydata[sm,pl][iin])/model(xin,fit.p)
                axi.fill_between(xin,1+gv.sdev(ye),1-gv.sdev(ye),color='C1',alpha=0.3)

                axi.axhline(1 , color='gray',alpha=0.5, linestyle=':')
                axi.set_ylim(.95,1.05)
                axi.legend()





jax.config.update("jax_enable_x64", True)
x = jax.random.uniform(jax.random.PRNGKey(0), (1000,))#, dtype=jnp.float64)
assert x.dtype == jnp.float64


def main(FLAG):
    ens      = 'Coarse-Phys'
    mes      = 'Dst'
    mom      = '100'
    binsize  = 19
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    cov_specs = dict(
        diag   = False,
        block  = False,
        scale  = False,
        shrink = False,
        cutsvd = 0.1
    )

    TRANGE_EFF = (15,25) 
    TRANGE     = (5,25)

    effm,effa = stag.meff(TRANGE_EFF,verbose=True,**cov_specs)
    priors = stag.priors(3,Meff=effm,Aeff=effa)

    fit = stag.fit(
        Nstates = 3,
        trange  = TRANGE,
        priors  = priors,
        verbose = True,
        **cov_specs
    )



    xdata,_ = stag.format(trange=TRANGE)
    mod,jac,hes = stag.diff_model(xdata,3)

    xdata,ydata = stag.format(trange=TRANGE)
    yvec = np.concatenate([ydata[k] for k in stag.keys])

    # stag.chi2exp(
    #     Nexc = 3,
    #     trange = TRANGE,
    #     popt   = dict(fit.pmean),
    #     fitcov = gv.evalcov(fit.y),
    #     priors = fit.prior,
    # )

    res = stag.fit_result(3,TRANGE,error=True)



    breakpoint()


