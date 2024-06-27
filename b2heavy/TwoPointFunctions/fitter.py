import os
import itertools
import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)

import pandas            as pd
import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import lsqfit
from   tqdm import tqdm

from .utils     import p_value, Tmax, load_toml, MPHYS
from .types2pts import CorrelatorIO, Correlator



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
                # priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev*100)
                priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,1.0)

    return priors  




def fexp(Nt):
    return lambda t,E,Z: Z * ( np.exp(-E*t) + np.exp(-E*(Nt-t)) )

def corr2(Nstates,Nt,smr,pol):
    s1,s2 = smr.split('-')
    mix = s1!=s2

    def _fcn(t,p):
        erg    = np.exp(p[f'dE'  ])
        ergo   = np.exp(p[f'dE.o'])
        erg [0] = p[f'dE'][0]
        ergo[0] = erg[0] + ergo[0]

        c2 = 0.
        for n in range(Nstates):
            Z0 = np.exp(p[f'Z.{s1}.{pol}'  ][n]) * np.exp(p[f'Z.{s2}.{pol}'  ][n])
            Z1 = np.exp(p[f'Z.{s1}.{pol}.o'][n]) * np.exp(p[f'Z.{s2}.{pol}.o'][n])

            if n>0:
                Z0 = p[f'Z.{smr if mix else s1}.{pol}'  ][n-1 if mix else n]**2
                Z1 = p[f'Z.{smr if mix else s1}.{pol}.o'][n-1 if mix else n]**2

            Ephy = sum(erg[ :n+1])
            Eosc = sum(ergo[:n+1])

            c2 += fexp(Nt)(t,Ephy,Z0) + fexp(Nt)(t,Eosc,Z1) * (-1)**((t+1))

        return c2

    return _fcn


def fexp_jax(Nt):
    return lambda t,E,Z: Z * ( jnp.exp(-E*t) + jnp.exp(-E*(Nt-t)) )

def corr2_jax(Nstates,Nt,smr,pol):
    s1,s2 = smr.split('-')
    mix = s1!=s2

    def _fcn(t,p):
        c2 = 0.

        Ephy = p['dE'][0]
        Eosc = p['dE'][0] + jnp.exp(p['dE.o'][0])
        for n in range(Nstates):
            Z0 = jnp.exp(p[f'Z.{s1}.{pol}'  ][n]) * jnp.exp(p[f'Z.{s2}.{pol}'  ][n])
            Z1 = jnp.exp(p[f'Z.{s1}.{pol}.o'][n]) * jnp.exp(p[f'Z.{s2}.{pol}.o'][n])

            if n>0:
                Z0 = p[f'Z.{smr if mix else s1}.{pol}'  ][n-1 if mix else n]**2
                Z1 = p[f'Z.{smr if mix else s1}.{pol}.o'][n-1 if mix else n]**2

                Ephy += jnp.exp(p['dE'  ][n])
                Eosc += jnp.exp(p['dE.o'][n])

            c2 += fexp_jax(Nt)(t,Ephy,Z0) + fexp_jax(Nt)(t,Eosc,Z1) * (-1)**((t+1))


        return c2

    return _fcn


def standard_p(io, fit:lsqfit.nonlinear_fit):
    chi2red = fit.chi2
    for k,pr in fit.prior.items():
        for i,p in enumerate(pr):
            chi2red -= ((fit.pmean[k][i]-p.mean)/p.sdev)**2
    
    ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 

    aux = Correlator(io=io,jkBin=0)
    nconf = list(aux.data.values())[0].shape[0]

    return p_value(chi2red,nconf,ndof)



def expose(fitp):
    aux,idx = [],[]
    for k in fitp:
        if k.endswith('o'):
            continue
        if k=='pstd':
            continue

        tmp = {}
        for n in range(len(fitp[k])):
            if k.startswith('Z') and n==0:
                fcn = lambda x:np.exp(x)
            else:
                fcn = lambda x:x

            if isinstance(fitp[k][n],np.ndarray):
                pass #FIXME
                # jp  = fcn(fitp[k       ][:,n])
                # jpo = fcn(fitp[f'{k}.o'][:,n])

                # tmp[f'{n+1 if "-" in k else n}']   = gv.gvar(jp.mean() , jp.std()*np.sqrt(len(jp)-1))
                # tmp[f'{n+1 if "-" in k else n}.o'] = gv.gvar(jpo.mean(), jpo.std()*np.sqrt(len(jpo)-1))
            else:
                tmp[f'{n+1 if "-" in k else n}']   = fcn(fitp[k][n])
                tmp[f'{n+1 if "-" in k else n}.o'] = fcn(fitp[f'{k}.o'][n])

        aux.append(tmp)
        idx.append(k)

    return pd.DataFrame(aux,index=idx).transpose()



def par_to_new(p,smearing,polarization):
    pr = {}
    pr['dE']    = p['E'][::2]  
    pr['dE.o']  = p['E'][1::2]
    pr['dE'][0] = p['E'][0]

    for sm in smearing:
        for pol in polarization:
            sm1,sm2 = sm.split('-')
            smr = sm1 if sm1==sm2 else sm

            pr[f'Z.{smr}.{pol}']   = p[f'Z_{smr}_{pol}'][::2] 
            pr[f'Z.{smr}.{pol}.o'] = p[f'Z_{smr}_{pol}'][1::2]
    
    return pr






class StagFitter(Correlator):
    def __init__(self, io:CorrelatorIO, priors_policy=None, **kwargs):
        super().__init__(io,**kwargs)
        self.priors_policy = priors_policy
        self.fits  = {}
        self.tmaxs = {}

        self.nconf = len(io.ReadCorrelator().jkbin)

    def priors(self,Nstates,Meff=None, Aeff=None):
        tmp = set_priors_phys(self,Nstates,Meff=Meff,Aeff=Aeff)

        smr = self.data.smearing.to_numpy()
        pol = self.data.polarization.to_numpy()
        pr = par_to_new(tmp,smr,pol)

        return pr

    # def priors(self, Nstates, Meff=None, Aeff=None):
    #     tmp = {}

    #     tmp['dE']    = ['-0.2(1.4)'] * Nstates
    #     tmp['dE.o']  = ['-0.2(1.4)'] * Nstates
    #     tmp['dE'][0] = '1(1)' if Meff is None else gv.gvar(Meff.mean,0.5) 

    #     for sm,pl in self.keys:
    #         s1,s2 = sm.split('-')
    #         mix = s1!=s2

    #         if s1==s2:
    #             tmp[f'Z.{s1}.{pl}'  ]    = ['0.2(3.5)'] * Nstates
    #             tmp[f'Z.{s1}.{pl}.o']    = ['0.2(3.5)'] * Nstates

    #             # fundamental
    #             a = Aeff[sm,pl].mean
    #             absp = np.sqrt(sum((np.array(self.io.pmom)*2*np.pi / self.io.mData['L'])**2))
    #             err = 1.*(1+2*self.io.mData['alphaS']*absp)
    #             tmp[f'Z.{s1}.{pl}'][0] = gv.gvar(np.log(a)/2,err)

    #             # fund. oscillating
    #             tmp[f'Z.{s1}.{pl}.o'][0] = gv.gvar('0.2(1.2)')

    #         else:
    #             tmp[f'Z.{sm}.{pl}'  ] = ['0.2(3.5)'] * (Nstates-1)
    #             tmp[f'Z.{sm}.{pl}.o'] = ['0.2(3.5)'] * (Nstates-1)

    #     return gv.gvar(tmp)


    def seek_tmax(self,errmax,**cov_specs):
        x, ydata = self.format(**cov_specs)
        tmax = {}
        for smr,pol in self.keys:
            tmax[smr,pol] = Tmax(ydata[smr,pol],errmax=errmax)
        return tmax


    def fitformat(self, trange, **cov_specs):
        tmin,tmax = trange
        xdata, ydata, yfull = self.format(alljk=True,**cov_specs)

        xfit, yfit, yjk = {},{},{}
        for smr,pol in self.keys:
            if isinstance(tmax,int):
                maxt = tmax  
            elif isinstance(tmax,dict):
                maxt = tmax[smr,pol]
            elif tmax<1.:
                maxt = Tmax(ydata[smr,pol],errmax=tmax)

            xfit[smr,pol] = xdata[tmin:(maxt+1)]
            yfit[smr,pol] = ydata[smr,pol][tmin:(maxt+1)]
            yjk [smr,pol] = yfull[smr,pol][:,tmin:(maxt+1)]

        yflat   = np.concatenate([yfit[k] for k in self.keys])
        yflatjk = np.hstack(      [yjk[k] for k in self.keys])

        return xfit, yflat, yflatjk


    def fit(self, Nstates, trange, priors=None, verbose=False, jkfit=False, boost=False, **cov_specs):        
        xfit, yflat, yflatjk = self.fitformat(trange, **cov_specs)

        if isinstance(trange[1],int):
            self.tmaxs[Nstates,trange] = trange[1]  
        elif isinstance(trange[1],dict):
            self.tmaxs[Nstates,trange] = trange[1]
        elif trange[1]<1.:
            self.tmaxs[Nstates,trange] = self.seek_tmax(trange[1],**cov_specs)

        def fitfcn(xd,p):
            tmp = []
            for sm,pl in self.keys:
                fcn = corr2(Nstates,self.Nt,sm,pl)
                tmp.append(fcn(xd[sm,pl], p))
            return np.concatenate(tmp)

        pr = self.priors(self,Nstates) if priors is None else priors

        if verbose:
            print(f'---------- {Nstates}+{Nstates} fit in {trange} for mes: {self.info.meson} of ens: {self.info.ensemble} for mom: {self.info.momentum} --------------')

        fit = lsqfit.nonlinear_fit(
            data = (xfit, yflat),
            fcn  = fitfcn,
            prior=pr
        )
        self.fits[Nstates,trange] = fit

        jpars = []
        jpval = []
        if jkfit:
            ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()])

            cov = gv.evalcov(yflat)
            jpr = gv.gvar(fit.pmean,gv.sdev(pr)) if boost else pr

            for ijk in tqdm(range(yflatjk.shape[0])):
                # fit function
                jfit = lsqfit.nonlinear_fit(
                    data  = (xfit, yflatjk[ijk,:],cov),
                    fcn   = fitfcn,
                    prior = jpr, 
                )
                jpars.append(jfit.pmean)

                # calculate p-value
                chi_pr = 0.
                for k,_pr in jfit.prior.items():
                    dr = ((gv.mean(_pr) - jfit.pmean[k])/gv.sdev(_pr))**2
                    chi_pr += dr.sum()
                chi2red = jfit.chi2 - chi_pr                
                pvalue = p_value(chi2red,self.nconf,ndof)
                jpval.append(pvalue)

            pkeys = sorted(fit.p.keys())
            fitjk = {k: np.asarray([jf[k] for jf in jpars]) for k in pkeys}
            fitjk['pstd'] = jpval

        return fitjk if jkfit else fit


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
            tmp = []
            for sm,pl in self.keys:
                fcn = corr2_jax(Nstates,self.Nt,sm,pl)
                tmp.append(fcn(xdata[sm,pl], pdict))
            return jnp.concatenate(tmp)

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


    def chi2exp(self, Nstates, trange, popt, fitcov, priors=None, pvalue=True, Nmc=10000, method='eigval', verbose=False):
        # Format data and estimate covariance matrix
        tmax = self.tmaxs[Nstates,trange]
        xfit, yflat, yflatjk = self.fitformat((trange[0],tmax))
        cov = gv.evalcov(yflat)

        pkeys = sorted(popt.keys())

        # Estimate jacobian
        fun,jac,_ = self.diff_model(xfit,Nstates)
        _jc  = jac(popt)
        Jac = np.hstack([_jc[k] for k in pkeys])

        # Check chi2 from fit
        w2 = np.linalg.inv(fitcov)
        res = gv.mean(yflat) - fun(popt)
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
        pvalue = p_value(chi2,self.nconf,ndof)

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
        # FIXME
        fit = self.fits[Nexc,trange]
        xdata,ydata = self.format()

        # for i,k in enumerate(self.keys):
        for i,sm in enumerate(self.smr):
            for j,pl in enumerate(self.pol):
                model = corr2(Nexc,self.Nt,sm,pl)

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




def main():
    ens      = 'Coarse-1'
    mes      = 'Dst'
    mom      = '200'
    binsize  = 11
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    Nstates = 3
    trange  = (5,0.3)

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    cov_specs = dict(scale=True,shrink=True,cutsvd=1e-12)

    effm,effa = stag.meff(trange=(15,25))
    pr = stag.priors(Nstates,Meff=effm,Aeff=effa)

    fit = stag.fit(
        Nstates = Nstates,
        trange  = trange,
        priors  = pr,    
        # jkfit   = True,
        # boost   = True,
        **cov_specs)

    breakpoint()

    stag.chi2exp(Nstates,trange,dict(fit.pmean),gv.evalcov(fit.y),priors=fit.prior)

    # stag.fit_result(Nstates,trange,verbose=True)

if __name__=='__main__':
    main()