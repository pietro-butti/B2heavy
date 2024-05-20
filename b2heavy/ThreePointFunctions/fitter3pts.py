import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)

import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import lsqfit

from tqdm       import tqdm

from .types3pts                import Ratio, RatioIO
from .utils                    import read_config_fit
from ..TwoPointFunctions.utils import p_value

def ModelRatio_jax(T,rsp,sm,Nexc):
    simple = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 
    dE_k2 = 'dE_src' if simple else 'dE_snk'

    def _model(t,p):
        tmp = jnp.full(len(t),1.)
        for iexc in range(Nexc):
            tmp = tmp + p[f'A_{sm}'][iexc] * jnp.exp(- jnp.exp(p['dE_src'][iexc]) * (t)) + \
                        p[f'B_{sm}'][iexc] * jnp.exp(- jnp.exp(p[dE_k2][iexc]) * (T-t))
        tmp = p['ratio'][0]*tmp
        return tmp

    return _model

def ModelRatio(T,rsp,sm,Nexc):
    simple = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 
    dE_k2 = 'dE_src' if simple else 'dE_snk'

    def _model(t,p):
        tmp = np.full(len(t),1.)
        for iexc in range(Nexc):
            tmp = tmp + p[f'A_{sm}'][iexc] * np.exp(- np.exp(p['dE_src'][iexc]) * (t)) + \
                        p[f'B_{sm}'][iexc] * np.exp(- np.exp(p[dE_k2][iexc]) * (T-t))
        tmp = p['ratio'][0]*tmp
        return tmp

    return _model


def phys_energy_priors(ens, mes, mom, nstates, readfrom=None, error=1.0):
    fit,p = read_config_fit(
        tag  = f'fit2pt_config_{ens}_{mes}_{mom}',
        path = readfrom
    )

    # Check length
    dE = []
    for n in range(nstates):
        try:
            pr = gv.gvar(
                p['E'][2*n+2].mean,
                # p['E'][2*n+2].sdev*3
                error
            )
        except IndexError:
            pr = gv.gvar('-1.5(1.0)')

        dE.append(pr)

    return dE


def standard_p(io, fit:lsqfit.nonlinear_fit):
    chi2red = fit.chi2
    for k,pr in fit.prior.items():
        for i,p in enumerate(pr):
            chi2red -= ((fit.pmean[k][i]-p.mean)/p.sdev)**2
    
    ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 

    req = {
        'E0'     : 1.,
        'm0'     : 1.,
        'Z0'     : {'1S': 1., 'RW': 1.},
        'Zpar'   : {'1S': 1., 'RW': 1.},
        'Zbot'   : {'1S': 1., 'RW': 1.},
        'wrecoil': 1.
    }
    aux = Ratio(io=io,jkBin=0,**req)
    nconf = list(aux.data.values())[0].shape[0]

    return p_value(chi2red,nconf,ndof)


class RatioFitter(Ratio):
    def __init__(self, io:RatioIO, **kwargs):
        super().__init__(io,**kwargs)
        self.Ta, self.Tb = io.mData['hSinks']
        self.fits = {}


    def priors(self, Nstates, K=None, dE_src=None, dE_snk=None):
        rsp = self.specs
        single = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 

        pr = {
            'ratio' : [gv.gvar(-0.9,0.1 if self.info.ratio=='RA1' else 0.05) if K is None else K],
            'dE_src': [gv.gvar(-1.5,1.0) for _ in range(Nstates)] if dE_src is None else dE_src,
        }
        if not single:
            pr['dE_snk'] = [gv.gvar(-1.5,1.0) for _ in range(Nstates)] if dE_snk is None else dE_snk

        for sm in self.smr:
            pr[f'A_{sm}'] = [gv.gvar('0(1)') for _ in range(Nstates)]
            pr[f'B_{sm}'] = [gv.gvar('0(1)') for _ in range(Nstates)]

        return pr


    def fit(self, Nstates, trange, verbose=True, priors=None, p0=None, svdcut=0., jkfit=False, **data_kwargs):
        # Format data
        xdata, ydata = self.format(
            trange  = trange, 
            flatten = True, 
            **data_kwargs
        )

        # Prepare the model
        def _model(xdata,pdict):
            return np.concatenate([
                ModelRatio(self.Tb,self.specs,sm,Nstates)(xdata,pdict)
                for sm in self.smr
            ])

        if verbose:
            print(f'---------- {Nstates}+{Nstates} fit in {trange} for mes: {self.info.ratio} of ens: {self.info.ensemble} for mom: {self.info.momentum} --------------')

        # Prepare priors
        pr = self.priors(Nstates) if priors is None else priors

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data   = (xdata,ydata),
            fcn    = _model,
            prior  = pr,
            maxit  = 50000,
            svdcut = svdcut
        )
        self.fits[Nstates,trange] = fit

        if verbose:
            print(fit)

        if jkfit:
            # get info for standard p-value
            ndof  = len(fit.y) - sum([len(pr) for k,pr in fit.prior.items()]) 
            aux = Ratio(io=self.io,jkBin=0)
            nconf = aux.nbins

            xdata, ydata, ally = self.format(
                trange  = trange, 
                flatten = True, 
                alljk   = True,
                **data_kwargs
            )   
            cov = gv.evalcov(ydata)         

            pkeys = sorted(fit.p.keys())
            fitpjk = []
            pstds  = []
            for ijk in tqdm(range(ally.shape[0])):
                ydata = gv.gvar(ally[ijk,:],cov)
                fit = lsqfit.nonlinear_fit(
                    data   = (xdata,ydata),
                    fcn    = _model,
                    prior  = pr,
                    maxit  = 50000,
                    svdcut = svdcut
                )                

                fitpjk.append(fit.pmean)

                # calculate p-value
                chi_pr = 0.
                for k, _pr in fit.prior.items():
                    dr = ((gv.mean(_pr) - fit.pmean[k])/gv.sdev(_pr))**2
                    chi_pr += dr.sum()
                chi2red = fit.chi2 - chi_pr
                pvalue = p_value(chi2red,nconf,ndof)
                pstds.append(pvalue)


            fitjk = {k: np.asarray([jf[k] for jf in fitpjk]).T for k in pkeys}
            fitjk['pstd'] = pstds

        return fitjk if jkfit else fit


    def plot_fit(self,ax,Nstates,trange,color='C0',color_res='crimson',alpha=1.):
        fit = self.fits[Nstates,trange]
        x,y = self.format()

        mrk = ['o','^']

        for i,sm in enumerate(self.smr):
            model = ModelRatio(self.Tb,self.specs,sm,Nstates)

            iin = np.array([min(trange)<=x<=max(trange) for x in x])

            # label = None if label is None else f'{label} [{sm}]'
            
            off = (-1)**i*0.05
            # Plot fit points 
            xin = x[iin]
            yin = y[sm][iin]
            ax.errorbar(xin+off,gv.mean(yin),gv.sdev(yin),fmt=mrk[i%2], ecolor=color, mfc='w', color=color, capsize=2.5,alpha=alpha)
            
            # Plot other points
            xout = x[~iin]
            yout = y[sm][~iin]
            ax.errorbar(xout+off,gv.mean(yout),gv.sdev(yout),fmt=mrk[i%2], ecolor=color, mfc='w', color=color, capsize=2.5, alpha=alpha/5)
            
            # fit bands
            xrange = np.arange(-1.,max(x)+1,0.01)
            ye = model(xrange,fit.p)
            ax.fill_between(xrange,gv.mean(ye)+gv.sdev(ye),gv.mean(ye)-gv.sdev(ye),color=color,alpha=alpha/6)

            # results
            ax.errorbar(-0.25,fit.p['ratio'][0].mean,fit.p['ratio'][0].sdev,color=color_res,fmt='D', capsize=2.5)


    def diff_model(self, xdata, Nstates):
        def _model(pdict):
            return jnp.concatenate([
                ModelRatio_jax(self.Tb,self.specs,sm,Nstates)(xdata,pdict)
                for sm in self.smr
            ])
        
        def _jac(pdict):
            return jax.jacfwd(_model)(pdict)

        def _hes(pdict):
            return jax.jacfwd(jax.jacrev(_model))(pdict)
            
        return _model, _jac, _hes


    def chi2exp(self, Nexc, trange, popt, fitcov, priors=None, pvalue=True, Nmc=10000, verbose=True, method='eigval'):
        # Format data and estimate covariance matrix
        xdata,ydata = self.format(trange=trange)
        yvec = np.concatenate([ydata[k] for k in self.smr])
        cov = gv.evalcov(yvec)

        pkeys = sorted(popt.keys())

        # Estimate jacobian
        fun,jac,_ = self.diff_model(xdata,Nexc)
        _jc = jac(popt)
        Jac = np.hstack([_jc[k] for k in pkeys])

        # Check chi2 from fit
        w2 = np.linalg.inv(fitcov)
        res = gv.mean(yvec) - fun(popt)
        chi2 = float(res.T @ w2 @ res)

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

        chiexp = np.trace(proj @ cov)
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
        aux = Ratio(io=self.io,jkBin=0)
        nconf = aux.format(flatten=True,alljk=True)[-1].shape[0]
        pvalue = p_value(chi2,nconf,ndof)

        return dict(chi2=chi2, chi2_pr=chi_pr, chi2_aug=chi2_aug, pstd=pvalue)


    def fit_result(self,Nexc,trange,verbose=True, priors=None):
        fit = self.fits[Nexc,trange]
        xdata  = fit.x
        yvec   = gv.mean(fit.y)
        fitcov = gv.evalcov(fit.y)
        popt   = dict(fit.pmean)

        _, chiexp, p = self.chi2exp(
            Nexc, 
            trange, 
            popt, 
            fitcov, 
            pvalue=True, 
            priors=priors
        )

        chi2dict = self.chi2(Nexc,trange)

        if verbose:
            print(f'# ---------- {Nexc}+{Nexc} fit in {trange} for mes: {self.info.ratio} of ens: {self.info.ensemble} for mom: {self.info.momentum} --------------')
            print(fit)
            print(f'# red chi2       = {chi2dict["chi2"]:.2f}')
            print(f'# aug chi2       = {chi2dict["chi2_aug"]:.2f}')
            print(f'# chi2_exp       = {chiexp:.2f}')
            print(f'# chi2/chi_exp   = {chi2dict["chi2_aug"]/chiexp:.2f}')
            print(f'# p-value (exp)  = {p:.2f}')
            print(f'# p-value (std)  = {chi2dict["pstd"]:.2f}')


        res = dict(
            fit     = fit,
            chi2red = chi2dict['chi2'],
            chi2aug = chi2dict['chi2_aug'],
            chiexp  = chiexp,
            pexp    = p,
            pstd    = chi2dict['pstd'],
        )

        return res



from b2heavy.ThreePointFunctions.types3pts import ratio_prerequisites

def main():
    ens = 'Coarse-1'
    rat = 'xfstpar'
    mom = '100'
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex'
    readfrom = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/report'

    requisites = ratio_prerequisites(
        ens, rat, mom, readfrom=readfrom, jk=True
    )

    io = RatioIO(ens,rat,mom,PathToDataDir=frm)
    ratio = RatioFitter(
        io,
        jkBin    = 11,
        smearing = ['1S','RW'],
        **requisites
    )

    COV_SPECS = dict(
        diag   = False,
        block  = False,
        scale  = True,
        shrink = True,
        cutsvd = 1E-12
    )

    nstates = 1
    trange  = (1,ratio.Ta-1-1)


    x,y = ratio.format()
    kmean = np.mean([y[sm][ratio.Ta//2].mean for sm in ratio.smr])
    Kmean = gv.gvar(kmean,0.1)

    dE_src = phys_energy_priors(ens,'Dst',mom,nstates,readfrom=readfrom)
    dE_snk = phys_energy_priors(ens,'B'  ,mom,nstates,readfrom=readfrom)


    priors = ratio.priors(nstates,K=Kmean,dE_src=dE_src,dE_snk=dE_snk)

    fit = ratio.fit(
        Nstates = nstates,
        trange  = trange,
        priors  = priors,
        verbose = False,
        **COV_SPECS
    )

    ratio.fit_result(nstates,trange,priors=priors)

    f, ax = plt.subplots(1,1,figsize=(6,6))
    ratio.plot_fit(ax,nstates,trange)
    plt.show()