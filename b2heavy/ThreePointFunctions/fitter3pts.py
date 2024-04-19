import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)

import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import lsqfit


from .types3pts import Ratio, RatioIO


def ModelRatio_jax(T,rsp,sm,Nexc):
    single = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 
    dE_k2 = 'dE_src' if single else 'dE_snk'

    def _model(t,p):
        tmp = jnp.full(len(t),1.)
        for iexc in range(Nexc):
            tmp = tmp + p[f'A_{sm}'][iexc] * jnp.exp(- jnp.exp(p['dE_src'][iexc]) * (t)) + \
                        p[f'B_{sm}'][iexc] * jnp.exp(- jnp.exp(p[dE_k2][iexc]) * (T-t))
        tmp = p['ratio'][0]*tmp
        return tmp

    return _model

def ModelRatio(T,rsp,sm,Nexc):
    single = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 
    dE_k2 = 'dE_src' if single else 'dE_snk'

    def _model(t,p):
        tmp = np.full(len(t),1.)
        for iexc in range(Nexc):
            tmp = tmp + p[f'A_{sm}'][iexc] * np.exp(- np.exp(p['dE_src'][iexc]) * (t)) + \
                        p[f'B_{sm}'][iexc] * np.exp(- np.exp(p[dE_k2][iexc]) * (T-t))
                        # p[f'{"A" if single else "B"}_{sm}'][iexc] * np.exp(- np.exp(p[dE_k2][iexc]) * (T-t))
        tmp = p['ratio'][0]*tmp
        return tmp

    return _model



class RatioFitter(Ratio):
    def __init__(self, io:RatioIO, **kwargs):
        super().__init__(io,**kwargs)
        self.Ta, self.Tb = io.mData['hSinks']
        self.fits = {}

    def priors(self, Nstates, K=None, dE_src=None, dE_snk=None):
        rsp = self.specs
        single = (not isinstance(rsp['source'],list)) and rsp['source']==rsp['sink'] 

        pr = {
            'ratio' : [gv.gvar(-0.9,0.5) if K is None else K],
            'dE_src': [gv.gvar(-1.5,1.0) for _ in range(Nstates)] if dE_src is None else dE_src,
            # 'log(dE_src)': [gv.log(gv.gvar('0.22(22)')) for _ in range(Nstates)] if dE_src is None else dE_src,
        }
        if not single:
            pr['dE_snk'] = [gv.gvar(-1.5,1.0) for _ in range(Nstates)] if dE_snk is None else dE_snk
            # pr['log(dE_snk)'] = [gv.log(gv.gvar('0.22(22)')) for _ in range(Nstates)] if dE_snk is None else dE_snk

        for sm in self.smr:
            pr[f'A_{sm}'] = [gv.gvar('0(1)') for _ in range(Nstates)]
            pr[f'B_{sm}'] = [gv.gvar('0(1)') for _ in range(Nstates)]

        return pr

    def fit(self, Nstates, trange, verbose=True, priors=None, p0=None, svdcut=0., **data_kwargs):
        # Format data
        xdata, ydata = self.format(
            trange  = trange, 
            flatten = True, 
            **data_kwargs
        )

        # Prepare the model
        def _model(xdata,pdict):
            return [
                ModelRatio(self.Tb,self.specs,sm,Nstates)(xdata,pdict)
                for sm in self.smr
            ]

        # Prepare priors
        pr = self.priors(Nstates) if priors is None else priors
        # p0 = p0 if p0 is not None else gv.mean(pr)

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data   = (xdata,ydata),
            fcn    = _model,
            prior  = pr,
            # p0     = p0,
            maxit  = 50000,
            svdcut = svdcut
        )
        self.fits[Nstates,trange] = fit

        if verbose:
            print(fit)

        return fit

    def plot_fit(self,ax,Nstates,trange):
        fit = self.fits[Nstates,trange]
        x,y = self.format()

        for sm in self.smr:
            model = ModelRatio(self.Tb,self.specs,sm,Nstates)

            iin = np.array([min(trange)<=x<=max(trange) for x in x])
            
            # Plot fit points 
            xin = x[iin]
            yin = y[sm][iin]
            ax.errorbar(xin,gv.mean(yin),gv.sdev(yin),fmt='o', ecolor='C0', mfc='w', capsize=2.5, label=sm)
            
            # Plot other points
            xout = x[~iin]
            yout = y[sm][~iin]
            ax.errorbar(xout,gv.mean(yout),gv.sdev(yout),fmt='o', ecolor='C0', mfc='w', color='C0', capsize=2.5, alpha=0.2)
            

            xrange = np.arange(0,max(x)+1,0.01)
            ye = model(xrange,fit.p)
            ax.fill_between(xrange,gv.mean(ye)+gv.sdev(ye),gv.mean(ye)-gv.sdev(ye),alpha=0.3)

            ax.legend()  

            ax.set_xlim(-0.1,max(x)+1)          

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

    def chi2exp(self, Nexc, trange, popt, fitcov, priors=None, pvalue=True, Nmc=10000):
        # Format data and estimate covariance matrix
        xdata,ydata = self.format(trange=trange,flatten=False)
        yvec = np.concatenate([ydata[k] for k in self.smr])
        cov = gv.evalcov(yvec)

        pkeys = sorted(popt.keys())

        # Estimate jacobian
        fun,jac,_ = self.diff_model(xdata,Nexc)
        _jc = jac(popt)
        Jac = np.hstack([_jc[k] for k in pkeys])

        # Check chi2 from fit
        w2 = jnp.linalg.inv(fitcov)
        res = gv.mean(yvec) - fun(popt)
        chi2 = res.T @ w2 @ res

        # Add prior part if needed
        if priors is not None:
            # Enlarge jacobian
            Jac = np.vstack([Jac,-np.eye(Jac.shape[-1])])

            # Enlarge chi2
            prio_v = np.concatenate([priors[k] for k in pkeys])
            popt_v = np.concatenate([popt[k]   for k in pkeys])
            r      = (gv.mean(prio_v) - popt_v)**2

            dpr   = np.diag(1/gv.sdev(prio_v))
            chi2 += r.T @ dpr @ r

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

        if not pvalue:
            return chi2, chiexp
        
        else:
            l,evec = np.linalg.eig(cov)

            if np.any(np.real(l)<0):
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
    ens = 'Coarse-1'
    rat = 'RA1'
    mom = '000'
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex'

    io = RatioIO(ens,rat,mom,PathToDataDir=frm)
    ratio = RatioFitter(
        io,
        jkBin    = 11,
        smearing = ['1S']
    )
