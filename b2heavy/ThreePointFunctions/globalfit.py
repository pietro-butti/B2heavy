import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import pandas            as pd
import lsqfit
import itertools
from tqdm import tqdm

import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)

from ..FnalHISQMetadata         import params
from ..TwoPointFunctions.utils  import p_value
from .corr3pts                  import BINSIZE
from .types3pts                 import Ratio, RatioIO, ratio_prerequisites



def model_single2(t,p,Ta,mom,ratio,smr,src='dM_B',snk='dE_D'):
    return p[f'{ratio}_{mom}_f0'] + \
        p[f'{ratio}_{mom}_{smr}'][0] * np.exp(-(   t))**p[snk] + \
        p[f'{ratio}_{mom}_{smr}'][1] * np.exp(-(Ta-t))**p[src] 

model_ratio = lambda T,mom,smr : {
    'XF'     : lambda t,p:  model_single2(t,p,T,mom,'XF'     ,smr,src='dM_D',snk='dE_D'),
    'XFSTPAR': lambda t,p:  model_single2(t,p,T,mom,'XFSTPAR',smr,src='dM_D',snk='dE_D'),
    'RMINUS' : lambda t,p:  model_single2(t,p,T,mom,'RMINUS' ,smr,src='dM_B',snk='dE_D'),
    'RPLUS'  : lambda t,p:  model_single2(t,p,T,mom,'RPLUS'  ,smr,src='dM_B',snk='dE_D'), 
    'QPLUS'  : lambda t,p:  model_single2(t,p,T,mom,'QPLUS'  ,smr,src='dM_B',snk='dE_D'), 
    'ZRA1'   : lambda t,p:  model_single2(t,p,T,mom,'ZRA1'   ,smr,src='dM_B',snk='dE_D'),
    'RA1'    : lambda t,p:  model_single2(t,p,T,mom,'RA1'    ,smr,src='dM_B',snk='dE_D'),
    'XV'     : lambda t,p:  model_single2(t,p,T,mom,'XV'     ,smr,src='dM_B',snk='dE_D'),
    'R0'     : lambda t,p:  model_single2(t,p,T,mom,'R0'     ,smr,src='dM_B',snk='dE_D'),
    'R1'     : lambda t,p:  model_single2(t,p,T,mom,'R1'     ,smr,src='dM_B',snk='dE_D'),
}


def model_single2_jax(t,p,Ta,mom,ratio,smr,src='dM_B',snk='dE_D'):
    return p[f'{ratio}_{mom}_f0'] + \
        p[f'{ratio}_{mom}_{smr}'][0] * jnp.exp(-(   t))**p[snk] + \
        p[f'{ratio}_{mom}_{smr}'][1] * jnp.exp(-(Ta-t))**p[src] 

model_ratio_jax = lambda T,mom,smr : {
    'XF'     : lambda t,p:  model_single2_jax(t,p,T,mom,'XF'     ,smr,src='dM_D',snk='dE_D'),
    'XFSTPAR': lambda t,p:  model_single2_jax(t,p,T,mom,'XFSTPAR',smr,src='dM_D',snk='dE_D'),
    'RMINUS' : lambda t,p:  model_single2_jax(t,p,T,mom,'RMINUS' ,smr,src='dM_B',snk='dE_D'),
    'RPLUS'  : lambda t,p:  model_single2_jax(t,p,T,mom,'RPLUS'  ,smr,src='dM_B',snk='dE_D'), 
    'QPLUS'  : lambda t,p:  model_single2_jax(t,p,T,mom,'QPLUS'  ,smr,src='dM_B',snk='dE_D'), 
    'ZRA1'   : lambda t,p:  model_single2_jax(t,p,T,mom,'ZRA1'   ,smr,src='dM_B',snk='dE_D'),
    'RA1'    : lambda t,p:  model_single2_jax(t,p,T,mom,'RA1'    ,smr,src='dM_B',snk='dE_D'),
    'XV'     : lambda t,p:  model_single2_jax(t,p,T,mom,'XV'     ,smr,src='dM_B',snk='dE_D'),
    'R0'     : lambda t,p:  model_single2_jax(t,p,T,mom,'R0'     ,smr,src='dM_B',snk='dE_D'),
    'R1'     : lambda t,p:  model_single2_jax(t,p,T,mom,'R1'     ,smr,src='dM_B',snk='dE_D'),
}




def show(cases,params):
    df = []
    for mom,ratio in cases:
        df.append({
            'mom'  : mom,
            'ratio': ratio,
            'f0'   : params[f'{ratio}_{mom}_f0'],
            'A_1S' : params[f'{ratio}_{mom}_1S'] if f'{ratio}_{mom}_1S' in params else None,
            'A_RW' : params[f'{ratio}_{mom}_RW'] if f'{ratio}_{mom}_RW' in params else None,
            'B_1S' : params[f'{ratio}_{mom}_1S'] if f'{ratio}_{mom}_1S' in params else None,
            'B_RW' : params[f'{ratio}_{mom}_RW'] if f'{ratio}_{mom}_RW' in params else None,
        })
    return pd.DataFrame(df).set_index(['mom','ratio'])



class RatioSet:
    def __init__(self, ensemble, moms, rats, sms=['1S','RW']):
        self.ensemble    = ensemble
        self.mdata       = params(ensemble)
        self.Ta, self.Tb = self.mdata['hSinks']
        self.nconf       = self.mdata['nConfs']

        self.momlist = moms
        self.ratlist = rats
        self.smslist = sms

        self.cases = []
        self.keys  = []
        for mom,rat,sms in itertools.product(moms,rats,sms):
            if rat     in ['RPLUS','ZRA1'] and mom!='000' or \
               rat not in ['RPLUS','ZRA1'] and mom=='000':
                continue
            else:
                self.keys.append((mom,rat,sms))
                if (mom,rat) not in self.cases:
                    self.cases.append((mom,rat))

        self.objs = None

        return

    # read data
    def collect(self,datadir,datadir_2pt,jk=True):
        self.objs  = {mm:{} for mm in self.momlist}
        for mom,ratio in tqdm(self.cases):
            # Read requisites from 2pts analysis
            requisites = ratio_prerequisites(
                ens      = self.ensemble,
                ratio    = ratio,
                mom      = mom,
                readfrom = datadir_2pt,
                jk       = jk,
                w_from_corr3 = False
            )                    

            # Read ratio and save object
            io = RatioIO(self.ensemble,ratio,mom,PathToDataDir=datadir)
            robj = Ratio(
                io,
                jkBin     = BINSIZE[self.ensemble],
                smearing  = self.smslist,
                **requisites
            )

            self.objs[mom][ratio] = robj

        return

    def remove(self,*args):
        for k in args:
            try:
                self.keys.remove(k)
            except ValueError:
                print(f'{k} already in list')
                continue
        return


    # format in time-range and return a tree
    def format(self,tmin,alljk=False,**cov_specs):
        rdata   = {mm:{} for mm in self.momlist}
        rdatajk = {mm:{} for mm in self.momlist}
        xdata   = {mm:{} for mm in self.momlist}

        for mom,ratio in self.cases:
            tmp = self.objs[mom][ratio].format(
                trange   = (tmin, self.Ta-tmin),
                alljk    = alljk,
                **cov_specs
            )
            rdata  [mom][ratio] = tmp[1]
            xdata  [mom][ratio] = {sm: tmp[0] for sm in tmp[1]}
            rdatajk[mom][ratio] = tmp[-1]

        return (xdata,rdata,rdatajk) if alljk else (xdata,rdata)

    # plot ratio
    def plot(self,ratio,ax, factor=None):
        I = 0
        for mom,rat in self.cases:
            if rat!=ratio:
                continue

            x,yd = self.objs[mom][rat].format()

            for smr in self.smslist:
                yv = yd[smr] * (1 if factor is None else factor)
                ax.errorbar(
                    x,gv.mean(yv),gv.sdev(yv),
                    color = f'C{I}',
                    fmt   = '.'
                )
            I += 1

        return


    def params(self,dE_D=None,dM_B=None, dM_D=None):
        tmp = {
            'dE_D': '0.5(1.)' if dE_D is None else dE_D,
            'dM_B': '0.5(1.)' if dM_B is None else dM_B,
            'dM_D': '0.5(1.)' if dM_D is None else dM_D,
        }

        for mom,ratio in self.cases:
            x,ydata = self.objs[mom][ratio].format()
            f0 = np.mean([yy[len(yy)//2] for yy in ydata.values()])
            tmp[f'{ratio}_{mom}_f0'] = gv.gvar(f0.mean,1.5)

            for smr in self.smslist:
                if (mom,ratio,smr) in self.keys:
                    tmp[f'{ratio}_{mom}_{smr}'] = ['0(1)']*2

        return gv.gvar(tmp)


    def fit(self, tmin, priors=None, jkfit=False, **cov_specs):
        # Cut data
        xdata,rdata, rdatajk = self.format(tmin,alljk=True,**cov_specs)

        # Prepare fitting data
        tmp = []
        for mom,rat,smr in self.keys:
            tmp.append(rdata[mom][rat][smr])
        tmp = np.concatenate(tmp)

        yvec = gv.mean(tmp)
        ycov = gv.evalcov(tmp)  

        # Define flat model
        def global_model(x,p):
            aux = []
            for mom,rat,smr in self.keys:
                time = x[mom][rat][smr]
                aux.append(
                    model_ratio(self.Ta,mom,smr)[rat.upper()](time,p)
                )
            return np.concatenate(aux)

        # Fit function to data
        fit = lsqfit.nonlinear_fit(
            data   = (xdata,yvec,ycov),
            fcn    = global_model,
            prior  = self.params() if priors is None else priors,
        )

        jpars = []
        jpval = []
        if jkfit:
            ndof  = len(fit.y) - sum([len(pr) if np.ndim(pr)>0 else 1 for k,pr in fit.prior.items()])

            vecjk = [rdatajk[m][r][s] for m,r,s in self.keys]
            vecjk = np.hstack(vecjk)

            for ijk in tqdm(np.arange(vecjk.shape[0])):
                jfit = lsqfit.nonlinear_fit(
                    data   = (xdata,vecjk[ijk,:],ycov),
                    fcn    = global_model,
                    prior  = fit.prior,
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
            # fitjk['fit']  = fit          

        return fitjk if jkfit else fit
    



    def diff_model(self,xdata):
        def _model(pdict):
            aux = []
            for mom,rat,smr in self.keys:
                time = xdata[mom][rat][smr]
                aux.append(
                    model_ratio_jax(self.Ta,mom,smr)[rat.upper()](time,pdict)
                )
            return jnp.concatenate(aux)

        def _jac(pdict):
            return jax.jacfwd(_model)(pdict)

        def _hes(pdict):
            return jax.jacfwd(jax.jacrev(_model))(pdict)
            
        return _model, _jac, _hes


    def diff_cost(self, xdata, W2):
        model,_,_ = self.diff_model(xdata)
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


    def chi2exp(self, tmin, popt, fitcov, priors=None, pvalue=True, Nmc=1000, method='eigval', verbose=False):
        # Format and estimate covariance matrix
        xfit, yfit = self.format(tmin)

        yflat = []
        for mom,rat,smr in self.keys:
            yflat.append(yfit[mom][rat][smr])
        yflat = np.concatenate(yflat)
        cov = gv.evalcov(yflat)  

        pkeys = sorted(popt.keys())

        # Estimate jacobian
        fun,jac,_ = self.diff_model(xfit)
        _jc  = jac(popt)

        Jac = [_jc[k] for k in pkeys]
        for i,jv in enumerate(Jac):
            if np.ndim(jv)==1:
                Jac[i] = np.array([[j] for j in jv])
        Jac = np.hstack(Jac)

        # Check chi2 from fit
        w2 = np.linalg.inv(fitcov)
        res = gv.mean(yflat) - fun(popt)
        chi2 = float(res.T @ w2 @ res) # FIXME: when using rescaling of covariance matrix, the data are averaged from the jkbin=1 data and therefore central value can change a bit. This results in a O(.5) difference in calculated chi2. 

        # Add prior part if needed
        if priors is not None:
            # Enlarge jacobian
            Jac = np.vstack([Jac,-np.eye(Jac.shape[-1])])

            # Enlarge chi2
            prio_v = np.concatenate([[priors[k]] if np.ndim(priors[k])==0 else priors[k] for k in pkeys])
            popt_v = np.concatenate([[popt  [k]] if np.ndim(popt  [k])==0 else popt  [k] for k in pkeys])
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


    def plot_fit(self, ratio, fit, ax, factor=None):
        I = 0
        for mom,rat in self.cases:
            if rat!=ratio:
                continue

            xdata,yd = self.objs[mom][rat].format()

            for smr in self.smslist:
                if (mom,rat,smr) not in self.keys:
                    continue
                
                yv = yd[smr] * (1 if factor is None else factor)

                mask = np.array([x in fit.x[mom][rat][smr] for x in xdata])

                # Plot inside points
                ax.errorbar(
                    xdata[mask],
                    gv.mean(yv[mask]),
                    gv.sdev(yv[mask]),
                    color = f'C{I}',
                    fmt   = 'v' if smr=='RW' else '^' 
                )

                # Plot outside points
                ax.errorbar(
                    xdata[~mask],
                    gv.mean(yv[~mask]),
                    gv.sdev(yv[~mask]),
                    color = f'C{I}',
                    fmt   = 'v' if smr=='RW' else '^',
                    alpha = 0.2
                )

                # Plot fit
                xplot = np.arange(min(xdata[mask]),max(xdata[mask]),0.01)
                ax.plot(
                    xplot,
                    model_ratio(self.Ta,mom,smr)[ratio](xplot,fit.pmean)  * (1 if factor is None else factor),
                    color = f'C{I}',
                    linestyle = ':',
                )

                # Plot result
                f0 = fit.p[f'{ratio}_{mom}_f0']  * (1 if factor is None else factor)
                ax.axhspan(f0.mean-f0.sdev,f0.mean+f0.sdev,color=f'C{I}',alpha=0.2)

            I += 1

        return


    def chi2(self, fit):
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
        ndof  = len(fit.y) - sum([len(pr) if np.ndim(pr)>0 else 1 for k,pr in fit.prior.items()]) 
        pvalue = p_value(chi2,self.nconf,ndof)

        return dict(chi2=chi2, chi2_pr=chi_pr, chi2_aug=chi2_aug, pstd=pvalue)


    def fit_result(self, tmin, fit, verbose=True, priors=None):
        xdata  = fit.x
        yvec   = gv.mean(fit.y)
        fitcov = gv.evalcov(fit.y)
        popt   = dict(fit.pmean)

        chi2, chiexp, p = self.chi2exp(
            tmin, 
            popt, 
            fitcov, 
            pvalue = True, 
            priors = priors
        )

        chi2 = self.chi2(fit)

        if verbose:
            print(f'# ---------- Global fit in {tmin} for ens: {self.ensemble} --------------')
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

        return res


    def show(self,fit):
        return show(self.cases,fit.p)




def main():
    DATA_DIR = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    DATA_2PT = '/Users/pietro/Desktop/lattice24/0.25/corr2_3'


    MOMLIST = ['000','100','200','300']
    RATLIST = ['XFSTPAR','ZRA1','RA1','XV','R0','R1']
    SMSLIST = ['1S','RW']

    ENSEMBLE = 'Coarse-1'

    cov_specs = dict(scale=True,shrink=True,cutsvd=1E-12)


    xx = RatioSet(ENSEMBLE,MOMLIST,RATLIST,SMSLIST)
    xx.collect(DATA_DIR,DATA_2PT)

    xx.remove(('000','ZRA1','RW'))

    priors = xx.params()
    fit = xx.fit(
        tmin   = 2,
        priors = priors,
        # jkfit  = True,
        **cov_specs
    )   

    breakpoint()
    print('results')
    xx.fit_result(2,fit,error=True,priors=priors)

