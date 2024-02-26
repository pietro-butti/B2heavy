import scipy
import itertools
import autograd
import autograd.numpy    as anp
import jax 
import jax.numpy         as jnp
import numpy             as np
import gvar              as gv

from warnings import warn

from .types2pts import CorrelatorInfo, CorrelatorIO, Correlator
from .fitter    import CorrFitter

def ExpDecay(Nt, osc=False):
    return lambda t,E,Z : Z * (anp.exp(-E*t) + anp.exp(-E*(Nt-t))) * ((-1)**(t+1) if osc else 1.)

def StagDecay(Nt,Nexc):
    def model(t,E0,Z0a,Z0b, E1,Z1a,Z1b, *high):
        C_t  = ExpDecay(Nt)         ( t, E0, Z0a * Z0b )
        C_t += ExpDecay(Nt,osc=True)( t, E1, Z1a * Z1b )
        for n in range(2*Nexc-2):
            en, zn = high[2*n], high[2*n+1]**2
            C_t += ExpDecay(Nt,osc=(False if n%2==0 else True))( t, en, zn)
        return C_t
    return model

def Energies(Evec):
    '''
        `Energies(Evec)`

        Returns a vector `erg` of the same size of `Evec` with:
        - `erg[0] = Evec[0]`
        - `erg[1] = anp.exp(Evec[1]) + erg[0]` 
        - `erg[2] = anp.exp(Evec[2]) + erg[0]` 
        - `erg[3] = anp.exp(Evec[3]) + erg[1]` 
        - `erg[4] = anp.exp(Evec[3]) + erg[0]` 
        - `erg[5] = anp.exp(Evec[3]) + erg[1]` 
    '''
    erg = [Evec[0]]
    for n in range(1,len(Evec)):
        erg.append(anp.exp(Evec[n]) + erg[n-(1 if n==1 else 2)])
    return erg





def PeriodicExpDecay(Nt):
    return lambda t,E,Z: Z * ( jnp.exp(-E*t) + jnp.exp(-E*(Nt-t)) ) 

def NplusN2ptModel(Nstates,Nt,sm,pol):
    sm1,sm2 = sm.split('-')
    mix = sm1!=sm2

    def aux(t,p):
        E0, E1 = p['E'][0], p['E'][0] + jnp.exp(p['E'][1])
        Z0 = jnp.exp(p[f'Z_{sm1}_{pol}'][0]) * jnp.exp(p[f'Z_{sm2}_{pol}'][0])
        Z1 = jnp.exp(p[f'Z_{sm1}_{pol}'][1]) * jnp.exp(p[f'Z_{sm2}_{pol}'][1])
        ans = PeriodicExpDecay(Nt)(t,E0,Z0) + PeriodicExpDecay(Nt)(t,E1,Z1) * (-1)**(t+1)

        Es = [E0,E1]
        for i in range(2,2*Nstates):
            Ei = Es[i-2] + jnp.exp(p['E'][i])
            Z = p[f'Z_{sm if mix else sm1}_{pol}'][i-2 if mix else i]**2
            ans += PeriodicExpDecay(Nt)(t,Ei,Z) * (-1)**(i*(t+1))

            Es.append(Ei)
        return ans

    return aux


class StagFitter(Correlator):
    def __init__(self, io:CorrelatorIO, smearing=None, polarization=None, **kwargs):
        bsize = kwargs.get('jkBin')
        super().__init__(io,jkBin=bsize)

        self.smr = smearing     if not smearing==None     else list(self.data.smearing.values)
        self.pol = polarization if not polarization==None else list(self.data.polarization.values)
        self.data = self.data.loc[self.smr,self.pol,:,:]

        self.keys = sorted(itertools.product(self.smr,self.pol))


    def Kpars(self,Nexc):
        '''
            Builds all the keys for dictionary of parameters and returns a dictionary:
            - keys are `(k,n)` where `k` is the name of the coefficient and the `n` is the index in the corresponding vector
            - values are the index number of the corresponding element in a flattened array of parameters.

            E.g.
            #          k name   idx       flattened_idx
                k = (    E     , 0)  ---> 0 
                k = ( Z_1S_Bot , 0)  ---> 1 
                k = ( Z_1S_Par , 0)  ---> 2 
                k = ( Z_d_Bot  , 0)  ---> 3 
                k = ( Z_d_Par  , 0)  ---> 4 
                k = (    E     , 1)  ---> 5 
                k = ( Z_1S_Bot , 1)  ---> 6 
                k = ( Z_1S_Par , 1)  ---> 7 
                k = ( Z_d_Bot  , 1)  ---> 8 
                k = ( Z_d_Par  , 1)  ---> 9 
                k = (    E     , 2)  ---> 10
                k = ( Z_1S_Bot , 2)  ---> 11
                k = ( Z_1S_Par , 2)  ---> 12
                k = (Z_d-1S_Bot, 0)  ---> 13
                k = (Z_d-1S_Par, 0)  ---> 14
                k = ( Z_d_Bot  , 2)  ---> 15
                k = ( Z_d_Par  , 2)  ---> 16
                k = (    E     , 3)  ---> 17
                k = ( Z_1S_Bot , 3)  ---> 18
                k = ( Z_1S_Par , 3)  ---> 19
                k = (Z_d-1S_Bot, 1)  ---> 20
                k = (Z_d-1S_Par, 1)  ---> 21
                k = ( Z_d_Bot  , 3)  ---> 22
                k = ( Z_d_Par  , 3)  ---> 23
        '''
        keys = []
        for n in range(2*Nexc):
            keys.append(('E',n))
            for (sm,pol) in self.keys:
                sm1,sm2 = sm.split('-')
                mix = sm1!=sm2

                k = f'Z_{sm if mix else sm1}_{pol}'
                if not (mix and n<2):
                    keys.append((k,n-2 if mix else n))


        return {k: i for i,k in enumerate(keys)}

    def pars_from_dict(self, pdict):
        '''
            Given a dictionary of parameters `pdict` e.g.
                `pdict['E']          = [ 1.14478779, -2.61763813, -0.99322927, -3.50597109, -2.09325374, -1.25464778]`
                `pdict['Z_1S_Unpol'] = [ 0.18893983, -0.20667998,  0.4060087 ,  0.37910282,  0.40034158,  0.27345506]  `
                ...
            returns an array of all elements in pdict, flattened according to the mapping of `self.Kpars`
        '''
        erg = Energies(pdict['E'])
        Nexc = len(erg)//2

        kidx = self.Kpars(Nexc)

        flatpar = [None]*len(np.concatenate(pdict.values()))
        for (k,n),i in kidx.items():
            if k=='E':
                flatpar[i] = erg[n]
            elif k.startswith('Z') and n<2 and '-' not in k:
                flatpar[i] = jnp.exp(pdict[k][n])
            else:
                flatpar[i] = pdict[k][n]

        return jnp.array(flatpar)

    def scalar_model(self, Nexc):
        '''
            It returns a differentiable version of the correlator function `model(tau,*params)` where
            - `tau` is the timeslice multiplied by 10**i, i being the position index of the key (`smr`,`pol`) in `self.keys`. E.g.
                ('1S-1S','Unpol') ---> tau = [11  , 12  , 13  , 14  , 15  , 16  , 17  ]
                ('d-1S' ,'Unpol') ---> tau = [110 , 120 , 130 , 140 , 150 , 160 , 170 ]
                ('d-d'  ,'Unpol') ---> tau = [1100, 1200, 1300, 1400, 1500, 1600, 1700]
            - `params` is the flattened vector of parameters. See documentation of `Kpars` and `pars_from_dict`
        '''

        kidx = self.Kpars(Nexc)

        def _model(tau, *params):
            i_sp   = int(jnp.log10(tau))-1
            t = tau/10**i_sp

            sm,pol = self.keys[i_sp] 
            sm1,sm2 = sm.split('-')
            mix = sm1!=sm2

            # Fundamental physical state
            e0  = params[kidx['E',0]]
            z0a = params[kidx[f'Z_{sm1}_{pol}',0]]
            z0b = params[kidx[f'Z_{sm2}_{pol}',0]]

            # Oscillating physical state
            e1  = params[kidx['E',1]]
            z1a = params[kidx[f'Z_{sm1}_{pol}',1]]
            z1b = params[kidx[f'Z_{sm2}_{pol}',1]]

            # Excited stated
            high = []
            for n in range(2,2*Nexc):
                en = params[kidx['E',n]]
                Zn = params[kidx[f'Z_{sm if mix else sm1}_{pol}',n-2 if mix else n]]
                high.extend([en,Zn])

            return StagDecay(self.Nt, Nexc)(t,e0,z0a,z0b,e1,z1a,z1b,*high)

        return _model

    def cost_func(self, xdata, ydata, cor, Nexc, prior=None, vcost=False):
        condn = np.linalg.cond(cor)
        if condn > 1e13:
            warn(f'Correlation matrix may be ill-conditioned, condition number: {condn:1.2e}', RuntimeWarning)
        if condn > 0.1 / np.finfo(float).eps:
            warn(f'Correlation matrix condition number exceed machine precision {condn:1.2e}', RuntimeWarning)

        yvec = np.concatenate([ydata[k] for k in self.keys])
        covd = np.diag(1./gv.sdev(yvec))
        chol = np.linalg.cholesky(cor)
        chol_inv = scipy.linalg.solve_triangular(chol, covd, lower=True)

        func, _,_ = self.diff_model(xdata,Nexc)

        if prior is None:
            def _vector_cost(pdict):
                return chol_inv @ (func(pdict) - gv.mean(yvec))
        else:
            def _vector_cost(pdict):
                dt_c = chol_inv @ (func(pdict) - gv.mean(yvec))
                pr_c = jnp.concatenate([(gv.mean(prior[k])-pdict[k])/gv.sdev(prior[k]) for k in pdict])
                return jnp.concatenate(dt_c,pr_c)

        def _scalar_cost(pdict):
            res = _vector_cost(pdict)
            return res @ res

        return _vector_cost if vcost else _scalar_cost

    def diff_model(self, xdata, Nexc):
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
                [NplusN2ptModel(Nexc,self.Nt,smr,pol)(xdata[smr,pol],pdict) for smr,pol in self.keys]
            )

        def _jac(pdict):
            return jax.jacfwd(_model)(pdict)

        def _hes(pdict):
            return jax.jacfwd(jax.jacrev(_model))(pdict)
            
        return _model, _jac, _hes


    def diff_cost(self, Nexc, W2=None, **fitdata_args):
        '''
            Return a `jax`-differentiable cost function.
        '''
        xdata,ydata = self.format(flatten=False, **fitdata_args)

        # Prepare model, vector with y-data and weight matrix
        model, jac, hess = self.diff_model(xdata,Nexc)
        yvec = np.concatenate([ydata[k] for k in self.keys])
        wmat2 = jnp.asarray(W2) if W2 is not None else jnp.linalg.inv(gv.evalcov(yvec))

        def _cost(y,pdict):
            res = y - model(pdict)
            return jnp.matmul(jnp.transpose(res),jnp.matmul(wmat2,res))

        def _

        return _cost





    def chi2exp(self, Nexc, trange, popt, fitcov, pvalue=True, Nmc=10000):
        # Format data and estimate covariance matrix
        xdata,ydata = self.format(trange=trange,flatten=False,covariance=True)
        yvec = np.concatenate([ydata[k] for k in self.keys])
        cov = gv.evalcov(yvec)

        # Estimate jacobian and hessian
        fun,jac,hes = self.diff_model(xdata,Nexc)
        j = jac(popt)
        Jac = np.hstack([j[k] for k in popt])

        # Calculate chi2 from fit
        cdiag = np.diag(1./np.sqrt(np.diag(fitcov)))
        cor = cdiag @ fitcov @ cdiag
        chi2 = self.cost_func(xdata,ydata,cor,Nexc)(popt)

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
            
            return chiexp, p
        

def test():
    jax.config.update("jax_enable_x64", True)
    x = jax.random.uniform(jax.random.PRNGKey(0), (1000,))#, dtype=jnp.float64)
    assert x.dtype == jnp.float64


    ens      = 'Coarse-1'
    mes      = 'Dst'
    mom      = '200'
    binsize  = 11
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    NEXC     = 2
    TLIM     = (10,25)

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    self = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    data_spec = dict(
        covariance        = False, 
        scale_covariance  = True, 
        shrink_covariance = True
    )

    # ========================================= fit =============================================
    fitter = CorrFitter(self,smearing=smlist)
    (X,meff,aeff), MEFF,AEFF, m_pr, apr = self.EffectiveCoeff(trange=TLIM,covariance=False)
    pr = fitter.set_priors_phys(NEXC,Meff=MEFF,Aeff=AEFF)
    fitter.fit( Nstates=NEXC, trange=TLIM, priors=pr, override=True, **data_spec, verbose=True)
    fit = fitter.fits[NEXC,TLIM]
    # ===========================================================================================

    ce,p = self.chi2exp(
        Nexc   = NEXC,
        trange = TLIM,
        popt   = dict(fit.pmean),
        fitcov = gv.evalcov(fit.y),
        pvalue = True
    )
    print(fit.chi2red,ce,p)

    print(
        self.diff_cost(NEXC, trange=TLIM, **data_spec)(
            gv.mean(fit.y),dict(fit.pmean)
        )
    )