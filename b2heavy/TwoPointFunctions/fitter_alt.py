import scipy
import itertools
import autograd
import autograd.numpy    as anp
import numpy             as np
import gvar              as gv


from .types2pts import CorrelatorInfo, CorrelatorIO, Correlator
from .utils     import load_toml, NplusN2ptModel, p_value, ConstantModel
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



class StagFitter(Correlator):
    def __init__(self, io:CorrelatorIO, smearing=None, polarization=None, **kwargs):
        bsize = kwargs.get('jkBin')
        super().__init__(io,jkBin=bsize)

        self.smr = smearing     if not smearing==None     else list(self.data.smearing.values)
        self.pol = polarization if not polarization==None else list(self.data.polarization.values)
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

    def diff_model(self, Nexc):
        kidx = self.Kpars(Nexc)

        def model(tau, *params):
            i_sp   = int(anp.log10(tau))-1
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

        return model

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
                flatpar[i] = np.exp(pdict[k][n])
            else:
                flatpar[i] = pdict[k][n]

        return flatpar

    def residuals(self, Nexc, trange, wmat=None, **kwargs):
        vmodel = anp.vectorize(self.diff_model(Nexc))

        (xdata,ydata) = self.format(
            trange       = trange,
            smearing     = self.smr,
            polarization = self.pol,
            flatten      = False,
            **kwargs
        )
        xfit   = np.concatenate([xdata[k]*10**i for i,k in enumerate(self.keys)])
        yfit   = np.concatenate([ydata[k]       for   k in self.keys])

        w      = wmat if wmat is not None else np.linalg.inv(gv.evalcov(yfit))
        U,S,Vh = np.linalg.svd(w,hermitian=True)
        W = U @ np.diag(np.sqrt(S)) @ Vh

        def _residual(pars):
            r = gv.mean(yfit) - vmodel(xfit,*pars)
            return W @ r

        return _residual



def test():
    ens      = 'Coarse-1'
    mes      = 'Dst'
    mom      = '200'
    binsize  = 11
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    NEXC     = 2
    TLIM     = (10,19)

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    corr = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    fitter = CorrFitter(corr,smearing=['d-d','1S-1S','d-1S'])
    (X,meff,aeff), MEFF,AEFF, m_pr, apr = corr.EffectiveCoeff(trange=TLIM,covariance=False)
    fitter.fit(
        Nstates           = NEXC,
        trange            = TLIM,
        verbose           = True,
        priors            = fitter.set_priors_phys(NEXC,Meff=MEFF,Aeff=AEFF),
        # scale_covariance  = True,
        # shrink_covariance = True,
        covariance        = False,
        override          = True
    )
    fit = fitter.fits[NEXC,TLIM]

    parms = corr.pars_from_dict(gv.mean(fit.prior))
    resid = corr.residuals(NEXC,TLIM,covariance=False)

    # res = scipy.optimize.minimize(chisq,parms,method='Nelder-Mead')  
    res = scipy.optimize.least_squares(
        fun  = resid,
        x0   = parms,
        loss = 
        verbose = 2
    )

    true_p = corr.pars_from_dict(gv.mean(fit.p))
    print(res.x)
    print(true_p)