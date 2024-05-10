import sys
import itertools
import argparse
import toml
import numpy  as np
import gvar   as gv
import pandas as pd

from matplotlib  import pyplot as plt
from tqdm        import tqdm

from b2heavy.FnalHISQMetadata            import params as mData
from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, Correlator, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter    import StagFitter
from routines.fit_2pts_utils             import load_toml


binSizes  = {
    'MediumCoarse': 13,
    'Coarse-2'    : 16,
    'Coarse-1'    : 11,
    'Coarse-Phys' : 19,
    'Fine-1'      : 16,
    'Fine-Phys'   : 16,
    'SuperFine'   : 22
}

def find_eps(stag,trange,**cov_specs):
    # cov_specs = dict(shrink=True, scale=True)
    x,y, data = stag.format(trange=trange,flatten=True,alljk=True,**cov_specs)    

    cov = np.cov(data.T) * (data.shape[0]-1)
    cdiag = np.diag(1./np.sqrt(np.diag(cov)))
    cor = cdiag @ cov @ cdiag

    eval,evec = np.linalg.eigh(cor)
    y = sorted(abs(eval))/max(eval)

    I=None
    for i,r in enumerate((y/np.roll(y,1))[1:]):
        if r>1e+05:
            I=i+1
            break

    return 10E-12 if I is None else sorted(abs(eval))[I]


def metapars():
    tmin1 = 1.4
    tmin2 = 1.021 
    tmin3 = 0.631

    # config = toml.load('../routines/2pts_fit_config.toml')


    mes      = 'Dst'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    ens_list = [
        # 'MediumCoarse',
        # 'Coarse-2',
        # 'Coarse-1',
        # 'Coarse-Phys',
        # 'Fine-1'
        'Fine-Phys'
    ]
    mom_list = [
        '000',
        '100',
        '200',
        '300',
        '400',
        '110',
        '211',
        '222'
    ]
    mes_list = ['Dst','B']


    config = {
        'fit': {
            'Fine-Phys': {
                'Dst': {'mom':{}},
                'B'  : {'mom':{}}
            }
        }
    }

    aux = []
    for ens,mes,mom in itertools.product(ens_list,mes_list,mom_list):
        print(f'-------- {ens,mom} --------')
        a_fm = mData(ens)['aSpc'].mean

        try:
            io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
            stag = StagFitter(
                io       = io,
                jkBin    = binSizes[ens],
                smearing = smlist
            )
        except:
            continue

        # choose tmax
        tmax = stag.tmax(threshold=0.3)
        
        # choose tmin
        Tmin1  = int(tmin1/a_fm)
        tmin  = int(tmin3/a_fm)
        Tmin2 = int(tmin2/a_fm)

        #  diagnose correlation
        eps = find_eps(stag,(tmin,tmax),scale=True,shrink=True)

        config['fit'][ens][mes]['mom'][mom] = {}

        config['fit'][ens][mes]['mom'][mom]['nstates']    = 3
        config['fit'][ens][mes]['mom'][mom]['trange_eff'] = [Tmin1,tmax] 
        config['fit'][ens][mes]['mom'][mom]['trange']     = [tmin,tmax]
        config['fit'][ens][mes]['mom'][mom]['svd']        = float(eps)

        d = {
            'ensemble'   : ens,
            'meson'      : mes,
            'momentum'   : mom,
            'tmin(3+3)'  : tmin,
            'tmin(2+2)'  : Tmin2,
            'tmax'       : tmax,
            'svd'        : eps,
        }

        aux.append(d)
    
    df = pd.DataFrame(aux).set_index(['meson','ensemble','momentum'])
    print(df)


    with open('scemo_2.toml','w') as f:
        toml.dump(config,f)










def test():
    only_eff = False


    ## ========================================

    tmin1 = 1.4
    tmin2 = 1.021 
    tmin3 = 0.631


    config = '../routines/2pts_fit_config.toml'

    mes      = 'Dst'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

        
    ens_list = [
        # 'MediumCoarse',
        # 'Coarse-2',
        # 'Coarse-1',
        # 'Coarse-Phys',
        # 'Fine-1'
        'Fine-Phys'
    ]
    mom_list = [
        # '000',
        # '100',
        '200',
        # '300',
        # '400',
        # '110',
        # '211',
        # '222'
    ]


    aux = []
    for ens,mom in itertools.product(ens_list,mom_list):
        print(f'-------- {ens,mom} --------')
        a_fm = mData(ens)['aSpc'].mean

        try:
            io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
            stag = StagFitter(
                io       = io,
                jkBin    = binSizes[ens],
                smearing = smlist
            )
        except:
            continue

        # choose tmax
        tmax = stag.tmax(threshold=0.3)
        
        # choose tmin
        tmin  = int(tmin3/a_fm)
        Tmin2 = int(tmin2/a_fm)

        # diagnose correlation
        # xdata,ydata,yjk = stag.format(trange=(tmin,tmax),alljk=True,flatten=True)
        # svd = round(correlation_diagnostics(yjk,verbose=False),ndigits=4)
        eps = find_eps(stag,(tmin,tmax),scale=True,shrink=True)


        # print effective mass and p-value
        fit_conf = load_toml(config)['fit'][ens][mes]['mom'][mom]
        teff_range = fit_conf['trange_eff']


        # tmin = 7
        # tmax = 29
        # teff_range = (15,33)



        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True ,
            shrink = True ,
            cutsvd = eps,
        )
        effm,effa = stag.meff(trange=teff_range,variant='cosh', **cov_specs)
        chi2, chiexp, p = stag.chiexp_meff(trange=teff_range,variant='cosh',pvalue=True,**cov_specs)

        d = {
            'ensemble'   : ens,
            'momentum'   : mom,
            'tmin(3+3)'  : tmin,
            # 'tmin(2+2)'  : Tmin2,
            'tmax'       : tmax,
            'svd'        : eps,#svd,
            'trange_eff' : teff_range,
            'E0 (eff)'   : effm,
            'pval (eff)' : p,
        }

        if not only_eff:
            # tmax = 25

            # fit correlator
            fit = stag.fit(
                Nstates = 3,
                trange  = (tmin,tmax),
                priors  = stag.priors(3,Meff=effm,Aeff=effa),
                **cov_specs
            )
            res = stag.fit_result(3,(tmin,tmax),priors=fit.prior)
            # res = stag.fit_result(3,(tmin,tmax))

            d['E0'  ] = fit.p['E'][0]
            d['pval'] = res['pvalue']


        aux.append(d)
    
    df = pd.DataFrame(aux).set_index(['ensemble','momentum'])
    print(df)

def eff_coeffs(FLAG):
    ens      = 'MediumCoarse'
    mes      = 'B'
    mom      = '000'
    binsize  = 16
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 


    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    stag = Correlator(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )



    print(f'{mom = } {stag.tmax(threshold=0.3) = }')

    # ----------------------------- Correlation analysis -----------------------------
    if FLAG==1:
        tmin = 9
        tmax = 25
        trange = (tmin,tmax)
        tmaxe = stag.tmax(threshold=0.3)
        xdata,ydata,yjk = stag.format(trange=(tmin,tmax),alljk=True,flatten=True)

        print(f'-------- Correlation diagnostics whithin ({tmin},{tmax}) --------- ')
        print(f'{tmaxe = }')
        correlation_diagnostics(yjk,plot=True)
        plt.show()
    # --------------------------------------------------------------------------------


    # ----------------------------- Effective corr analysis -----------------------------
    elif FLAG==2:
        tmin = 9
        tmax = 17
        tmaxe = stag.tmax(threshold=0.3)
        trange = (tmin,tmax)

        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True,
            shrink = True,
            cutsvd = 0.01,
        )

        args = stag.meff(trange=trange,verbose=True,plottable=True,variant='cosh', **cov_specs)
        chi2, chiexp,p = stag.chiexp_meff(trange=trange,variant='cosh',pvalue=True,**cov_specs)

        print('--------------------------')
        print(f'{ens = }, {mes = }, {mom = }')
        print(tmaxe,trange)
        print(f'meff = {args[-3]}')
        print(f'{trange = }')
        print(f'{chi2/chiexp = :.2f}')
        print(f'{p = }')
        print('--------------------------')
        plot_effective_coeffs(trange, *args)
        plt.show()
    # --------------------------------------------------------------------------------

    # ------------------------- Effective range analysis -----------------------------
    elif FLAG==3:
        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True,
            shrink = True,
            cutsvd = 0.01,
        )

        tmins = np.arange(8,17)
        tmaxs = [stag.tmax(threshold=0.3)]
        print(tmaxs)
        tranges = itertools.product(tmins,tmaxs)

        MEFF = []
        AEFF = []
        TIC = []
        Tranges = []
        for trange in tranges:
            print(f'--------------- {trange} --------------')
            ydata, Aeff, aplt, apr, Meff, meffs, mpr = stag.meff(
                trange    = trange,
                plottable = True,
                variant   = 'cosh', 
                **cov_specs
            )

            try:
                chi2, chiexp = stag.chiexp_meff(
                    trange  = trange,
                    variant = 'cosh',
                    **cov_specs
                )
            except:
                continue

            MEFF.append(Meff)
            AEFF.append([Aeff[k] for k in stag.keys])
            TIC.append(chi2-2*chiexp)
            Tranges.append(trange)

        MEFF = np.array(MEFF)
        AEFF = np.array(AEFF)
        TIC = np.array(TIC)

        tic = np.exp(-TIC/2)
        tic = tic/tic.sum()

        tmin = np.array([min(t) for t in Tranges])
        tmax = np.array([max(t) for t in Tranges])

        print(f't_min = {np.sum(tmin*tic)}')
        print(f't_min = {np.sum(tmax*tic)}')
        print(f'E_eff = {np.sum(MEFF*tic)}')
        for i,k in enumerate(stag.keys):
            print(f'A_eff[{k}] = {np.sum(AEFF[:,i]*tic)}')
    # --------------------------------------------------------------------------------




if __name__=='__main__':
    # prs = argparse.ArgumentParser()
    # prs.add_argument('--eff', action='store_true')
    # args = prs.parse_args()

    test()
    # metapars()


