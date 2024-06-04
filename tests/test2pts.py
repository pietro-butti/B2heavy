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
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, Correlator, plot_effective_coeffs, find_eps_cut
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

tranges = { #(tmin1,tmin3,tmin2,tmax)
    'D': (1.8,0.9,0.45,2.7),
    'B': (1.8,0.9,0.45,2.7),
    'Dst': (1.5,0.631,1.021,'30%')
}




def metapars():
    mes      = 'Dst'
    tmin1 = tranges[mes][0]
    tmin2 = tranges[mes][2]
    tmin3 = tranges[mes][1]

    Tmax = tranges[mes][3]

    # config = toml.load('../routines/2pts_fit_config.toml')


    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    ens_list = [
        'MediumCoarse',
        'Coarse-2',
        'Coarse-1',
        'Coarse-Phys',
        'Fine-1',
        'Fine-Phys',
        'SuperFine',
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
    # mes_list = ['Dst','B']
    mes_list = [mes]


    config = {'fit': {}}

    aux = []
    for ens in ens_list:
        config['fit'][ens] = {}
        for mes in mes_list:
            config['fit'][ens][mes] = {'mom': {}}
            for mom in mom_list:
                config['fit'][ens][mes]['mom'][mom] = {}

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
                # tmax = int(Tmax/a_fm)
                
                # choose tmin
                Tmin1  = int(tmin1/a_fm)
                Tmin2 = int(tmin2/a_fm)

                tmin = int(tmin3/a_fm)


                # scemo1,scemo2,ysamples = stag.format(trange=(tmin,tmax),flatten=True,alljk=True)
                # eps = correlation_diagnostics(ysamples,verbose=False)
                eps = 1E-12


                config['fit'][ens][mes]['mom'][mom] = {}

                config['fit'][ens][mes]['mom'][mom]['tag']        = f'{ens}_{mes}_{mom}' # 'Coarse-Phys_B_211'
                config['fit'][ens][mes]['mom'][mom]['nstates']    = 3
                config['fit'][ens][mes]['mom'][mom]['trange_eff'] = [Tmin1,tmax] 
                config['fit'][ens][mes]['mom'][mom]['trange']     = [tmin,tmax]
                config['fit'][ens][mes]['mom'][mom]['svd']        = 1E-12

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


    with open('scemo.toml','w') as f:
        toml.dump(config,f)





def test():
    only_eff = False

    ## ========================================

    mes = 'Dst'
    (tmin1,tmin2,tmin3,Tmax) = tranges[mes]

    config = '../routines/2pts_fit_all.toml'

    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

        
    ens_list = [
        'MediumCoarse',
        'Coarse-2',
        'Coarse-1',
        'Coarse-Phys',
        'Fine-1',
        'Fine-Phys',
        'SuperFine'
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
        tmax1 = stag.tmax(threshold=0.3)
        tmax2 = int(Tmax/a_fm)
        tmax = min(tmax1,tmax2)


        # choose tmin
        Tmin3 = int(tmin3/a_fm)
        Tmin2 = int(tmin2/a_fm)
        tmin_eff = int(tmin1/a_fm)


        # print effective mass and p-value
        fit_conf = load_toml(config)['fit'][ens][mes]['mom'][mom]
        # teff_range = fit_conf['trange_eff']
        teff_range = (tmin_eff,tmax)


        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True ,
            shrink = True ,
            cutsvd = 1E-12,
        )
        effm,effa,p = stag.meff(trange=teff_range,variant='cosh', pvalue=True, **cov_specs)
        # chi2, chiexp, p = stag.chiexp_meff(trange=teff_range,variant='cosh',pvalue=True,**cov_specs)


        d = {
            'ensemble'   : ens,
            'momentum'   : mom,
            'tmin(3+3)'  : Tmin3,
            'tmin(2+2)'  : Tmin2,
            'trange_eff' : teff_range,
            'tmax'       : tmax,
            'E0 (eff)'   : effm,
            'pval (eff)' : p,
        }

        if not only_eff:
            # fit correlator
            fit = stag.fit(
                Nstates = 3,
                trange  = (Tmin3,tmax),
                priors  = stag.priors(3,Meff=effm,Aeff=effa),
                **cov_specs
            )
            res = stag.fit_result(3,(Tmin3,tmax),priors=fit.prior)

            d['E0[3]'  ] = fit.p['E'][0]
            d['pval[3]'] = res['pstd']

            fit2 = stag.fit(
                Nstates = 2,
                trange  = (Tmin2,tmax),
                priors  = stag.priors(2,Meff=effm,Aeff=effa),
                **cov_specs
            )
            res2 = stag.fit_result(2,(Tmin2,tmax),priors=fit2.prior)

            d['E0[2]'  ] = fit2.p['E'][0]
            d['pval[2]'] = res2['pstd']


        aux.append(d)
    
    df = pd.DataFrame(aux).set_index(['ensemble','momentum'])
    print(df)







if __name__=='__main__':
    # test()
    metapars()







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



