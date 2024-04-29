import sys
import itertools
import argparse
import numpy as np
import gvar  as gv

from matplotlib  import pyplot as plt
from tqdm        import tqdm

from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, Correlator, plot_effective_coeffs



def tmaxs():
    ens      = 'Coarse-2'
    mes      = 'Dst'
    binsize  = 16
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 

    for mom in ['000','100','200','300','110','211']:
        io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
        stag = Correlator(
            io       = io,
            jkBin    = binsize,
            smearing = smlist
        )
        print(f'{mom = } {stag.tmax(threshold=0.3,criterion=np.mean) = }')


def eff_coeffs(FLAG):
    ens      = 'Coarse-2'
    mes      = 'Dst'
    mom      = '211'
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

        print(tmaxe,trange)

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
    prs = argparse.ArgumentParser()
    prs.add_argument('--do', type=int)

    args = prs.parse_args()

    eff_coeffs(args.do)
    # tmaxs()