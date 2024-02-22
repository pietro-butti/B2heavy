import gvar as gv
import numpy as np
import corrfitter


from b2heavy.TwoPointFunctions.types2pts  import CorrelatorIO, Correlator
from b2heavy.TwoPointFunctions.fitter     import CorrFitter




def make_model(corr,ydict):
    models = {}
    for sm,pol in ydict:
        sm1,sm2 = sm.split('-')

        models[sm,pol] = corrfitter.Corr2(
            datatag = (sm,pol),
            tp      = corr.Nt,
            s       = -1.,
            a       = f'Z_{sm1}_Bot',
            b       = f'Z_{sm2}_Bot',
            dE      = 'dE'
        )


def main():
    ens = 'Coarse-1'
    mes = 'Dst'
    mom = '200'
    binsize = 11
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'

    io     = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    corr   = Correlator(io,jkBin=binsize)
    fitter = CorrFitter(corr,smearing=['1S-1S','d-d','d-1S'])


    Nstates = 2
    pr = fitter.set_priors_phys(Nstates=2)

    print(pr)

    # c2pt = corrfitter.Corr2(
    #     datatag = ('1S-1S','Bot'),
    #     tp      = corr.Nt,
    #     s       = -1.,
    #     a       = 'Z_1S_Bot',
    #     b       = 'Z_1S_Bot',
    #     dE      = 'E'
    # )



if __name__=='__main__':
    main()