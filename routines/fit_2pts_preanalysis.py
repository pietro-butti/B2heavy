# This routine analyzes general properties of a fit given 
# ensemble, meson, momentum


usage = '''
python fit_2pts_preanalysis.py --config   [file location of the toml config file]         
                               --ensemble [list of ensembles analyzed]                    
                               --meson    [list of meson analyzed]                        
                               --mom      [list of momenta considered]
                               --saveto   [where do you want to save? Defaults='./' while 'default' goes to DEFAULT_ANALYSIS_ROOT]                     
                               --maxerr   [Error percentage for Tmax]                     
                               --Nstates  [list of N for (N+N) fit (listed without comas)]
                               --tmins    [list of tmins (listed without commas)]



Examples
python fit_2pts_preanalysis.py --ensemble MediumCoarse --meson Dsst --mom 000 --maxerr 25 --Nstates 1 2 3 --tmins 8 9 10 11 12 13 14 15 16 17 18 19      
'''





from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import argparse
import pickle
import sys
import tomllib
import os
import datetime
import pandas as pd

from prettytable import PrettyTable

from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, Correlator
from b2heavy.TwoPointFunctions.fitter    import CorrFitter

HERE = os.path.join('/',*__file__.split('/')[:-1])
sys.path.append(HERE)

import fit_2pts_stability_test as utils



def Ndof(Npol,trange,nexc,Nsmr=2):
    Npar = 2*nexc + 2*nexc + 2*nexc + 2*(nexc-1)
    Npoints = Npol*(Nsmr*(Nsmr+1)//2)*(max(trange)-min(trange))
    return Npoints-Npar





prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default=os.path.join(HERE,'2pts_fit_config.toml'))
prs.add_argument('--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('--meson'   , type=str,  nargs='+',  default=None)
prs.add_argument('--mom'     , type=str,  nargs='+',  default=None)
prs.add_argument('--Nstates' , type=int,  nargs='+',  default=None)
prs.add_argument('--tmins'   , type=int,  nargs='+',  default=None)
prs.add_argument('--saveto'  , type=str,  default=None)
prs.add_argument('--maxerr'  , type=int,  default=25)


def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ENSEMBLE_LIST = args.ensemble if args.ensemble is not None else config['ensemble']['list']
    MESON_LIST    = args.meson    if args.meson    is not None else config['meson']['list']
    MOM_LIST      = args.mom      if args.mom      is not None else []

    print(f'# B2heavy AUTO: ------------- RUNNING 2pt fit pre-analysis -------------')
    print(f'# B2heavy AUTO: ------------- {ENSEMBLE_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MESON_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MOM_LIST if MOM_LIST else "all"} -------------')

    MAX_ERR = args.maxerr

    DF = {
        'tag': [],
        'tmin': [], 'Nstates': [], 'Ndof': [],
        'scale':[],'shrink':[],
        'time':[],'chi2 [red]':[],'chi2 [aug]':[],'p value':[],'E0':[]
    }



    for ens in ENSEMBLE_LIST:
        for meson in MESON_LIST:
            for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][meson]['mom'].keys()):


                data_dir = config['data'][ens]['data_dir']
                binsize  = config['data'][ens]['binsize'] 
                smlist   = config['fit'][ens][meson]['smlist'] 

                io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
                corr = Correlator(io,jkBin=binsize)

                # TMAX ANALYSIS ----------------------- 30% error -----------------------
                df = {}
                for tmax in corr.data.timeslice.values:
                    x,Y = corr.format(trange=(tmax,tmax))
                    s,i = [],[]
                    for (smr,pol),y in Y.items():
                        s.append( round(abs(y[0].sdev/y[0].mean)*100,1) )
                        i.append(f'{smr}_{pol}')
                    df[tmax] = pd.Series(s,index=i)
                df = pd.DataFrame(df)

                TMAX = min(df.mean().where(lambda x: x>MAX_ERR).dropna().index)


                # TMIN pre-ANALYSIS -----------------------------------------------------
                NEXC  = [1,2,3]                    if not args.Nstates else args.Nstates 
                TMINS = corr.data.timeslice.values if not args.tmins   else args.tmins
                NPOL  = len(corr.data.polarization)
                

                for tmin in TMINS:
                    for nexc in NEXC:
                        trange = (tmin,TMAX)
                        _,MEFF,AEFF,_,_ = corr.EffectiveCoeff(trange,smearing=smlist)

                        for rescale in [False,True]:
                            for shrink in [False,True]:
                                print(f'# -------------- mes: {corr.info.meson} of ens: {corr.info.ensemble} for mom: {corr.info.momentum} --------------')
                                print(f'#     tmin={tmin}')
                                print(f'#         nexc={nexc}')
                                print(f'#             (scale,shrink)=({rescale},{shrink})')

                                fitter = CorrFitter(corr,smearing=smlist)
                                priors = fitter.set_priors_phys(nexc,Meff=MEFF,Aeff=AEFF)
                                fitter.fit(
                                    Nstates = nexc,
                                    trange  = trange,
                                    verbose = False,
                                    priors  = priors,
                                    scale_covariance   = rescale,
                                    shrink_covariance  = shrink
                                )
                                fit = fitter.fits[nexc,trange]

                                DF['tag'].append(f'{ens}_{meson}_{mom}')
                                DF['tmin'].append(      tmin         )
                                DF['Nstates'].append(   nexc         )
                                DF['Ndof'].append(Ndof(NPOL,trange,nexc))
                                DF['scale'].append(     rescale      )
                                DF['shrink'].append(    shrink       )
                                DF['time'].append(      fit.time     )
                                DF['chi2 [red]'].append(fit.chi2red  )
                                DF['chi2 [aug]'].append(fit.chi2     )
                                DF['p value'].append(   fit.pvalue   )
                                DF['E0'].append(        fit.p['E'][0])

    DF = pd.DataFrame(DF)#.set_index(['ensemble','meson','momentum','tmin','Nstates','scale','shrink'],inplace=True)
    DF.set_index(['tag','tmin','Nstates','scale','shrink'],inplace=True)
    
    print(DF)

    DF.style \
    .format(precision=3, thousands=".", decimal=",") \
    .format_index(str.upper, axis=1)

    print(DF)





if __name__ == "__main__":
    main()


    

