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
                               --tmaxs    [list of tmaxs (listed without commas), if not specified, the 30 criterion will be applied]
                               --verbose

                               --diag
                               --block
                               --scale
                               --svdcut


Examples
python fit_2pts_preanalysis.py --ensemble MediumCoarse --meson Dsst --mom 000 --maxerr 25 --Nstates 1 2 3 --tmins 8 9 10 11 12 13 14 15 16 17 18 19      
'''

import argparse
import sys
import os

import pandas as pd
import gvar   as gv
import numpy  as np

import fit_2pts_utils as utils

from DEFAULT_ANALYSIS_ROOT               import DEFAULT_ANALYSIS_ROOT
from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO
from b2heavy.TwoPointFunctions.fitter    import StagFitter



HERE = os.path.join('/',*__file__.split('/')[:-1])
sys.path.append(HERE)

prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default=os.path.join(HERE,'2pts_fit_config.toml'))
prs.add_argument('-e','--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('-m','--meson'   , type=str,  nargs='+',  default=None)
prs.add_argument('-mm','--mom'     , type=str,  nargs='+',  default=None)
prs.add_argument('--Nstates' , type=int,  nargs='+',  default=None)
prs.add_argument('--tmins'   , type=int,  nargs='+',  default=None)
prs.add_argument('--tmaxs'   , type=int,  nargs='+',  default=None)
prs.add_argument('--saveto'  , type=str,  default=None)
prs.add_argument('--maxerr'  , type=int,  default=25)
prs.add_argument('--verbose', action='store_true')

prs.add_argument('--full'    , action='store_true')
prs.add_argument('--diag'    , action='store_true')
prs.add_argument('--block'   , action='store_true')
prs.add_argument('--scale'   , action='store_true')
prs.add_argument('--svdcut'  , type=float, nargs='+', default=None)




def append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,scale,diag,block,svd,full,time,chi2,chi2exp,p,fit):
    DF['tag'].append(f'{ens}_{meson}_{mom}')
    DF['Tmax'].append(Tmax)
    DF['tmin'].append(      tmin         )
    DF['tmax'].append(      tmax         )
    DF['Nstates'].append(   nexc         )
    DF['Ndof'].append(utils.Ndof(NPOL,trange,nexc))
    DF['scale+shrink'].append( 'x' if scale else ' '     )
    DF['diag'].append(         'x' if diag  else ' '     )
    DF['block'].append(        'x' if block else ' '     )
    DF['full'].append(         'x' if full  else ' '     ) 
    DF['svd'].append(          svd       )
    DF['time'].append(         time      )
    DF['chi2'].append(   round(chi2   ,1))
    DF['chi2exp'].append(round(chi2exp,1))
    DF['chi2/chi2exp'].append(round(chi2/chi2exp,1))
    DF['chi2/dof'].append(round(chi2/utils.Ndof(NPOL,trange,nexc),1))
    DF['p'].append(            p         )
    DF['E0'].append(        fit.p['E'][0])



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

    MAX_ERR = args.maxerr/100


    DF = {
        'tag': [],
        'tmin': [], 'tmax': [], 'Nstates': [], 'Ndof': [],
        'Tmax': [],
        'scale+shrink':[],'diag':[],'block':[],'svd':[],'full':[],
        'time':[],'chi2':[],'chi2exp':[],'chi2/chi2exp':[],'chi2/dof':[],'p':[],'E0':[]
    }


    for ens in ENSEMBLE_LIST:
        for meson in MESON_LIST:
            for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][meson]['mom'].keys()):
                data_dir = config['data'][ens]['data_dir']
                binsize  = config['data'][ens]['binsize'] 
                smlist   = config['fit'][ens][meson]['smlist'] 

                stag = StagFitter(
                    io       = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir),
                    jkBin    = binsize,
                    smearing = smlist
                )

                # TMAX ANALYSIS ----------------------- 30% error -----------------------
                xdata,ydata = stag.format()
                rel = np.vstack([abs(gv.sdev(y)/gv.mean(y)) for y in ydata.values()]).mean(axis=0)
                Tmax = min([t for t,r in enumerate(rel) if r>=MAX_ERR])

                # TMIN pre-ANALYSIS -----------------------------------------------------
                NEXC  = [1,2,3]                    if not args.Nstates else args.Nstates 
                TMINS = stag.data.timeslice.values if not args.tmins   else args.tmins
                TMAXS = [Tmax]                     if not args.tmaxs   else args.tmaxs
                NPOL  = len(stag.data.polarization)
                

                for tmax in TMAXS:
                    for tmin in TMINS:
                        for nexc in NEXC:
                            trange = (tmin,tmax)

                            xdata,ydata,ally = stag.format(trange=trange, alljk=True, flatten=True)
                            svdcut = correlation_diagnostics(ally,verbose=True)

                            if args.verbose:
                                print(f'# -------------- mes: {stag.info.meson} of ens: {stag.info.ensemble} for mom: {stag.info.momentum} --------------')
                                print(f'#     tmin={tmin},tmax={tmax},')
                                print(f'#         nexc={nexc}')

                            Meff, Aeff = stag.meff(trange)
                            pr = stag.priors(nexc,Meff=Meff,Aeff=Aeff)

                            if args.full:
                                print(f'#             cov opt: full')
                                fit = stag.fit(nexc,trange,priors=pr,cutsvd=1e-12)
                                _, pars, chi2, chiexp, p = stag.fit_result(nexc,trange,verbose=False)
                                append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,False,False,False,None,False,fit.time,chi2,chiexp,p,fit)

                            if args.diag:
                                print(f'#             cov opt: diag')
                                fit = stag.fit(nexc,trange,priors=pr,diag=True)
                                _, pars, chi2, chiexp, p = stag.fit_result(nexc,trange,verbose=False)
                                append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,False,True,False,None,False,fit.time,chi2,chiexp,p,fit)

                            if args.block:
                                print(f'#             cov opt: block')
                                fit = stag.fit(nexc,trange,priors=pr,block=True)
                                _, pars, chi2, chiexp, p = stag.fit_result(nexc,trange,verbose=False)
                                append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,False,False,True,None,False,fit.time,chi2,chiexp,p,fit)

                            if args.scale:
                                print(f'#             cov opt: scale+shrink')
                                fit = stag.fit(nexc,trange,priors=pr,scale=True,shrink=True)
                                _, pars, chi2, chiexp, p = stag.fit_result(nexc,trange,verbose=False)
                                append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,True,False,False,None,False,fit.time,chi2,chiexp,p,fit)

                            if args.svdcut is not None:
                                for svd in args.svdcut:
                                    print(f'#             cov opt: {svd = }')
                                    fit = stag.fit(nexc,trange,priors=pr,cutsvd=svd)
                                    _, pars, chi2, chiexp, p = stag.fit_result(nexc,trange,verbose=False)
                                    append_row(DF,ens,meson,mom,Tmax,tmin,tmax,nexc,NPOL,trange,False,False,False,None,False,fit.time,chi2,chiexp,p,fit)
        

    DF = pd.DataFrame(DF)#.set_index(['ensemble','meson','momentum','tmin','Nstates','scale','shrink'],inplace=True)
    DF.set_index(['tag','Tmax','tmax','tmin','Nstates'],inplace=True)
    print(DF)






    # if args.cov_filter is not None:
    #     DF = DF[DF['scale']==args.cov_filter[0]]
    #     DF = DF[DF['shrink']==args.cov_filter[1]]


    
    # DF.style \
    # .format(precision=3, thousands=".", decimal=",") \
    # .format_index(str.upper, axis=1)






if __name__ == "__main__":
    main()


    

