# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_config.py --config [file location of the toml config file]         
                   --only_ensemble [list of ensembles analyzed]                    
                   --only_meson    [list of meson analyzed]                        
                   --only_mom      [list of momenta considered]                    
                   --jkfit         [repeat same fit inside each bin. Default: false]               
                   --saveto        [*where* do you want to save files.]             
                   --logto         [Log file name]                                 
                   --override      [do you want to override pre-existing analysis?]
                   --scale         [rescale the covariance matrix with the diagonal]
                   --shrink        [shrink covariance matrix]

Examples
python 2pts_fit_config.py --only_ensemble MediumCoarse --only_meson Dsst --only_mom 000 100 --saveto .     
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT


import argparse
import pickle
import sys
import tomllib
import os
import gvar as gv

import datetime

from b2heavy.TwoPointFunctions.types2pts  import CorrelatorIO, Correlator
from b2heavy.TwoPointFunctions.fitter import CorrFitter

import fit_2pts_utils as utils

def fit_2pts_single_corr(
    ens, meson, mom, 
    data_dir, 
    binsize, 
    smslist, 
    nstates, 
    trange, 
    saveto = None, 
    meff   = False, 
    aeff   = False, 
    jkfit  = False,
    scale  = False,
    shrink = False,
):
    """
        This function perform a fit to 2pts correlation function

        Arguments
        ---------

            ens: str
                Ensemble name, e.g. `MediumCoarse`
            meson: str
                Meson name, e.g. `Dsst`
            mom: str
                Momentum string, e.g. `200`
            data_dir: str
                Location of the `Ensemble` directory
            binsize: int
                Bin size for jackknife error
            smslist: list
                List of smearing to be considered
            nstates: int
                Number of states to be fitted. E.g. set to 2 for "2+2" fits.
            saveto: str
                Path of the location where to save the results
            meff: bool
    """
    
    # initialize structures
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr = Correlator(io,jkBin=binsize)

    if meff:
        _,MEFF,_ = corr.EffectiveMass(trange=trange,smearing=smslist)
    if aeff:
        _,MEFF,AEFF,_,_ = corr.EffectiveCoeff(trange,smearing=smslist)


    fitter = CorrFitter(corr,smearing=smslist)
    priors = fitter.set_priors_phys(nstates,Meff=MEFF if meff else None, Aeff=AEFF if aeff else None)
    fit = fitter.fit(
        Nstates           = nstates,
        trange            = trange,
        verbose           = True,
        pval              = True,
        jkfit             = jkfit,
        priors            = priors,
        scale_covariance  = scale,
        shrink_covariance = shrink
    )

    # Missing plot dump of effective mass FIXME

    if saveto is not None:
        name = f'{saveto}_fit.pickle'
        if jkfit:
            with open(name, 'wb') as handle:
                pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            f = fitter.fits[nstates,trange]
            utils.dump_fit_object(f,saveto)

    return





def log(tag,ens,meson,mom,data_dir,smlist,trange,saveto,JKFIT,shrink,scale):
    return f'''
# ================================== ({tag}) ==================================
# fit_2pts_single_corr from {__file__} called at {datetime.datetime.now()} with
#        ens      = {ens}                                                           
#        meson    = {meson}                                                       
#        mom      = {mom}                                                        
#        data_dir = {data_dir}                                                      
#        smlist   = {smlist}                                                      
#        trange   = {trange}                                                       
#        saveto   = {saveto}                                                       
#        jkfit    = {JKFIT}                                                        
#        meff     = True,
#        aeff     = True,                                                           
#        shrink   = {shrink},
#        scale    = {scale}                          
# =============================================================================
''' 



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('--only_ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('--only_meson'   , type=str,  nargs='+',  default=None)
prs.add_argument('--only_mom'     , type=str,  nargs='+',  default=None)
prs.add_argument('--saveto'       , type=str,  default=None)
prs.add_argument('--jkfit'        , action='store_true')
prs.add_argument('--override'     , action='store_true')
prs.add_argument('--logto'        , type=str, default=None)
prs.add_argument('--debug'        , action='store_true')
prs.add_argument('--scale'        , action='store_true', default=False)
prs.add_argument('--shrink'       , action='store_true', default=False)

def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ENSEMBLE_LIST = args.only_ensemble if args.only_ensemble is not None else config['ensemble']['list']
    MESON_LIST    = args.only_meson    if args.only_meson    is not None else config['meson']['list']
    MOM_LIST      = args.only_mom      if args.only_mom      is not None else []

    print(f'# B2heavy AUTO: ------------- RUNNING 2pt fits for -------------')
    print(f'# B2heavy AUTO: ------------- {ENSEMBLE_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MESON_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MOM_LIST} -------------')

    JKFIT  = True if args.jkfit else False

    for ens in ENSEMBLE_LIST:
        for meson in MESON_LIST:
            for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][meson]['mom'].keys()):

                tag = config['fit'][ens][meson]['mom'][mom]['tag']
                data_dir = config['data'][ens]['data_dir']
                binsize  = config['data'][ens]['binsize'] 
                smlist   = config['fit'][ens][meson]['smlist'] 
                nstates  = config['fit'][ens][meson]['nstates'] 
                trange   = tuple(config['fit'][ens][meson]['mom'][mom]['trange']) 

                SAVETO = DEFAULT_ANALYSIS_ROOT if args.saveto=='default' else args.saveto

                if SAVETO is not None:
                    if not os.path.exists(SAVETO):
                        raise Exception(f'{SAVETO} is not an existing location.')
                    else:
                        if not JKFIT:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}' if args.saveto is not None else None
                        else:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}_jk' if args.saveto is not None else None
                    
                        # Check if there is an already existing analysis =====================================
                        if os.path.exists(f'{saveto}.pickle'):
                            if args.override:
                                print(f'Already existing analysis for {tag}, but overriding...')
                            else:
                                print(f'Analysis for {tag} already up to date')
                                continue


                # Perform analysis ===================================================================
                # try:
                fit_2pts_single_corr(
                    ens, meson, 
                    mom, 
                    data_dir, 
                    binsize, 
                    smlist, 
                    nstates, 
                    trange, 
                    saveto = saveto, 
                    jkfit  = JKFIT,
                    meff   = True, 
                    aeff   = True,
                    shrink = args.shrink,
                    scale  = args.scale
                )
                # except Exception:
                #     print('PORCODIO')
                #     pass

                # LOG analysis =======================================================================

                if SAVETO is not None:
                    logfile = f'{saveto}.log' if args.logto==None else args.logto
                    with open(logfile,'w') as f:
                        f.write(log(tag,ens,meson,mom,data_dir,smlist,trange,saveto,JKFIT,args.shrink,args.scale))





if __name__ == "__main__":
    main()


    





    