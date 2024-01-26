# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_config.py --config [file location of the toml config file]         
                   --only_ensemble [list of ensembles analyzed]                    
                   --only_meson    [list of meson analyzed]                        
                   --only_mom      [list of momenta considered]                    
                   --jkfit         [repeat same fit inside each bin]               
                   --saveto        [*where* do you want to save files]             
                   --logto         [Log file name]                                 
                   --override      [do you want to override pre-existing analysis?]

Examples
python 2pts_fit_config.py --only_ensemble MediumCoarse --only_meson Dsst --only_mom 000 100 --saveto .     
'''

DEFAULT_ANALYSIS_ROOT = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/01-23-2023'

import argparse
import pickle
import sys
import tomllib
import os

import datetime

from b2heavy.TwoPointFunctions.corr  import CorrelatorIO, Correlator
from b2heavy.TwoPointFunctions.fitter import CorrFitter

def fit_2pts_single_corr(ens, meson, mom, data_dir, binsize, smslist, nstates, trange, saveto=None, meff=True, jkfit=False, priors='meff'):
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
                Location of the `Ensemble` directori
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

    fitter = CorrFitter(corr,smearing=smslist)
    fit = fitter.fit(
        Nstates=nstates,
        trange=trange,
        verbose=True,
        pval=True,
        jkfit=jkfit,
        priors = fitter.set_priors_phys(nstates,Meff=MEFF if meff else None)
    )

    # Missing plot dump of effective mass FIXME

    if saveto is not None:
            with open(saveto, 'wb') as handle:
                if not jkfit:
                    f = fitter.fits[nstates,trange]
                    aux = dict(
                        x       = f.x,
                        y       = f.y,
                        p       = f.p,
                        pvalue  = f.pvalue,
                        chi2    = f.chi2,
                        chi2red = f.chi2red,
                        info    = (nstates,trange)
                    )
                    pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return



def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data



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

def main():
    args = prs.parse_args()

    config_file = args.config
    config = load_toml(config_file)

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

                if os.path.exists(SAVETO):
                    if not JKFIT:
                        saveto = f'{SAVETO}/fit2pt_config_{tag}.pickle' if args.saveto is not None else None
                    else:
                        saveto = f'{SAVETO}/fit2pt_config_jk_{tag}.pickle' if args.saveto is not None else None
                else:
                    raise NameError(f'{SAVETO} is not an existing location.')

                # Check if there is an already existing analysis =====================================
                if os.path.exists(saveto):
                    if args.override:
                        print(f'Already existing analysis for {tag}, but overriding...')
                    else:
                        print(f'Analysis for {tag} already up to date')
                        continue

                # Perform analysis ===================================================================
                try:
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
                        priors = 'meff'
                    )
                except:
                    pass

                # LOG analysis =======================================================================
                log = f'''
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
#        meff     = True                                                           
#        priors   = 'meff'                                                        
# =============================================================================
                ''' 
                point = '.'
                if not JKFIT:
                    logfile = f'{args.saveto if args.saveto is not None else point}/fit2pt_config_{tag}.log' if args.logto==None else args.logto
                else:
                    logfile = f'{args.saveto if args.saveto is not None else point}/fit2pt_config_jk_{tag}.log' if args.logto==None else args.logto
                with open(logfile,'a+') as f:
                    f.write(log)







if __name__ == "__main__":
    main()


    





    