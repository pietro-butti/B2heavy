USAGE = '''
python 2pts_fit_config.py --config [file location of the toml config file]         
                   --ensemble [list of ensembles analyzed]                    
                   --meson    [list of meson analyzed]                        
                   --mom      [list of momenta considered]                    
                   --jkfit    [repeat same fit inside each bin. Default: false]               
                   --saveto   [*where* do you want to save files.]             
                   --logto    [Log file name]                                 
                   --override [do you want to override pre-existing analysis?]

                   --diag
                   --block
                   --svd
                   --scale         [rescale the covariance matrix with the diagonal]
                   --shrink        [shrink covariance matrix]
                   
                   --no_priors_chi [don't consider priors in the calculation of chi2]

                   --plot_eff
                   --plot_fit
                   --show

Examples
python 2pts_fit_config.py --ensemble MediumCoarse --meson Dsst --mom 000 100 --saveto .     
'''
import pickle
import argparse
import os
import datetime
import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)
import pandas            as pd
import matplotlib.pyplot as plt


from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter    import StagFitter

from b2heavy.ThreePointFunctions.utils     import read_config_fit, dump_fit_object
import fit_2pts_utils as utils


def fit_2pts_single_corr(
    ens, meson, mom, 
    data_dir, 
    binsize, 
    smslist, 
    nstates, 
    trange, 
    saveto     = None, 
    meff       = True, 
    trange_eff = None, 
    jkfit      = False,
    wpriors    = False,
    **cov_specs
):
    # Initialize objets and fits
    io   = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smslist
    )

    # Calculate effective mass 
    if meff:
        tr = trange if trange_eff is None else trange_eff 
        effm,effa = stag.meff(tr,**cov_specs)
    else:
        effm, effa = None, None

    # Perform fit
    pr = stag.priors(nstates,Meff=effm,Aeff=effa)

    fit = stag.fit(
        Nstates = nstates,
        trange  = trange,
        priors  = pr,
        verbose = True,
        **cov_specs
    )

    if jkfit:
        fitjk = stag.fit(
            Nstates = nstates,
            trange  = trange,
            priors  = pr,
            jkfit   = True,
            verbose = False,
            **cov_specs
        )

    fitres = stag.fit_result(
        nstates,
        trange,
        verbose = True,
        priors  = pr if wpriors else None
    )
    if saveto is not None:
        dump_fit_object(saveto,fit,**fitres)

        if jkfit:
            name = f'{saveto}_fit.pickle'
            with open(name,'wb') as handle:
                pickle.dump(fitjk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return stag,fitres




def log(tag,ens,meson,mom,data_dir,smlist,trange,trange_eff,nexc,saveto,JKFIT,**cov_specs):
    return f'''
# ================================== ({tag}) ==================================
# fit_2pts_single_corr from {__file__} called at {datetime.datetime.now()} with
#        ens        = {ens}                                                           
#        meson      = {meson}                                                       
#        mom        = {mom}                                                        
#        data_dir   = {data_dir}                                                      
#        smlist     = {smlist}                                                      
#        trange     = {trange}                                                       
#        trange_eff = {trange_eff}                                                   
#        nstates    = {nexc}                                                       
#        saveto     = {saveto}                                                       
#        jkfit      = {JKFIT}                                                        
#        meff       = True,
#        diag       = {cov_specs.get("diag")},
#        block      = {cov_specs.get("block")},                                                       
#        shrink     = {cov_specs.get("shrink")},
#        scale      = {cov_specs.get("scale")},
#        svd        = {cov_specs.get("cutsvd")},                     
# =============================================================================
''' 




prs = argparse.ArgumentParser(usage=USAGE)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('-m','--meson'   , type=str,  nargs='+',  default=None)
prs.add_argument('-mm','--mom'    , type=str,  nargs='+',  default=None)
prs.add_argument('--saveto'       , type=str,  default=None)
prs.add_argument('--jkfit'        , action='store_true')
prs.add_argument('--override'     , action='store_true')
prs.add_argument('--logto'        , type=str, default=None)
prs.add_argument('--debug'        , action='store_true')
prs.add_argument('--diag'         , action='store_true')
prs.add_argument('--block'        , action='store_true')
prs.add_argument('--scale'        , action='store_true')
prs.add_argument('--shrink'       , action='store_true')
prs.add_argument('--svd'          , type=float, default=None)
prs.add_argument('--no_priors_chi', action='store_true')
prs.add_argument('--verbose'      , action='store_true')
prs.add_argument('--plot_eff'     , action='store_true')
prs.add_argument('--plot_fit'     , action='store_true')
prs.add_argument('--show'         , action='store_true')

def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ENSEMBLE_LIST = args.ensemble if args.ensemble is not None else config['ensemble']['list']
    MESON_LIST    = args.meson    if args.meson    is not None else config['meson']['list']
    MOM_LIST      = args.mom      if args.mom      is not None else []

    print(f'# B2heavy AUTO: ------------- RUNNING 2pt fits for -------------')
    print(f'# B2heavy AUTO: ------------- {ENSEMBLE_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MESON_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MOM_LIST} -------------')

    JKFIT  = True if args.jkfit else False

    aux = []
    for ens in ENSEMBLE_LIST:
        for meson in MESON_LIST:
            for mom in (MOM_LIST if MOM_LIST else config['data'][ens]['mom_list']):
            # for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][meson]['mom'].keys()):

                tag = config['fit'][ens][meson]['mom'][mom]['tag']
                data_dir   = config['data'][ens]['data_dir']
                binsize    = config['data'][ens]['binsize'] 
                smlist     = config['fit'][ens][meson]['smlist'] 
                nstates    = config['fit'][ens][meson]['mom'][mom]['nstates'] 
                trange     = tuple(config['fit'][ens][meson]['mom'][mom]['trange']) 
                trange_eff = tuple(config['fit'][ens][meson]['mom'][mom]['trange_eff']) 
                cutsvd     = config['fit'][ens][meson]['mom'][mom]['svd']

                # 
                SAVETO = DEFAULT_ANALYSIS_ROOT if args.saveto=='default' else args.saveto
                saveto = None
                if SAVETO is not None:
                    if not os.path.exists(SAVETO):
                        raise Exception(f'{SAVETO} is not an existing location.')
                    else:
                        if not JKFIT:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}' if args.saveto is not None else None
                        else:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}_jk' if args.saveto is not None else None
                    
                        # Check if there is an already existing analysis =====================================
                        if os.path.exists(f'{saveto}_fit.pickle'):
                            if args.override:
                                print(f'Already existing analysis for {tag}, but overriding...')
                            else:
                                print(f'Analysis for {tag} already up to date')


                # Perform analysis ===================================================================

                cov_specs = dict(
                    diag   = args.diag,
                    block  = args.block,
                    scale  = args.scale,
                    shrink = args.shrink,
                    cutsvd = args.svd if args.svd is not None else cutsvd
                )

                stag,fitres = fit_2pts_single_corr(
                    ens, meson, mom, 
                    data_dir, 
                    binsize, 
                    smlist, 
                    nstates, 
                    trange, 
                    saveto     = saveto, 
                    meff       = True, 
                    trange_eff = trange_eff, 
                    jkfit      = JKFIT,
                    wpriors    = False if args.no_priors_chi else True,
                    **cov_specs
                )

                aux.append({
                    'ensemble'     : ens,
                    'meson'        : meson,
                    'momentum'     : mom,
                    'tmin(3+3)'    : trange[0],
                    'tmax'         : trange[1],
                    'svd'          : cov_specs['cutsvd'],
                    'trange_eff'   : trange_eff,
                    'E0'           : stag.fits[nstates,trange].p['E'][0],
                    'chiaug/chiexp': fitres['chi2aug']/fitres['chiexp'],
                    'chi2red'      : fitres['chi2red'],
                    'pexp'         : fitres['pexp'],
                    'pstd'         : fitres['pstd']
                })



                # LOG analysis and PLOTS =======================================================================
                if SAVETO is not None:
                    logfile = f'{saveto}.log' if args.logto==None else args.logto
                    with open(logfile,'w') as f:
                        f.write(
                            log(tag,ens,meson,mom,data_dir,smlist,trange,trange_eff,nstates,saveto,JKFIT,**cov_specs)
                        )

                if args.plot_eff:
                    toplot = stag.meff(trange_eff,**cov_specs,plottable=True)

                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['font.size'] = 12

                    plt.figure(figsize=(12, 6 if len(stag.keys)/len(smlist)==1 else 8))
                    plot_effective_coeffs(trange_eff,*toplot)

                    plt.tight_layout()

                    if SAVETO is not None:
                        plt.savefig(f'{SAVETO}/fit2pt_config_{tag}_eff.pdf')
                        print(f'Effective mass and coeff plot saved to {SAVETO}/fit2pt_config_{tag}_eff.pdf')
                    if args.show:
                        plt.show()

                if args.plot_fit:
                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['font.size'] = 12

                    npol = len(stag.keys)//len(smlist)
                    f, ax = plt.subplots(3,npol,figsize=(12,8))
                    stag.plot_fit(ax,nstates,trange)

                    plt.tight_layout()

                    if SAVETO is not None:
                        plt.savefig(f'{SAVETO}/fit2pt_config_{tag}_fit.pdf')
                        print(f'Effective mass and coeff plot saved to {SAVETO}/fit2pt_config_{tag}_fit.pdf')
                    if args.show:
                        plt.show()              

    df = pd.DataFrame(aux).set_index(['ensemble','meson','momentum'])
    print(df)


if __name__ == "__main__":
    main()


    





    