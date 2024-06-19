USAGE = '''
python 2pts_fit_config.py --config [file location of the toml config file]         
                   --ensemble [list of ensembles analyzed]                    
                   --ratio    [list of meson analyzed]                        
                   --mom      [list of momenta considered]                    
                   --jkfit    [repeat same fit inside each bin. Default: false]               
                   --readfrom
                   --saveto   [*where* do you want to save files.]             
                   --logto    [Log file name]                                 
                   --override [do you want to override pre-existing analysis?]

                   --diag
                   --block
                   --svd
                   --scale         [rescale the covariance matrix with the diagonal]
                   --shrink        [shrink covariance matrix]
                   
                   --no_priors_chi [don't consider priors in the calculation of chi2]

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
import matplotlib.pyplot as plt
import gvar              as gv
import numpy             as np
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)
import pandas            as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from b2heavy import FnalHISQMetadata

from b2heavy.ThreePointFunctions.utils      import read_config_fit, dump_fit_object
from b2heavy.ThreePointFunctions.types3pts  import Ratio, RatioIO, ratio_prerequisites, ratiofmt
from b2heavy.ThreePointFunctions.fitter3pts import RatioFitter, phys_energy_priors

import fit_2pts_utils as utils

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT



def fit_ratio(
    ratio:RatioFitter,
    nstates,
    trange,
    priors  = None,
    saveto  = None,
    jkfit   = False,
    wpriors = False,
    **cov_specs
):
    # Perform fit
    fit = ratio.fit(
        Nstates = nstates,
        trange  = trange,
        priors  = priors,
        verbose = False,
        **cov_specs
    )


    if jkfit:
        fitjk = ratio.fit(
            Nstates = nstates,
            trange  = trange,
            priors  = dict(fit.prior),
            jkfit   = True,
            **cov_specs
        )

    fitres = ratio.fit_result(
        nstates,
        trange,
        verbose = True,
        priors = fit.prior if wpriors else None
    )

    if saveto is not None:
        dump_fit_object(saveto,fit,**fitres)

        if jkfit:
            name = f'{saveto}_fit.pickle'
            with open(name, 'wb') as handle:
                pickle.dump(fitjk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return fitres






prs = argparse.ArgumentParser(usage=USAGE)
prs.add_argument('-c','--config'  , type=str,  default='./3pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('-r','--ratio'   , type=str,  nargs='+',  default=None)
prs.add_argument('-m','--mom'     , type=str,  nargs='+',  default=None)
prs.add_argument('--meson'        , type=str,  default='Dst')
prs.add_argument('--saveto'       , type=str,  default=None)
prs.add_argument('--readfrom'     , type=str,  default=None)
prs.add_argument('--jkfit'        , action='store_true', default=False)
prs.add_argument('--override'     , action='store_true')
prs.add_argument('--logto'        , type=str, default=None)
prs.add_argument('--debug'        , action='store_true')
prs.add_argument('--diag'         , action='store_true')
prs.add_argument('--block'        , action='store_true')
prs.add_argument('--scale'        , action='store_true')
prs.add_argument('--shrink'       , action='store_true')
prs.add_argument('--svd'          , type=float, default=None)
# prs.add_argument('--no_priors_chi', action='store_true')
prs.add_argument('--verbose'      , action='store_true')
prs.add_argument('--plot_fit'     , action='store_true')
prs.add_argument('--show'         , action='store_true')





def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    readfrom = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom
    saveto   = DEFAULT_ANALYSIS_ROOT if args.saveto   == 'default' else args.saveto

    ENSEMBLE_LIST = args.ensemble if args.ensemble is not None else config['ensemble']['list']
    RATIO_LIST    = args.ratio    if args.ratio    is not None else config['ratio']['list']
    MOM_LIST      = args.mom      if args.mom      is not None else []
    
    JKFIT  = True if args.jkfit else False


    aux = []
    for ens in ENSEMBLE_LIST:
        for rt in RATIO_LIST:
            ratio = ratiofmt(rt)
            for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][ratio]['mom'].keys()):
                if mom=='000' and ratio not in ['ZRA1','RPLUS']:
                    continue

                print(f'--------------- {ens} ------------ {ratio} ----------- {mom} -----------------')


                tag = config['fit'][ens][ratio]['mom'][mom]['tag']
                data_dir = config['data'][ens]['data_dir']
                binsize  = config['data'][ens]['binsize'] 
                smlist   = config['fit'][ens][ratio]['smlist'] 
                nstates  = config['fit'][ens][ratio]['mom'][mom]['nstates'] 
                trange   = tuple(config['fit'][ens][ratio]['mom'][mom]['trange']) 
                # svd      = config['fit'][ens][ratio]['mom'][mom]['svd']

                #  =======================================================================================
                SAVETO = DEFAULT_ANALYSIS_ROOT if args.saveto=='default' else args.saveto
                saveto = None
                if SAVETO is not None:
                    if not os.path.exists(SAVETO):
                        raise Exception(f'{SAVETO} is not an existing location.')
                    else:
                        if not args.jkfit:
                            saveto = f'{SAVETO}/fit3pt_config_{tag}' if args.saveto is not None else None
                        else:
                            saveto = f'{SAVETO}/fit3pt_config_{tag}_jk' if args.saveto is not None else None
                    
                        # Check if there is an already existing analysis =====================================
                        skip_fit = False
                        if os.path.exists(f'{saveto}_fit.pickle'):
                            if args.override:
                                print(f'Already existing analysis for {tag}, but overriding...')
                            else:
                                print(f'Analysis for {tag} already up to date')
                                
                                fit = read_config_fit(f'fit3pt_config_{tag}',jk=False,path=SAVETO)
                                aux.append({
                                    'ensemble'     : ens,
                                    'ratio'        : ratio,
                                    'momentum'     : mom,
                                    'trange'       : trange,
                                    'chiaug/chiexp': round(fit[0]['chi2aug']/fit[0]['chiexp'],3),
                                    'chi2red'      : round(fit[0]['chi2red'],3),
                                    'pexp'         : fit[0]['pexp'],
                                    'pstd'         : fit[0]['pstd'],
                                    'F0'           : fit[1]['ratio'][0],
                                })

                                continue


                # Perform analysis ===================================================================
                cov_specs = dict(
                    diag   = args.diag,
                    block  = args.block,
                    scale  = args.scale,
                    shrink = args.shrink,
                    cutsvd = args.svd
                )                

                if not skip_fit:
                    # Create ratio object
                    ratio_req = ratio_prerequisites(
                        ens      = ens,
                        ratio    = ratio,
                        mom      = mom,
                        readfrom = readfrom,
                        jk       = args.jkfit,
                    )

                    io = RatioIO(ens,ratio,mom,PathToDataDir=data_dir)
                    robj = RatioFitter(
                        io,
                        jkBin    = binsize,
                        smearing = smlist,
                        **ratio_req
                    )

                    if ratio in ['RPLUS','RMINUS','XF','QPLUS']:
                        meson1 = 'D'
                        meson2 = 'B' 
                    elif ratio in ['ZRA1','RA1','XFSTPAR','R0','R1','XV']:
                        meson1 = 'Dst'
                        meson2 = 'B'
                    elif ratio in ['ZRA1S','RA1S','XFSSTPAR','R0S','R1S','XVS']:
                        meson1 = 'Dsst'
                        meson2 = 'Bs'


                    # Compute priors
                    dE_src = phys_energy_priors(ens,meson1,mom,nstates,readfrom=readfrom,error=1.)
                    dE_snk = phys_energy_priors(ens,meson2,mom,nstates,readfrom=readfrom,error=1.)

                    x,ydata = robj.format(trange,flatten=True)
                    f_0    = gv.gvar(np.mean(ydata).mean,0.1)
                    pr = robj.priors(nstates, K=f_0, dE_src=dE_src, dE_snk=dE_snk)

                    # Perform fit
                    fitres = fit_ratio(
                        robj,
                        nstates,
                        trange,
                        saveto  = saveto, 
                        priors  = pr,
                        jkfit   = JKFIT,  
                        wpriors = True,
                        **cov_specs
                    )

                    f0 = robj.fits[nstates,trange].p['ratio'][0]
                    pval = fitres['pstd']
                
                else:
                    fitres = read_config_fit(
                        f'fit3pt_config_{tag}',
                        jk=args.jkfit,
                        path=SAVETO
                    )

                    if args.jkfit:
                        f0 = fitres['ratio'][0]
                        f0 = gv.gvar(f0.mean(),f0.std()*np.sqrt(len(f0)-1))
                        pval = np.mean(fitres['pstd'])
                        breakpoint()
                    else:
                        f0 = fitres[-1]['ratio'][0]
                        pval = fitres[0]['pstd']

                aux.append({
                    'ensemble' : ens,
                    'ratio'    : ratio,
                    'momentum' : mom,
                    'tmin'     : trange[0],
                    'tmax'     : trange[1],
                    'F0'       : f0,
                    'pval'     : pval
                })

                # Plot ==============================================================================
                if args.plot_fit:
                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['font.size'] = 12

                    f, ax = plt.subplots(1,1,figsize=(4,8))

                    robj.plot_fit(ax,nstates,trange)

                    f0 = pr['ratio'][0]
                    ax.axhspan(ymin=f0.mean-f0.sdev,ymax=f0.mean+f0.sdev,alpha=0.1,color='gray')

                    ax.set_xlim(xmin=-0.5)
                    ax.grid(alpha=0.2)
                    ax.legend()
                    ax.set_title(tag)

                    if SAVETO is not None:
                        plt.savefig(f'{SAVETO}/fit3pt_config_{tag}_fit.pdf')
                        print(f'{ratio} plot saved to {SAVETO}/fit3pt_config_{tag}_fit.pdf')
                    if args.show:
                        plt.show()


    df = pd.DataFrame(aux).set_index(['ensemble','ratio','momentum'])
    print(df)

if __name__=='__main__':
    main()