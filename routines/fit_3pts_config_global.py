import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import pandas            as pd
import lsqfit
import itertools
from tqdm import tqdm
import argparse
import os
import pickle

from b2heavy.ThreePointFunctions.globalfit import RatioSet, show
import fit_2pts_utils as utils
from b2heavy.ThreePointFunctions.utils import dump_fit_object, read_config_fit

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT



prs = argparse.ArgumentParser()
prs.add_argument('-c','--config'  , type=str,  default='./3pts_fit_config_global.toml')
prs.add_argument('-e','--ensemble', type=str,  default=None)
prs.add_argument('--meson'        , type=str,  default=None)
prs.add_argument('-r','--ratio'   , type=str,  nargs='+',  default=None)
prs.add_argument('-m','--mom'     , type=str,  nargs='+',  default=None)

prs.add_argument('--saveto'       , type=str,  default=None)
prs.add_argument('--readfrom'     , type=str,  default=None)
prs.add_argument('--read2pts'     , type=str,  default=None)
prs.add_argument('--jkfit'        , action='store_true', default=False)
prs.add_argument('--override'     , action='store_true')


prs.add_argument('--priors_2pts'  , action='store_true')

prs.add_argument('--diag'         , action='store_true')
prs.add_argument('--block'        , action='store_true')
prs.add_argument('--scale'        , action='store_true')
prs.add_argument('--shrink'       , action='store_true')
prs.add_argument('--svd'          , type=float, default=None)

prs.add_argument('--verbose'      , action='store_true')
prs.add_argument('--plot_fit'     , action='store_true')
prs.add_argument('--show'         , action='store_true')



def check_ratio_list(meson,ratlist):
    for ratio in ratlist:
        if meson=='Dst':
            if ratio.upper() not in ['RA1','ZRA1','XFSTPAR','XFSTBOT','R0','R1','XV']:
                raise KeyError(f'{ratio} is not one of {ratlist} for {meson}')
        elif meson=='D':
            if ratio.upper() not in ['XF','QPLUS','RPLUS','RMINUS']:
                raise KeyError(f'{ratio} is not one of {ratlist} for {meson}')
    return


def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    readfrom = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom
    read2pts = DEFAULT_ANALYSIS_ROOT if args.read2pts == 'default' or args.read2pts is None else args.read2pts
    saveto   = DEFAULT_ANALYSIS_ROOT if args.saveto   == 'default' else args.saveto

    ens = args.ensemble

    JKFIT  = True if args.jkfit else False

    tag      = config['fit'][ens]['tag']
    data_dir = config['data'][ens]['data_dir']
    binsize  = config['data'][ens]['binsize'] 

    momlist = config['fit'][ens]['momlist']
    ratlist = config['fit'][ens]['ratlist']
    smslist = config['fit'][ens]['smslist']
    toremove = [tuple(rm) for rm in config['fit'][ens]['remove']]

    tmin    = config['fit'][ens]['tmin']

    check_ratio_list(args.meson,ratlist)

    # ====================================================================================
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
                    skip_fit = True
                    fit = read_config_fit(f'fit3pt_config_{ens}_global',path=readfrom)
                    

    if not skip_fit:
        cov_specs = dict(
            diag   = args.diag,
            block  = args.block,
            scale  = args.scale,
            shrink = args.shrink,
            cutsvd = args.svd
        )   

        # Initialize objects and collect data
        rset = RatioSet(ens,momlist,ratlist,smslist)
        rset.collect(data_dir,read2pts)

        rset.remove(*toremove)

        # Set prior from 2pts fits
        if args.priors_2pts:
            tag = f'fit2pt_config_{ens}_{args.meson}_000'
            fit,pars = read_config_fit(tag,path=read2pts)
            dE_D = np.exp(pars['dE'][1])

            tag = f'fit2pt_config_{ens}_B_000'
            fit,pars = read_config_fit(tag,path=read2pts)
            dM_B = np.exp(pars['dE'][1])

            priors = rset.params(
                dE_D = gv.gvar(dE_D.mean,0.5),
                dM_D = gv.gvar(dE_D.mean,0.5),
                dM_B = gv.gvar(dM_B.mean,0.5),
            )

        else:
            priors = rset.params()

        # Perform the global fit
        fit = rset.fit(
            tmin   = tmin,
            priors = priors,
            **cov_specs
        )
        fitres = rset.fit_result(
            tmin = tmin,
            fit  = fit,
            verbose = True,
            priors = fit.prior
        )

        if saveto is not None:
            dump_fit_object(saveto,fit,**fitres)

        if JKFIT:
            fitjk = rset.fit(
                tmin   = tmin,
                priors = priors,
                jkfit  = True,
                **cov_specs
            )

            if saveto is not None:
                name = f'{saveto}_fit.pickle'
                with open(name, 'wb') as handle:
                    pickle.dump(fitjk, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(show(rset.cases,fit.p))

    else:
        cases = []
        for k,p in fit[1].items():
            if k.endswith('f0'):
                ratio,mom,_ = k.split('_')
                cases.append((mom,ratio))
        print(show(cases,fit[-1]))

if __name__=='__main__':
    main()