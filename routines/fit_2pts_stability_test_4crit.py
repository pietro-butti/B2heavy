# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_stability_test.py --config       [file location of the toml config file]
                                  --ensemble     [which ensemble?]                       
                                  --meson        [which meson?]                          
                                  --mom          [which momentum?]                       
                                  --prior_trange [trange for effective mass priors]
                                  --Nstates      [list of N for (N+N) fit (listed without comas)]
                                  --tmins        [list of tmins (listed without commas)]
                                  --tmaxs        [list of tmaxs (listed without commas)]
                                  --nochipr      [compute chi2exp with priors?]
                                  --saveto       [where do you want to save the analysis?]
                                  --readfrom    [name of the .pickle file of previous analysis]
                                  --override     [ ]
                                  --not_average  [list of tmins that do not have to be taken in model average]
                                  --show         [do you want to display the plot with plt.show()?]
                                  --plot         [do you want to plot data?]
                                  --plot_ymax    [set maximum y in the plot]
                                  --plot_ymin    [set minimum y in the plot]
                                  --plot_AIC     [do you want to plot also the AIC weight?]
Examples
python 2pts_fit_stability_test.py --ensemble MediumCoarse --meson Dst --mom 000 --Nstates 1 2 3 --tmins  6 7 8 9 10 --tmaxs 15 --plot --show --saveto . --plot_AIC
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import pickle
import tomllib
import argparse
import itertools
import os
import datetime
import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)
import numpy             as np
import gvar              as gv
import pandas            as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from b2heavy.FnalHISQMetadata            import params as mData
from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter    import StagFitter

import fit_2pts_utils as utils
from fit_2pts_stability_test import stability_test_model_average, read_results_stability_test



from b2heavy.TwoPointFunctions.utils import p_value

# chi2  = float(fitres['chi2'])
# nconf = stag.data.shape[-2]

# p_value(chi2,1017,ndof)




def chi2pr(fit):
    c = 0
    for k,p in fit.prior.items():
        c += sum((gv.mean(p)-fit.pmean[k])**2/gv.sdev(p)**2)
    return c

def stability_test_fit(
    ens, meson, mom_list,
    data_dir,binsize,smslist,
    prior_trange,
    tmin2, tmin3,
    tmax_err = 0.3, nexcrange=[2,3],
    chipr=True,
    **cov_specs
    ):



    aux = []
    for mom in mom_list:
        io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
        stag = StagFitter(
            io       = io,
            jkBin    = binsize,
            smearing = smslist
        )

        effm,effa = stag.meff(prior_trange[mom],**cov_specs)

        # infer tmax
        tmax = stag.tmax(threshold=tmax_err)
        tmax_range = np.arange(tmax-2,tmax+3)

        # tmin range
        tmin_range = {
            2: np.arange(tmin2-2,tmin3+3),
            3: np.arange(tmin3-2,tmin3+3)
        }

        for nstates in nexcrange:
            for tmax in tmax_range:
                for tmin in tmin_range[nstates]:
                    trange = (tmin,tmax)

                    pr = stag.priors(nstates,Meff=effm,Aeff=effa)
                    fit = stag.fit(
                        Nstates = nstates,
                        trange  = trange,
                        priors  = pr,
                        **cov_specs
                    )

                    ndof    = len(fit.y) - sum([len(pr[k]) for k in pr]) 
                    chi2red = fit.chi2-chi2pr(fit)
                    nconf   = stag.io.nConf  
                    pval    = round(p_value(chi2red,nconf,ndof), ndigits=3)

                    d = stag.fit_result(nstates,trange,priors=pr)
                    print(f'# p-value (doc) = {pval}')
                    d = dict(
                        x       = fit.x,
                        y       = fit.y,
                        p       = fit.p,
                        pexp    = d['pvalue'],
                        pvalue  = pval,
                        chi2    = fit.chi2,
                        chi2red = d['chi2'],
                        chiexp  = d['chiexp']
                    )

                    aux.append(d)

    print(pd.DataFrame(d))

    return aux



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson'   , type=str)
prs.add_argument('-mm','--mom_list'    , type=str, nargs='+')
prs.add_argument('--Nstates'      , type=int, nargs='+')
prs.add_argument('--tmin2'      , type=float)
prs.add_argument('--tmin3'      , type=float)
prs.add_argument('--tmax_err'   , type=float, default=None)

prs.add_argument('--shrink', action='store_true')
prs.add_argument('--scale' , action='store_true')
prs.add_argument('--diag'  , action='store_true')
prs.add_argument('--block' , action='store_true')
prs.add_argument('--svd'   , type=float, default=None)

prs.add_argument('--nochipr', action='store_false')

prs.add_argument('--saveto'  , type=str,  default=None)
prs.add_argument('--readfrom', type=str,  default=None)
prs.add_argument('--override', action='store_true')



def main():
    args = prs.parse_args()

    ens = args.ensemble
    mes = args.meson   

    config_file = args.config
    config = utils.load_toml(config_file)

    mom_list = list(config['fit'][ens][mes]['mom'].keys())# if not args.mom else args.mom

    Tmin2 = args.tmin2/(mData(ens)['aSpc']).mean
    Tmin3 = args.tmin3/(mData(ens)['aSpc']).mean

    cov_specs = dict(
        shrink = args.shrink ,
        scale  = args.scale  ,
        diag   = args.diag   ,
        block  = args.block  ,
        cutsvd = args.svd   
    )

    trange_eff = {mom: tuple(config['fit'][ens][mes]['mom'][mom]['trange_eff']) for mom in mom_list}

    fits = stability_test_fit(
        ens          = args.ensemble,
        meson        = args.meson,
        mom_list     = mom_list,
        data_dir     = config['data'][ens]['data_dir'],
        binsize      = config['data'][ens]['binsize'],
        smslist      = config['fit'][ens][mes]['smlist'],
        prior_trange = trange_eff,
        tmin2        = int(Tmin2),
        tmin3        = int(Tmin3),
        tmax_err     = args.tmax_err if args.tmax_err is not None else 0.3,
        chipr        = True,  
        **cov_specs
    ) 




        # e0,syst,sum_ws = stability_test_model_average(
        #     fits, config['data'][ens]['Nt'], args.not_average
        # )

if __name__ == "__main__":
    main()