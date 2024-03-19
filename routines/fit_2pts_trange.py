# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_trange.py --config       [file location of the toml config file]
                          --ensemble     [which ensemble?]                       
                          --meson        [which meson?]                          
                          --mom          [which momentum?]                       
                          --readfrom                      
Examples
python 2pts_fit_stability_test.py --ensemble MediumCoarse --meson Dst --mom 000 --Nstates 1 2 3 --tmins  6 7 8 9 10 --tmaxs 15 --plot --show --saveto . --plot_AIC
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import pickle
import tomllib
import argparse
import os
import datetime

import numpy             as np
import gvar              as gv
import pandas            as pd
import matplotlib.pyplot as plt

from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter    import StagFitter

import fit_2pts_utils as utils



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson'   , type=str)
prs.add_argument('-mm','--mom'    , type=str, nargs='+')
prs.add_argument('--readfrom'     , type=str)
prs.add_argument('--tmin'         , type=int)
prs.add_argument('--tmax'         , type=int)


def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ens = args.ensemble
    mes = args.meson   

    if args.mom is None:
        momlist = config['data'][ens]['mom_list']
    else:
        momlist = args.mom

    READFROM = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom



    for mom in momlist:
        tag = config['fit'][ens][mes]['mom'][mom]['tag']

        with open(f'{READFROM}/fit2pt_stability_test_{tag}.pickle','rb') as f:
            st = pickle.load(f)

            trange = (args.tmin,args.tmax)

            fit2 = st[2,trange]
            fit3 = st[3,trange]




if __name__=='__main__':
    main()