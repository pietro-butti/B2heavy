# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_disp_rel.py --config   [file location of the toml config file]
                        --ensemble [ensemble analyzed]
                        --momlist  [list of moments that have to be fitted to disp. relation]
                        --jkfit    [use jackknifed fit]
                        --readfrom [*where* are the `config` analysis results?]
                        --plot     [do you want to plot?]
                        --showfig  [do you want to display plot?]
                        --saveto   [*where* do you want to save files?]

Examples
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import argparse
import pickle
import sys
import os
import numpy as np
import gvar as gv
import lsqfit
import tomllib

import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import fit_2pts_utils as utils

from b2heavy import FnalHISQMetadata
from fit_2pts_dispersion_relation import mom_to_p2



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./3pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-mm','--momlist', type=str, nargs='+', default=[])
prs.add_argument('--jkfit', action='store_true')
prs.add_argument('--readfrom', type=str, default='./')
prs.add_argument('--saveto',   type=str, default='./')
prs.add_argument('--override', action='store_true')
prs.add_argument('--plot', action='store_true')
prs.add_argument('--showfig', action='store_true')



def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    readfrom = f'{DEFAULT_ANALYSIS_ROOT}/' if args.readfrom=='default' else args.readfrom
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    saveto = f'{DEFAULT_ANALYSIS_ROOT}/' if args.saveto=='default' else args.saveto
    if not os.path.exists(saveto):
        raise NameError(f'{saveto} is not an existing location')
    saveplot = f'{DEFAULT_ANALYSIS_ROOT}' if args.saveto=='default' else args.saveto


    ENSEMBLE_LIST = args.ensemble if args.ensemble is not None else config['ensemble']['list']
    MOM_LIST      = args.momlist  if args.momlist  is not None else []

    wrecoil = []
    for ens in ENSEMBLE_LIST:
        lvol = FnalHISQMetadata.params(ens)['L']

        file = os.path.join(readfrom,f'fit2pts_dispersion_relation_{ens}_Dst.pickle')
        with open(file,'rb') as f:
            aux = gv.load(f)
        m1,m2 = aux['M1'], aux['M2']

        for mom in (MOM_LIST if MOM_LIST else config['fit'][ens]['xfstpar']['mom'].keys()):
            # Compute recoil parameter -----------------------------------
            fit,p = utils.read_config_fit(f'fit3pt_config_{ens}_xfstpar_{mom}',path=readfrom)
            xf = p['ratio'][0]
            wr = (1+xf**2)/(1-xf**2)

            # Compute recoil parameter from dispersion relation
            w1 = np.sqrt(1+mom_to_p2(mom,L=lvol)/m1**2)
            w2 = np.sqrt(1+mom_to_p2(mom,L=lvol)/m2**2)


            # wrecoil['ensemble','mom','wr','w1','w2'] = [ens,mom,wr,w1,w2]
            wrecoil.append(
                dict(
                    ensemble = ens,
                    p        = mom,
                    from_xf  = wr,
                    from_m1  = w1,
                    from_m2  = w2
                )
            )

    df = pd.DataFrame(wrecoil).set_index(['ensemble','p'])


    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    f, ax = plt.subplots(len(ENSEMBLE_LIST),1,figsize=(6,7),sharex=True)
    for i,ens in enumerate(ENSEMBLE_LIST):
        for j,mom in enumerate(['100','200','300']):
            wrecoils = df.loc[ens,mom].values

            p2 = mom_to_p2(mom,L=lvol)

            ax[i].errorbar(p2-0.005,wrecoils[0].mean,wrecoils[0].sdev, fmt='o', color='C0', capsize=2.5)            
            ax[i].errorbar(p2      ,wrecoils[1].mean,wrecoils[1].sdev, fmt='o', color='C1', capsize=2.5)            
            ax[i].errorbar(p2+0.005,wrecoils[2].mean,wrecoils[2].sdev, fmt='o', color='C2', capsize=2.5)            


        ax[i].set_ylabel(r'$w$')
        ax[i].set_xticks([mom_to_p2(mom,L=lvol) for mom in ['100','200','300']])
        ax[i].set_xticklabels(['100','200','300'])
        ax[i].set_title(ens)

    ax[-1].set_xlabel(r'$\mathbf{p}$')


    plt.tight_layout()
    plt.show()





if __name__ =="__main__":
    main()