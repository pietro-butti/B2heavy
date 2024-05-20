# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python fit_3pts_recoil.py --config   [file location of the toml config file]
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
from b2heavy.ThreePointFunctions.utils import read_config_fit
from fit_2pts_dispersion_relation import mom_to_p2


prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./3pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str, nargs='+', default=None)
prs.add_argument('-mm','--momlist', type=str, nargs='+', default=[])
prs.add_argument('--jkfit', action='store_true')
prs.add_argument('--readfrom', type=str, default='./')
prs.add_argument('--saveto',   type=str, default='./')
prs.add_argument('--override', action='store_true')
prs.add_argument('--plot', action='store_true')
prs.add_argument('--show', action='store_true')



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
        
        # Extract values of M1 and M2 from dispersion relation
        try:
            file = os.path.join(readfrom,f'fit2pts_dispersion_relation_{ens}_Dst.pickle')
            with open(file,'rb') as f:
                aux = gv.load(f)
            m1,m2 = aux['M1'], aux['M2']
        except FileNotFoundError:
            print(f'Dispersion relation analysis for {ens} not found...')
            print(f'Searching in [{file}]')
            continue



        for mom in (MOM_LIST if MOM_LIST else config['fit'][ens]['xfstpar']['mom'].keys()):
            # Compute recoil parameter from dispersion relation
            w1 = np.sqrt(1+mom_to_p2(mom,L=lvol)/m1**2)
            w2 = np.sqrt(1+mom_to_p2(mom,L=lvol)/m2**2)


            # Compute recoil parameter from ratio
            try:
                res = read_config_fit(
                    f'fit3pt_config_{ens}_xfstpar_{mom}',
                    path=readfrom,
                    jk = args.jkfit
                )
            except FileNotFoundError:
                print(f'XFSTPAR not calculated for ({ens},{mom})...')
                continue   
            
            if not args.jkfit:
                xf = res[-1]['ratio'][0]
            else:
                f0 = res['ratio'][0] 
                xf = gv.gvar(f0.mean(),f0.std()*np.sqrt(len(f0)-1))

            wr = (1+xf**2)/(1-xf**2)

            wrecoil.append({
                'ensemble' : ens,
                'p'        : mom,
                'from_xf'  : wr,
                'from_m1'  : w1,
                'from_m2'  : w2
            })


    df = pd.DataFrame(wrecoil).set_index(['ensemble','p'])
    print(df)


    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    f, ax = plt.subplots(1,len(ENSEMBLE_LIST),figsize=(8,3),sharey=True)
    for i,ens in enumerate(ENSEMBLE_LIST):
        axi = ax[i] if len(ENSEMBLE_LIST)>1 else ax

        ps = []
        for j,mom in enumerate(['100','200','300','400']):
            if (ens,mom) in df.index:
                wrecoils = df.loc[ens,mom].values

                p2 = mom_to_p2(mom,L=lvol)

                axi.errorbar(p2-0.005,wrecoils[0].mean,wrecoils[0].sdev, fmt='o', color='C0', capsize=2.5)            
                axi.errorbar(p2      ,wrecoils[1].mean,wrecoils[1].sdev, fmt='o', color='C1', capsize=2.5)            
                axi.errorbar(p2+0.005,wrecoils[2].mean,wrecoils[2].sdev, fmt='o', color='C2', capsize=2.5)            

                ps.append(mom)

        # axi.set_ylabel(r'$w$')
        axi.set_xticks([mom_to_p2(mom,L=lvol) for mom in ps])
        axi.set_xticklabels(ps)
        axi.set_title(ens)
        axi.grid(alpha=0.2)
        axi.set_xlabel(r'$\mathbf{p}^2$')

    axm1 = ax[0] if len(ENSEMBLE_LIST)>1 else ax
    axm1.set_ylabel(r'$w$')


    axm1.errorbar([],[],[], fmt='o', color='C0', capsize=2.5, label=r'from $x_f$')            
    axm1.errorbar([],[],[], fmt='o', color='C1', capsize=2.5, label=r'from $aM_1$')            
    axm1.errorbar([],[],[], fmt='o', color='C2', capsize=2.5, label=r'from $aM_2$')            
    axm1.legend()

    plt.tight_layout()
    plt.savefig(f'{saveto}recoil_parameter.pdf')

    if args.show:
        plt.show()



if __name__ =="__main__":
    main()