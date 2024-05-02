# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python fits_2pts_summarize.py --config    [file location of the toml config file]         
                              --read_from [*where is the analysis located?* (Defaults to DEFAULT_ANALYSIS_ROOT)]
                              --latex     [do you want to produce a latex table?]
                              --saveto    [*where* do you want to save the latex table? (defaults to DEFAULT_ANALYSIS_ROOT/REPORTER)] 

**WARNING**
data has to be contained in a subfolder of `read_from`, named with the name of the ensemble. E.g.:
<read_from>/Coarse-1/fit2pt_config_Coarse-1_Dsst_000.pickle

Examples
'''

import argparse
import pickle
import sys
import tomllib
import os
import pandas as pd
from prettytable import PrettyTable
import gvar as gv
import numpy as np

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import fit_2pts_utils as utils

from b2heavy.ThreePointFunctions.utils import read_config_fit


prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c2','--config2' , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-c3','--config3' , type=str,  default='./3pts_fit_config.toml')
prs.add_argument('--readfrom'    , type=str)
prs.add_argument('--saveto'      , type=str, default=None)
prs.add_argument('--latex'       , action='store_true')
prs.add_argument('--only2'       , action='store_true')

def main():
    args = prs.parse_args()

    fit2 = utils.load_toml(args.config2)
    fit3 = utils.load_toml(args.config3)

    READFROM = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom


    df = []
    for ens in set(fit2['fit']).intersection(fit3['fit']):
        for mom in fit2['data'][ens]['mom_list']:
            specs = fit2['fit'][ens]['Dst']['mom'][mom]
            fit2pts,p2 = read_config_fit(
                tag  = f'fit2pt_config_{specs["tag"]}',
                path = READFROM
            )


            d = {
                'ens'        : ens,
                'mom'        : mom,
                'E_0 (D*)'   : p2['E'][0],
                'Z_1S_Par'   : np.exp(p2['Z_1S_Par'][0]  ) if mom not in ['000','110','211','222'] else 'X',
                'Z_1S_Bot'   : np.exp(p2['Z_1S_Bot'][0]  ) if mom not in ['000','110','211','222'] else 'X',
                'Z_1S_Unpol' : np.exp(p2['Z_1S_Unpol'][0]) if mom     in ['000','110','211','222'] else 'X',


                'Z_d_Par\u221A2E'   : np.exp(p2['Z_1S_Par'][0]  )* np.sqrt(2*p2['E'][0]) if mom not in ['000','110','211','222'] else 'X',
                'Z_d_Bot\u221A2E'   : np.exp(p2['Z_1S_Bot'][0]  )* np.sqrt(2*p2['E'][0]) if mom not in ['000','110','211','222'] else 'X',
                'Z_d_Unpol\u221A 2E' : np.exp(p2['Z_1S_Unpol'][0])* np.sqrt(2*p2['E'][0]) if mom     in ['000','110','211','222'] else 'X',

                'chi2/chiexp': f'{fit2pts["chi2red"]/fit2pts["chi2exp"]:.2f}',
                'p-value'    : f'{fit2pts["pvalue"]:.2f}'
            }

            if not args.only2:
                for ratio in fit3['fit'][ens].keys(): 
                    if (mom=='000' and ratio!='RA1') or mom in ['110','211']:
                        d[ratio]                    = 'X'
                        d[f'chi2/chiexp [{ratio}]'] = 'X'
                        d[f'p-value     [{ratio}]'] = 'X'

                    try:

                        fit3pts,p3 = read_config_fit(tag=f'fit3pt_config_{ens}_{ratio}_{mom}',path=DEFAULT_ANALYSIS_ROOT)
                        try:
                            c = fit3pts['chi2red']/fit3pts['chi2exp']
                        except TypeError:
                            c = -1

                        d[ratio]                    = p3['ratio'][0]
                        d[f'chi2/chiexp [{ratio}]'] = f'{c:.2f}'
                        d[f'p-value     [{ratio}]'] = f'{fit3pts["pvalue"]:.2f}'
                    except FileNotFoundError:
                        continue

            df.append(d)


    df = pd.DataFrame(df).set_index(['ens','mom']).replace('X',' ')

    print(df)




if __name__=='__main__':
    main()