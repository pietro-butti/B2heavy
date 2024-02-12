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


# def extract_parameters(location,config,jkfile):
#     pars = {}
    
#     with open(os.path.join(location,config),'rb') as f:
#         fit = pickle.load(f)

#         Nexc,trange = fit['info']
#         pars['tmin'],pars['tmax'] = trange
#         pars['Nexc'] = f'{Nexc}+{Nexc}'

#         Ndata = len(fit['y'])
#         Npars = len(np.concatenate(list(fit['p'].values())))
#         pars['Ndof'] = Ndata-Npars

#         chi2red = round(fit['chi2red'],1)
#         pars['chi2'] = chi2red
#         pars['pval'] = round(fit['pvalue'],2)

#     with open(os.path.join(location,jkfile),'rb') as g:
#         jkfit = pickle.load(g)

#         E0 = gv.mean(jkfit['E'][:,0])
#         err = np.std(E0) * np.sqrt(len(E0)-1)  
#         pars['E0'] = gv.gvar(E0.mean(),err)

#         for k,z in jkfit.items():
#             if k.startswith('Z_'):
#                 sm = k.split('_')[1]
#                 if len(sm.split('-'))==1:
#                     Z = np.exp(gv.mean(z[:,0]))
#                     pars[k] = gv.gvar(Z.mean(),np.std(Z) *  np.sqrt(len(Z)-1))


#     return pars    



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('--read_from'    , type=str)
prs.add_argument('--saveto'       , type=str, default=None)
prs.add_argument('--latex'        , action='store_true')

def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)
    
    read_from = DEFAULT_ANALYSIS_ROOT if args.read_from=='default' else args.read_from
    if not os.path.exists(read_from):
        raise Exception(f'{read_from} is not a valid location')

    ENSEMBLE_LIST = config['ensemble']['list']
    MESON_LIST    = config['meson']['list']

    DF = {
        'ens':[],'mes':[],'mom':[],
        'fit':[],'jkfit':[],
        'trange':[], 'Nstates':[],'Ndof':[],
        'pvalue':[],'chi2red':[],
        'E0':[]#,'Z':[]
    }

    for ens in ENSEMBLE_LIST:
        for mes in MESON_LIST:
            MOMENTUM_LIST = config['data'][ens]['mom_list']

            for mom in MOMENTUM_LIST:
                tag = f'{ens}_{mes}_{mom}'

                DF['ens'].append(ens)
                DF['mes'].append(mes)
                DF['mom'].append(mom)

                fit_file = os.path.join(read_from,f'fit2pt_config_{tag}_fit.pickle')
                fit      = os.path.exists(fit_file)
                DF['fit'].append('x' if fit else ' ')

                jkfit_file = os.path.join(read_from,f'fit2pt_config_{tag}_jk_fit.pickle')
                jkfit      = os.path.exists(jkfit_file)
                DF['jkfit'].append('x' if jkfit else ' ')

                if fit:
                    fit,fitp,fity = utils.read_config_fit(f'fit2pt_config_{tag}',path=read_from)

                    tspan  = np.concatenate(fit['x'])
                    trange = (min(tspan),max(tspan))
                    Nexc   = len(fit['priors']['E'])//2
                    Npol = 1 if 'Unpol' in list(fitp.keys())[-1] else 3

                    DF['trange'].append(trange)
                    DF['Nstates'].append(Nexc)
                    DF['Ndof'].append(utils.Ndof(Npol,trange,Nexc))
                    

                    if not jkfit:
                        DF['pvalue'].append(fit['pvalue'])
                        DF['chi2red'].append(fit['chi2red'])

                        DF['E0'].append(fitp['E'][0])
                        # for k,z in fitp.items():    
                        #     if k.startswith('Z_'):
                        #         sm = k.split('_')[1]
                        #         if len(sm.split('-'))==1:
                        #             Z = np.exp(z[0])
                        #             DF['Z'].append(f'{k}_{Z}') 
                    else:
                        pass

                else:
                    DF['trange'].append(' ')
                    DF['Nstates'].append(' ')
                    DF['Ndof'].append(' ')
                    DF['pvalue'].append(' ')
                    DF['chi2red'].append(' ')
                    DF['E0'].append(' ')     
                    pass


    DF = pd.DataFrame(DF)
    DF.set_index(['ens','mes','mom'],inplace=True)

    DF.style \
    .format(precision=3, thousands=".", decimal=",") \
    .format_index(str.upper, axis=1)

    print(DF.to_string())


if __name__ == "__main__":
    main()