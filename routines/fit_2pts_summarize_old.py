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

def extract_parameters(location,config,jkfile):
    pars = {}
    
    with open(os.path.join(location,config),'rb') as f:
        fit = pickle.load(f)

        Nexc,trange = fit['info']
        pars['tmin'],pars['tmax'] = trange
        pars['Nexc'] = f'{Nexc}+{Nexc}'

        Ndata = len(fit['y'])
        Npars = len(np.concatenate(list(fit['p'].values())))
        pars['Ndof'] = Ndata-Npars

        chi2red = round(fit['chi2red'],1)
        pars['chi2'] = chi2red
        pars['pval'] = round(fit['pvalue'],2)

    with open(os.path.join(location,jkfile),'rb') as g:
        jkfit = pickle.load(g)

        E0 = gv.mean(jkfit['E'][:,0])
        err = np.std(E0) * np.sqrt(len(E0)-1)  
        pars['E0'] = gv.gvar(E0.mean(),err)

        for k,z in jkfit.items():
            if k.startswith('Z_'):
                sm = k.split('_')[1]
                if len(sm.split('-'))==1:
                    Z = np.exp(gv.mean(z[:,0]))
                    pars[k] = gv.gvar(Z.mean(),np.std(Z) *  np.sqrt(len(Z)-1))


    return pars    


MESON_LATEX = {
    'D'   : r'$D$',
    'Ds'  : r'$D_s$',
    'Dst' : r'$D^*$',
    'Dsst': r'$D_s^*',
    'B': r'$B',
    'Bs': r'$B_s',
}


COLS = [r'$N_{exc}$',r'$\frac{t}{a}$',r'$aE(\mathbf{p})$',r'$\mathcal{Z}_{1S}$',r'$\mathcal{Z}_{d}$',r'$\chi^2$/dof',r'$p$-value']
def tabulate_parameters(pars):
    Z1S = pars['Z_1S_Par'] if 'Z_1S_Par' in pars else pars['Z_1S_Unpol']
    Zd  = pars['Z_d_Par']  if 'Z_d_Par'  in pars else pars['Z_d_Unpol']

    row = {
        r'$N_{exc}$'         : pars['Nexc'],
        r'$\frac{t}{a}$'     : f"({pars['tmin']},{pars['tmax']})",
        r'$aE(\mathbf{p})$'  : f"{pars['E0']}",
        r'$\mathcal{Z}_{1S}$': f'{Z1S}',
        r'$\mathcal{Z}_{d}$' : f'{Zd}',
        r'$\chi^2$/dof'      : f"{pars['chi2']}/{pars['Ndof']}",
        r'$p$-value'         : f"{pars['pval']}"
    }
    return row

def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('--read_from'    , type=str)
prs.add_argument('--latex'        , action='store_true')
prs.add_argument('--saveto'       , type=str, default=None)

def main():
    args = prs.parse_args()

    config_file = args.config
    config = load_toml(config_file)
    

    read_from = DEFAULT_ANALYSIS_ROOT if args.read_from=='default' else args.read_from
    if not os.path.exists(read_from):
        raise ValueError(f'{read_from} is not a valid location')

    ENSEMBLE_LIST = config['ensemble']['list']
    MESON_LIST    = config['meson']['list']
    MOMENTUM_LIST = config['momenta']['list']

    tab = PrettyTable()
    tab.field_names = ['Ensemble','Meson','Momentum','fit','jkfit','stability an.','disp. rel.']

    div = False
    for ens in ENSEMBLE_LIST:
        location = os.path.join(read_from,ens)
        files = os.listdir(location)
        
        dfs  = []
        keys = []
        for mes in MESON_LIST:
            df = pd.DataFrame(columns=COLS,index=[f'({px},{py},{pz})' for px,py,pz in MOMENTUM_LIST])
            dispr = 'x' if f'fit2pt_disp_rel_{ens}_{mes}.pickle' in files else ' '
            for i,mom in enumerate(MOMENTUM_LIST):
                config = f'fit2pt_config_{ens}_{mes}_{mom}.pickle'
                jkfit  = f'fit2pt_config_jk_{ens}_{mes}_{mom}.pickle'
                stabts = f'fit2pt_stability_test_{ens}_{mes}_{mom}.pickle'

                # Produce table for terminal-log
                tab.add_row(
                    [
                        ens,mes,mom,
                        'x' if config in files else ' ',
                        'x' if jkfit  in files else ' ',
                        'x' if stabts in files else ' ',  
                        dispr  
                    ], divider=(i==len(MOMENTUM_LIST)-1)
                )

                # Feed latex table
                if jkfit in files and config in files:
                    p = extract_parameters(location,config,jkfit)
                    px,py,pz = mom
                    df.loc[f'({px},{py},{pz})']  = tabulate_parameters(p)

            dfs.append(df)
            keys.append(MESON_LATEX[mes])
        
        DF = pd.concat(dfs,keys=keys)
        if args.latex:
            lattab = DF.to_latex(caption = f'Fit specifics for ensemble {ens}')

            saveto = f'{read_from}/TABLES/' if args.saveto==None else args.saveto
            with open(f'{saveto}/fit2pt_parameter_table_{ens}.tex','w') as f:
                f.write(lattab)

    print(tab)



if __name__ == "__main__":
    main()


    





    