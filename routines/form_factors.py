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
prs.add_argument('-m','--meson'   , type=str, default=None)
prs.add_argument('-e','--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('--readfrom'  , type=str, default=None)



RATIOLIST = {
    'Dst': ['RA1','ZRA1','XFSTPAR','R0','R1','XV']
}



def main():
    args = prs.parse_args()

    meson = args.meson

    readfrom = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom

    # Explore list of ensemble 
    ens_list = [] if args.ensemble is None else args.ensemble
    for path, folders, files in os.walk(readfrom):
        for filename in files:
            if filename.startswith('fit3pt_config_'):
                ens = filename.split('_')[2]
                if os.path.isfile(f'{readfrom}/fit3pt_config_{ens}_global_fit_p.pickle'):
                    ens_list.append(ens)
    ens_list = np.unique(ens_list)

    # Loop over ensembles
    Rs = {}
    for ens in ens_list:
        # Read fit data
        tag = f'fit3pt_config_{ens}_global'
        fit,pars = read_config_fit(tag,path=readfrom)
        
        # Gather ratio values
        f0 = {}
        for k in pars:
            if k.endswith('f0'):
                rat,mom,_ = k.split('_')
                f0[mom,rat] = pars[k]

        momlist = sorted(set([m[0] for m in f0]))
        
        for mom in momlist:
            Rs[ens,mom] = {}
            for ratio in RATIOLIST[args.meson]:
                if (mom,ratio) in f0:
                    Rs[ens,mom][ratio] = f0[mom,ratio]

    # Create and format dataframe
    df = pd.DataFrame(Rs).transpose()
    if args.meson=='Dst':
        df['w'  ] = (1+df['XFSTPAR']**2)/(1-df['XFSTPAR']**2)
        for ens in ens_list:
            df.loc[(ens,'000'),'w']   =  gv.gvar('1.0(0.00001)')
            df.loc[(ens,'000'),'RA1'] =  df.loc[(ens,'000'),'ZRA1']
        df = df.drop('ZRA1',axis=1)
        df['RA1'] = df['RA1']**0.5

    else:
        df['w'  ] = (1+df['XF']**2)/(1-df['XF']**2)


    breakpoint()







if __name__=='__main__':
    main()