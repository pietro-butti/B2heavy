import sys
import itertools
import argparse
import numpy  as np
import gvar   as gv
import pandas as pd

from matplotlib  import pyplot as plt
from tqdm        import tqdm

from b2heavy.FnalHISQMetadata            import params as mData
from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics

from b2heavy.ThreePointFunctions.fitter3pts import RatioFitter, RatioIO

from routines.fit_2pts_utils             import load_toml, read_config_fit
from routines.fit_3pts_config            import elaborate_ratio, exists_2pts_analysis




def ra1():
    tmin = 0.4

    binsize  = {
        'MediumCoarse':13,
        'Coarse-2':    16,
        'Coarse-1':    11,
        'Coarse-Phys': 19,
        'Fine-1':      16,
        'Fine-Phys':   16,
        'SuperFine':   22
    }
    readfrom = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/presentation' 
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smslist  = ['1S']


    enss = ['Coarse-2','Coarse-1','Coarse-Phys','Fine-1']
    moms = ['000','100','200','300','400']
    ratio = 'RA1'

    itr = itertools.product(enss,moms)
    aux = []
    for ens,mom in itr:
        print('--------------',ens,mom,'--------------')
        try:
            fitres_mes = read_config_fit(
                tag = f'fit2pt_config_{ens}_Dst_{mom}',
                path=readfrom,
                jk=True
            )
            fitres_0  = read_config_fit(
                tag = f'fit2pt_config_{ens}_Dst_000',
                path=readfrom,
                jk=True
            )
        except FileNotFoundError:
            continue        

        e0 = fitres_mes['E'][:,0]
        m0 = fitres_0['E'][:,0]

        io = RatioIO(ens, 'RA1', mom, PathToDataDir=data_dir)
        robj = RatioFitter(
            io, 
            jkBin    = binsize[ens], 
            smearing = smslist,
            E0       = e0,
            m0       = m0,
            Zpar     = None,
            Zbot     = None if mom=='000' else np.exp(fitres_mes['Z_1S_Bot'][:,0]) * np.sqrt(2. * e0),
            Z0       = np.exp(fitres_0['Z_1S_Unpol'][:,0]) * np.sqrt(2. * m0),
            wrecoil  = e0/m0
        )

        Tmin = int(tmin/mData(ens)['aSpc'].mean)

        trange = (Tmin,robj.Tb-Tmin)

        xdata,ydata,yjk = robj.format(trange,flatten=True,alljk=True)
        # svd = round(correlation_diagnostics(yjk,verbose=False),ndigits=5)
        svd = correlation_diagnostics(yjk,verbose=False)

        aux.append({
            'ensemble': ens,
            'ratio'   : ratio,
            'momentum': mom,
            'tmin'    : Tmin,
            'svd'     : svd,
        })
    
    df = pd.DataFrame(aux).set_index(['ensemble','ratio','momentum'])
    print(df)





def main():
    tmin = 0.4


    binsize  = {
        'MediumCoarse':13,
        'Coarse-2':    16,
        'Coarse-1':    11,
        'Coarse-Phys': 19,
        'Fine-1':      16,
        'Fine-Phys':   16,
        'SuperFine':   22
    }
    readfrom = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/presentation' 
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smslist  = ['1S']


    enss = ['Coarse-2','Coarse-1','Coarse-Phys','Fine-1']
    moms = ['000','100','200','300','400']
    rats = ['xfstpar','XV','R0','R1']

    itr = itertools.product(enss,rats,moms)
    aux = []
    for ens,ratio,mom in tqdm(itr):
        try:
            if exists_2pts_analysis(readfrom,ens,mom,jkfit=True):
                robj = elaborate_ratio(ens,ratio,mom, data_dir, binsize[ens], smslist, readfrom=readfrom, jk=True)        
        except:
            # print(ens,ratio,mom)
            continue

        Tmin = int(tmin/mData(ens)['aSpc'].mean)

        trange = (Tmin,robj.Tb-Tmin)

        xdata,ydata,yjk = robj.format(trange,flatten=True,alljk=True)
        svd = round(correlation_diagnostics(yjk,verbose=False),ndigits=4)

        aux.append({
            'ensemble': ens,
            'ratio'   : ratio,
            'momentum': mom,
            'tmin'    : Tmin,
            'svd'     : svd,
        })
    
    df = pd.DataFrame(aux).set_index(['ensemble','ratio','momentum'])
    print(df)






if __name__=='__main__':
    ra1()