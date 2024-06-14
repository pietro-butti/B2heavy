import os
import h5py
import gvar   as gv
import numpy  as np
import xarray as xr
import pandas as pd

import jax 
jax.config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt

from ..FnalHISQMetadata import params

from ..TwoPointFunctions.utils   import jkCorr, compute_covariance

from .utils     import read_config_fit, exists, exists_analysis
from .types3pts import RatioFileString


BINSIZE  = {
    'MediumCoarse':13,
    'Coarse-2':    16,
    'Coarse-1':    11,
    'Coarse-Phys': 19,
    'Fine-1':      16,
    'Fine-Phys':   16,
    'SuperFine':   22
}


def corr3tag(src,mes1,cur,snk,mes2,tt,mom,smr,pol):
    return f'[{mes1}]{src}->{cur}->[{mes2}]{snk}.{tt}.{mom}.{smr}.{pol}'


class Correlator3:
    def __init__(self, ensemble, PathToFile=None, PathToDataDir=None, jkbin=None):
        # initialize metadata
        self.mData       = params(ensemble)
        self.Ta, self.Tb = self.mData['hSinks']
        self.jkbin       = BINSIZE[ensemble] if jkbin is None else jkbin
        self.jkCorr      = lambda x: jkCorr(x,bsize=self.jkbin)


        # initialize input archive
        if PathToFile is not None:
            self.file = PathToFile
        elif PathToDataDir is not None:
            path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File2'])
            if os.path.exists(path):
                self.file = path
            else:
                raise FileNotFoundError(f'The file {path} has not been found')
        else:
            raise FileNotFoundError(f'Please specify as optional argument either data file in `PathToFile` or `PathToDataDir` to look at default data file.')
        

        # initialize data archive
        self.arx = h5py.File(self.file)['data']

        self.data = {}
        self.data_full = {}

    def read(self,_src,_cur,_snk,smr,mom,heavy='B',light='Dst',strange=False,fulldata=True):
        hh = '_k' + self.mData['kBStr' if 'B' in heavy else 'kDStr']
        ll = '_k' + self.mData['kDStr' if 'D' in light else 'kBStr']
        qq = '_m' + self.mData['mlStr' if not strange else 'msStr']

        src = _src.upper()
        cur = _cur.upper()
        snk = _snk.upper()

        collinear = cur[-1]==snk[-1]
        par       = collinear and cur[-1]=='1'

        pol = 'Par' if par else 'Bot'
        sm  = 'd' if smr=='RW' else smr
        
        name = f'{src}_{cur}_{snk}_'
        corr1 = RatioFileString(name,self.Ta,hh,smr,qq,ll,mom)
        corr2 = RatioFileString(name,self.Tb,hh,smr,qq,ll,mom)

        if corr1 in self.arx:
            data1 = self.arx[corr1][:]
            data2 = self.arx[corr2][:]
        else:
            raise KeyError(f'{corr1} not found in data archive')

        tag1 = corr3tag(src,heavy,cur,snk,light,self.Ta,mom,sm,pol)  
        self.data[tag1] = self.jkCorr(data1)
        self.data_full[tag1] = jkCorr(data1,bsize=1)

        tag2 = corr3tag(src,heavy,cur,snk,light,self.Tb,mom,sm,pol)  
        self.data[tag2] = self.jkCorr(data2)
        self.data_full[tag2] = jkCorr(data2,bsize=1)

        return



def main():
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    c3  = Correlator3('Coarse-1',PathToDataDir=frm)

    c3.read('p5','a2','v2','1S','300')


if __name__=='__main__':
    main()