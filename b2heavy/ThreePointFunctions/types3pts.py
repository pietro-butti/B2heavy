import os
import h5py
import gvar   as gv
import numpy  as np
import xarray as xr

import matplotlib.pyplot as plt

from .. import FnalHISQMetadata
from ..TwoPointFunctions.utils import jkCorr, compute_covariance


def read_ratio_from_files_list(corrname,*files,verbose=False):
    for file in files:
        with h5py.File(file,'r') as f:
            try:
                d = f['data'][corrname][:]
                if verbose:
                    print(f'{corrname} found in {file}')
                break
            except KeyError:
                continue

    return d



class RatioInfo:
    def __init__(self, _name:str, _ens:str, _rat:str, _mom:str):
        self.name     = _name
        self.ensemble = _ens
        self.ratio    = _rat.upper()
        self.momentum = _mom
        self.tSink    = None
        self.binsize  = None
        self.filename = None

    def __str__(self):
        return f' # ------------- {self.name} -------------\n # ensemble = {self.ensemble}\n #    meson = {self.meson}\n # momentum = {self.momentum}\n #  binsize = {self.binsize}\n # filename = {self.filename}\n # ---------------------------------------'

class RatioIO:
    def __init__(self, _ens:str, _rat:str, _mom:str, PathToFile=None, PathToDataDir=None, name=None):
        dname = f'{_ens}_{_rat}_p{_mom}' if name is None else name

        self.info   = RatioInfo(dname,_ens,_rat,_mom)
        self.pmom   = [int(px) for px in self.info.momentum]
        self.mData  = FnalHISQMetadata.params(_ens)

        if PathToFile is not None:
            self.RatioFile = PathToFile
        elif PathToDataDir is not None:
            path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File'])
            if os.path.exists(path):
                self.RatioFile = path
            else:
                raise FileNotFoundError(f'The file {path} has not been found')
            
            path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File2'])
            if os.path.exists(path):
                self.RatioFile2 = path
            else:
                raise FileNotFoundError(f'The file {path} has not been found')
            
        else:
            raise FileNotFoundError(f'Please specify as optional argument either data file in `PathToFile` or `PathToDataDir` to look at default data file.')

    def RatioSpecs(self):
        bStr    = '_k' + self.mData['kBStr']
        cStr    = '_k' + self.mData['kDStr']
        sStr    = '_m' + self.mData['msStr']
        mStr    = '_m' + self.mData['mlStr']

        match self.info.ratio:
            case 'RA1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A2_V2_'], ['P5_A2_V2_']] # 'R'
                nFacs  = [[1., 1.], [ 1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']         
                
                self.specs  = dict(
                    source = ['B','Dst'],
                    sink   = ['B','Dst']
                )

            case 'ZRA1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
                nFacs  = [[1., 1., 1.], [1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
                
                self.specs  = {'num': None, 'den': None}

            case 'ZRA1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_']] # 'R'
                nFacs  = [[1., 1., 1.], [1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
                
                self.specs  = {'num': None, 'den': None}

            case 'RA1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
                nFacs  = [[1., 1.], [1., 1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5']] # 'V1_V4_V2_', 'V1_V4_V3_']
                
                self.specs  = dict(
                    source = ['Bs','Dsst'],
                    sink   = ['Bs','Dsst']
                )

            case 'R0':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A4_V1_']]
                nFacs  = [[1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'B',
                    sink   = 'Dst'
                )

            case 'R0S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A4_V1_']]
                nFacs  = [[1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'Bs',
                    sink   = 'Dsst'
                )

            case 'R1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_']]
                nFacs  = [[1.0]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'B',
                    sink   = 'Dst'
                )
            
            case 'R1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A1_V1_']]
                nFacs  = [[0.5]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'Bs',
                    sink   = 'Dsst'
                )

            case 'XV':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_V3_V2_', 'P5_V2_V3_']]
                nFacs  = [[1., -1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'B',
                    sink   = 'Dst'
                )

            case 'XVS':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_V3_V2_', 'P5_V3_V2_']]
                nFacs  = [[1., -1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
                
                self.specs  = dict(
                    source = 'Bs',
                    sink   = 'Dsst'
                )

            case 'XFSTPAR':
                hStr   = cStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['V1_V1_V1_']]
                nFacs  = [[1.]]
                dNames = ['V1_V4_V1_']
                
                self.specs  = dict(
                    source = 'Dst',
                    sink   = 'Dst'
                )

            case 'XFSTBOT':
                hStr   = cStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
                nFacs  = [[1., 1.]]
                dNames = ['V1_V4_V2_', 'V1_V4_V3_']
                
                self.specs  = dict(
                    source = 'Dst',
                    sink   = 'Dst'
                )

            case 'XFSSTPAR':
                hStr   = cStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['V1_V1_V1_']]
                nFacs  = [[1.]]
                dNames = ['V1_V4_V1_']
                
                self.specs  = dict(
                    source = 'Dsst',
                    sink   = 'Dsst'
                )

            case 'XFSSTBOT':
                hStr   = cStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
                nFacs  = [[1., 1.]]
                dNames = ['V1_V4_V3_']#, 'V1_V4_V3_']
                
                self.specs  = dict(
                    source = 'Dsst',
                    sink   = 'Dsst'
                )


        return { 
            'hStr'   : hStr  ,
            'lStr'   : lStr  ,
            'qStr'   : qStr  ,
            'nNames' : nNames,
            'nFacs'  : nFacs ,
            'dNames' : dNames
        }

    def RatioFileString(self,name,ts,h,hss,q,l,mom=None):
        p = self.info.momentum if mom is None else mom
        return f'{name}T{ts}{h}_RW_{hss}_rot_rot{q}{l}_p{p}'

    def RatioFileDict(self,sms = ['RW','1S']):
        specs = self.RatioSpecs()

        aux = {}
        for hss in sms:
            aux[hss] = {}
            for tSink in self.mData['hSinks']:
                nuCorr = []
                duCorr = []

                flag = False
                for j,(name,nFac) in enumerate(zip(specs['nNames'][0],specs['nFacs'][0])):
                        # filename = f'{name}T{tSink}{specs["hStr"]}_RW_{hss}_rot_rot{specs["qStr"]}{specs["lStr"]}_p{self.info.momentum}'
                        filename = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                        nuCorr.append(filename)

                if ('RA1' not in self.info.ratio) and (self.info.momentum!='000'):
                    flag = True
                    if len(specs['dNames'])>0:
                        for j,name in enumerate(specs['dNames']):
                            filename = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                            duCorr.append(filename)

                elif 'RA1' in self.info.ratio:
                    flag = True
                    target = specs['nNames'][0] if self.info.ratio!='ZRA1' else specs['nNames'][1]
                    for j,name in enumerate(target):
                        if self.info.ratio=='ZRA1':
                            filename = self.RatioFileString(name,tSink,specs["lStr"],hss,specs["qStr"],specs["hStr"])
                        else:
                            filename = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                        nuCorr.append(filename)

                if flag:
                    aux[hss][tSink] = dict(nu=nuCorr,du=duCorr)

        return aux

    def ReadRatio(self, sms = ['RW','1S'], jkBin=None, E0=None, m0=None, verbose=False, datafiles=None):
        if datafiles is None:
            # files = [self.RatioFile,self.RatioFile2]
            files = [self.RatioFile2,self.RatioFile]
        else:
            files = datafiles

        specs = self.RatioSpecs()
        T = self.mData['hSinks'][0]

        if ('RA1' not in self.info.ratio) and (self.info.momentum!='000'):
            not_RA1 = True
        elif 'RA1' in self.info.ratio:
            not_RA1 = False
        else:
            raise Exception(f'{self.info.ratio} is not a valid ratio name') 


        PROCESSED = {}
        for hss in sms:
            if not_RA1: # ==============================================================================================
                AUX = []
                for tSink in self.mData['hSinks']:
                    nuCorr = []
                    for j,(name,nFac) in enumerate(zip(specs['nNames'][0],specs['nFacs'][0])):
                        corrname = self.RatioFileString(
                            name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"]
                        )
                        nuCorr.append(
                            read_ratio_from_files_list(corrname,*files,verbose=verbose) * nFac
                        )

                    nuCorr = np.mean(nuCorr,axis=0)
                    nuCorr = jkCorr(nuCorr,bsize=0 if jkBin is None else jkBin)


                    duCorr = []
                    if len(specs['dNames'])>0:
                        for j,name in enumerate(specs['dNames']):
                            corrname = self.RatioFileString(
                                name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"]
                            )
                            duCorr.append(
                                read_ratio_from_files_list(corrname,*files,verbose=verbose)
                            )

                    duCorr = np.sum(duCorr,axis=0)
                    duCorr = jkCorr(duCorr,bsize=0 if jkBin is None else jkBin)
                    
                    AUX.append(
                        nuCorr/duCorr if list(duCorr) else nuCorr
                    )

                PROCESSED[hss] = \
                    0.5*AUX[0][:,0:T+1] + \
                    0.25*AUX[1][:,0:T+1] + \
                    0.25*np.roll(AUX[1], -1, axis=1)[:,0:T+1] 
                
            if not not_RA1: # ==========================================================================================
                AUX = []
                for tSink in self.mData['hSinks']:
                    aux    = []
                    nuCorr = []
                    for name in specs['nNames'][0]:
                        corrname = self.RatioFileString(
                            name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"]
                        )
                        aux.append(
                            jkCorr(
                                # read_ratio_from_files_list(corrname,*reversed(files),verbose=verbose),
                                # FIXME
                                read_ratio_from_files_list(corrname,*files,verbose=verbose),
                                bsize=0 if jkBin is None else jkBin
                            )
                        )
                    nuCorr.append(np.array(aux)[:,:,0:(tSink+1)])

                    target = specs['nNames'][0] if self.info.ratio!='ZRA1' else specs['nNames'][1]
                    aux = []
                    for j,name in enumerate(target):
                        if self.info.ratio=='ZRA1':
                            corrname = self.RatioFileString(
                                name,tSink,specs["lStr"],hss,specs["qStr"],specs["hStr"]
                            )
                        else:
                            corrname = self.RatioFileString(
                                name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"]
                            )

                        aux.append(
                            jkCorr(
                                read_ratio_from_files_list(corrname,*files,verbose=verbose),
                                bsize=0 if jkBin is None else jkBin
                            )
                        )
                    aux = np.flip(np.array(aux)[:,:,0:(tSink+1)], axis=2) if self.info.ratio!='ZRA1' else np.array(aux)[:,:,0:(tSink+1)]
                    nuCorr.append(aux)


                    duCorr = []
                    for mesStr, cName in zip([specs['lStr'],specs['hStr']],specs['dNames']):
                        aux = []
                        for name in cName:
                            corrname = self.RatioFileString(
                                name,tSink,mesStr,hss,specs["qStr"],mesStr,mom='000'
                            )
                            aux.append(
                                # read_ratio_from_files_list(corrname,*reversed(files),verbose=verbose)
                                # FIXME
                                read_ratio_from_files_list(corrname,*files,verbose=verbose)
                            )
                        aux = np.array(aux).mean(axis=0)[:,0:(tSink+1)]
                        duCorr.append(
                            jkCorr(aux,bsize=0 if jkBin is None else jkBin)
                        )

                    AUX.append(
                        (nuCorr[0]*nuCorr[1]).sum(axis=0)/(duCorr[0]*duCorr[1])
                    )

                if E0 is None and m0 is None:
                    raise Exception('Energy and rest mass must be provided')
                elif E0==0. and m0==0.:
                    Warning('E0 and m0 are set to 0')
                elif E0.shape!=m0.shape:
                    raise Exception(f'E0 ({E0.shape = }) and m0 ({m0.shape = }) must have compatible shapes ')
                elif E0.shape!=AUX[0].shape:
                    raise Exception(f'E0 ({E0.shape = }) and ratio data ({AUX[0].shape = }) must have compatible shapes ')

                PROCESSED[hss] = \
                    0.5*AUX[0][:,0:T+1]*np.exp((E0 - m0)*T) + \
                    0.25*AUX[1][:,0:T+1]*np.exp((E0 - m0)*(T+1)) + \
                    0.25*np.roll(AUX[1], -1, axis=1)[:,0:T+1]*np.exp((E0 - m0)*(T+1))

        return PROCESSED

class Ratio:
    def __init__(self, io:RatioIO, jkBin=None, E0=None, m0=None, smearing=None, **kwargs):
        self.io           = io
        self.info         = io.info
        self.info.binsize = jkBin
        self.binsize      = jkBin

        self.E0           = E0
        self.m0           = m0

        self.data = io.ReadRatio(
            jkBin= jkBin,
            E0   = E0,
            m0   = m0,
            sms  = ['1S','RW'] if smearing is None else smearing,
            **kwargs
        )

        self.specs = io.specs

        self.smr = sorted(list(self.data.keys()))

        shapes = [self.data[d].shape for d in self.data]
        shapes = np.unique(shapes)
        assert len(shapes)==2

        self.timeslice = np.arange(shapes[0])

    def format(self, trange=None, smearing=None, flatten=False, alljk=False, **cov_kwargs):
        # Select smearings
        if smearing is None:
            smr = self.smr
        else:
            if set(smearing)<=set(self.smr):
                smr = smearing
            else:
                raise KeyError(f'Smearing list {smearing=} contain at least one item which is not contained in {self.smr=}')

        # Select data in the timeranges
        it = self.timeslice if trange is None else np.arange(min(trange),max(trange)+1)
        xdata = self.timeslice[it]

        # Compute un-jk data
        if cov_kwargs.get('scale'):
            ydata_full = self.io.ReadRatio(jkBin=1,E0=self.E0,m0=self.m0)
            ally = {s: ydata_full[s][:,it] for s in smr}
        else:
            ally = None

        sliced = {s: self.data[s][:,it] for s in smr}
        ydata = compute_covariance(
            sliced, ally=ally, **cov_kwargs
        )
        if flatten:
            ydata = np.concatenate([ydata[s] for s in smr])

        if alljk and not flatten:
            yjk = {s: self.data[s][:,it] for s in smr}
        elif alljk and flatten:
            yjk = np.hstack([self.data[s][:,it] for s in smr])

        return (xdata,ydata) if not alljk else (xdata,ydata,yjk)





def main_scan_all_ratio():
    # mom_list   = ['000','100','200','300','110','211']
    mom_list   = ['000','100','200','300']#,'400']
    ratio_list = ['RA1','ZRA1','ZRA1S','RA1S']#,'R0','R0S','R1','R1S','XV','XVS','XFSTPAR','XFSTBOT','XFSSTPAR','XFSSTBOT']

    f,ax = plt.subplots(
        len(mom_list),len(ratio_list),
        layout="constrained",
        sharex=True,
        figsize=(20,10)
    )

    for i,ratio in enumerate(ratio_list):
        for j,mom in enumerate(mom_list):
            r = RatioIO(
                'Coarse-1',
                ratio,
                mom,
                PathToDataDir='/Users/pietro/code/data_analysis/BtoD/Alex'
            )

            try:
                d = r.ReadRatio(jkBin=11,E0=0,m0=0)
                print(r.specs)
            except Exception:
                d = None

            if d is not None:
                ydata = gv.gvar(
                    d['RW'].mean(axis=0),
                    np.cov(d['RW'],rowvar=False)*(d['RW'].shape[0]-1)
                )
                y  = gv.mean(ydata)
                ye = gv.sdev(ydata)
                x = np.arange(len(y)) 

                ax[j,i].errorbar(x,y,yerr=ye,fmt='.')
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])
            else:
                ax[j,i].scatter([],[])
                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

            if j==0:
                ax[0,i].set_title(ratio)
            if i==0:
                ax[j,0].set_ylabel(mom)

            print(mom,ratio)


    plt.tight_layout()
    # plt.savefig('/Users/pietro/Desktop/MediumCoarse.pdf')
    plt.show()

def main():
    ens = 'Coarse-1'
    rat = 'XFSTBOT'
    mom = '100'
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex'

    for mom in ['100','200','300']:
        io    = RatioIO(ens,rat,mom,PathToDataDir=frm)
        d = io.ReadRatio(jkBin=11,E0=0,m0=0)

        ydata = gv.gvar(
            d['RW'].mean(axis=0),
            np.cov(d['RW'],rowvar=False)*(d['RW'].shape[0]-1)
        )
        y  = gv.mean(ydata)
        ye = gv.sdev(ydata)
        x = np.arange(len(y)) 

        plt.errorbar(x,y,yerr=ye,fmt='.')

    plt.grid()
    plt.show()

