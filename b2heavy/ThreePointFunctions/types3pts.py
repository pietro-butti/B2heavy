import os
import h5py
import gvar as gv
import numpy as np
import pandas as pd

from .. import FnalHISQMetadata
from ..TwoPointFunctions.utils import jkCorr



def read_ratio_from_data(DATA,corrname):
    try:
        return DATA[corrname]
    except:
        raise Exception('WARNING: ',corrname,' not found')

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
            path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File2'])
            if os.path.exists(path):
                self.RatioFile = path
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
            case 'ZRA1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
                nFacs  = [[1., 1., 1.], [1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
            case 'ZRA1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_']] # 'R'
                nFacs  = [[1., 1., 1.], [1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
            case 'RA1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
                nFacs  = [[1., 1.], [1., 1., 1.]]
                dNames = [['V1_V4_V1_'],['P5_V4_P5']] # 'V1_V4_V2_', 'V1_V4_V3_']
            case 'R0':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A4_V1_']]
                nFacs  = [[1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'R0S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A4_V1_']]
                nFacs  = [[1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'R1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A1_V1_']]
                nFacs  = [[1.0]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'R1S':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_A1_V1_']]
                nFacs  = [[0.5]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'XV':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_V3_V2_', 'P5_V2_V3_']]
                nFacs  = [[1., -1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'XVS':
                hStr   = bStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['P5_V3_V2_', 'P5_V3_V2_']]
                nFacs  = [[1., -1.]]
                dNames = ['P5_A2_V2_', 'P5_A3_V3_']
            case 'XFSTPAR':
                hStr   = cStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['V1_V1_V1_']]
                nFacs  = [[1.]]
                dNames = ['V1_V4_V1_']
            case 'XFSTBOT':
                hStr   = cStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
                nFacs  = [[1., 1.]]
                dNames = ['V1_V4_V2_', 'V1_V4_V3_']
            case 'XFSSTPAR':
                hStr   = cStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['V1_V1_V1_']]
                nFacs  = [[1.]]
                dNames = ['V1_V4_V1_']
            case 'XFSSTBOT':
                hStr   = cStr
                lStr   = cStr
                qStr   = sStr
                nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
                nFacs  = [[1., 1.]]
                dNames = ['V1_V4_V3_']#, 'V1_V4_V3_']
            
        return { 
            'hStr'   : hStr  ,
            'lStr'   : lStr  ,
            'qStr'   : qStr  ,
            'nNames' : nNames,
            'nFacs'  : nFacs ,
            'dNames' : dNames
        }

    def RatioFileString(self,name,ts,h,hss,q,l):
        return f'{name}T{ts}{h}_RW_{hss}_rot_rot{q}{l}_p{self.info.momentum}'

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

    def ReadRatio(self, sms = ['RW','1S'], jkBin=None):
        DATA  = h5py.File(self.RatioFile,'r')['data']
        DATA2 = h5py.File(self.RatioFile,'r')['data']

        specs = self.RatioSpecs()
        T = self.mData['hSinks'][0]

        if ('RA1' not in self.info.ratio) and (self.info.momentum!='000'):
            FLAG_RA1 = True
        elif 'RA1' in self.info.ratio:
            FLAG_RA1 = False
        else:
            return 


        PROCESSED = {}
        for hss in sms:
            if FLAG_RA1: # ==============================================================================================

                AUX = []
                for tSink in self.mData['hSinks']:
                    nuCorr = []
                    for j,(name,nFac) in enumerate(zip(specs['nNames'][0],specs['nFacs'][0])):
                        corrname = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                        nuCorr.append(
                            read_ratio_from_data(DATA,corrname)[:]*nFac
                        )
                        # print(f'### nu: {corrname}')  
                        # print(read_ratio_from_data(DATA,corrname)[:]*nFac)
                    nuCorr = np.mean(nuCorr,axis=0)
                    nuCorr = jkCorr(nuCorr,bsize=0 if jkBin is None else jkBin)

                    duCorr = []
                    if len(specs['dNames'])>0:
                        for j,name in enumerate(specs['dNames']):
                            corrname = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                            duCorr.append(
                                read_ratio_from_data(DATA,corrname)
                            )
                    duCorr = np.sum(duCorr,axis=0)
                    duCorr = jkCorr(duCorr,bsize=0 if jkBin is None else jkBin)

                    AUX.append(
                        nuCorr/duCorr if list(duCorr) else nuCorr
                    )

                PROCESSED[hss] = 0.5*AUX[0][:,0:T+1] + 0.25*AUX[1][:,0:T+1] + 0.25*np.roll(AUX[1], -1, axis=1)[:,0:T+1]

            if not FLAG_RA1: # =====================================================================================================
                AUX = []
                for tSink in self.mData['hSinks']:
                    nuCorr = []
                    aux = []
                    for name in specs['nNames'][0]:
                        corrname = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])
                        aux.append(
                            jkCorr(
                                read_ratio_from_data(DATA,corrname)[:],
                                bsize=0 if jkBin is None else jkBin
                            )
                        )
                    nuCorr.append(np.array(aux)[:,:,0:(tSink+1)])


                    target = specs['nNames'][0] if self.info.ratio!='ZRA1' else specs['nNames'][1]
                    aux = []
                    for j,name in enumerate(target):
                        if self.info.ratio=='ZRA1':
                            corrname = self.RatioFileString(name,tSink,specs["lStr"],hss,specs["qStr"],specs["hStr"])
                        else:
                            corrname = self.RatioFileString(name,tSink,specs["hStr"],hss,specs["qStr"],specs["lStr"])

                        aux.append(
                            jkCorr(
                                read_ratio_from_data(DATA,corrname)[:],
                                bsize=0 if jkBin is None else jkBin
                            )
                        )
                    aux = np.flip(np.array(aux)[:,:,0:(tSink+1)], axis=2) if self.info.ratio!='ZRA1' else np.array(aux)[:,:,0:(tSink+1)]
                    nuCorr.append(aux)

                    duCorr = []
                    for mesStr, cName in zip([specs['lStr'],specs['hStr']],specs['dNames']):
                        aux = []
                        for name in cName:
                            corrname = self.RatioFileString(name,tSink,mesStr,hss,specs["qStr"],mesStr)
                            aux.append(
                                read_ratio_from_data(DATA,corrname)
                            )
                        aux = np.array(aux).mean(axis=0)[:,0:(tSink+1)]
                        duCorr.append(
                            jkCorr(aux,bsize=0 if jkBin is None else jkBin)
                        )
                    
                    AUX.append(
                        (nuCorr[0]*nuCorr[1]).sum(axis=0)/(duCorr[0]*duCorr[1])
                    )

                E0 = 0. # FIXME
                m0 = 0. # FIXME
                PROCESSED[hss] = 0.5*AUX[0][:,0:T+1]*np.exp((E0 - m0)*T) + 0.25*AUX[1][:,0:T+1]*np.exp((E0 - m0)*(T+1)) + 0.25*np.roll(AUX[1], -1, axis=1)[:,0:T+1]*np.exp((E0 - m0)*(T+1))

        return PROCESSED

def test():
    r0 = RatioIO(
        'Coarse-1',
        'RA1',
        '000',
        PathToDataDir='/Users/pietro/code/data_analysis/BtoD/Alex'
    )
    print(r0.RatioFile)

    p = r0.ReadRatio(jkBin=1)
    
    
    
    return p


