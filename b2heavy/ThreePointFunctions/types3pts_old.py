import os
import h5py
import gvar   as gv
import numpy  as np
import xarray as xr

import matplotlib.pyplot as plt

from .. import FnalHISQMetadata
from ..TwoPointFunctions.utils   import jkCorr, compute_covariance

from .utils import read_config_fit, exists, exists_analysis



def read_ratio_from_files_list(corrname,*files,verbose=False):
    for file in files:
        with h5py.File(file,'r') as f:
            try:
                d = f['data'][corrname][:]
                if verbose:
                    print(f'{corrname} found in {file}')
                break
            except KeyError:
                raise Exception(f'{corrname} has not been found in the following list: {files}')

    return d


def ratio_prerequisites(
    ens  ,   
    ratio, 
    mom  , 
    smearing = ['1S','d'],
    readfrom = None, 
    jk       = False,
    meson    = 'Dst'
):

    req = dict(
        E0      = None,
        m0      = None,
        Z0      = None,
        Zpar    = None,
        Zbot    = None,
        wrecoil = None
    )

    if ratio in ['R0','R1'] and mom!='000':
        if exists_analysis(readfrom,ens,meson,mom,type='2',jkfit=jk):
            tag = f'fit2pt_config_{ens}_{meson}_{mom}'
            p = read_config_fit(tag, path=readfrom, jk=jk)

            req['Zpar'] = {sm: None for sm in smearing}
            req['Zbot'] = {sm: None for sm in smearing}

            for sm in smearing:
                if not jk:
                    req['Zpar'][sm] = (np.exp(p[-1][f'Z_{sm}_Par'][0]) * np.sqrt(2*p[-1]['E'][0])).mean
                    req['Zbot'][sm] = (np.exp(p[-1][f'Z_{sm}_Bot'][0]) * np.sqrt(2*p[-1]['E'][0])).mean

                else:
                    req['Zpar'][sm] = np.exp(p[f'Z_{sm}_Par'][:,0]) * np.sqrt(2*p['E'][:,0])
                    req['Zbot'][sm] = np.exp(p[f'Z_{sm}_Bot'][:,0]) * np.sqrt(2*p['E'][:,0])


    elif 'RA1' in ratio:
        if exists_analysis(readfrom,ens,meson,mom,type='2',jkfit=jk) \
            and exists_analysis(readfrom,ens,meson,'000',type='2',jkfit=jk):
                tag = f'fit2pt_config_{ens}_{meson}_{mom}'
                p = read_config_fit(tag, path=readfrom, jk=jk)
                req['E0'] = p['E'][:,0] if jk else p[-1]['E'][0].mean

                tag = f'fit2pt_config_{ens}_{meson}_000'
                p0 = read_config_fit(tag, path=readfrom, jk=jk)
                req['m0'] = p0['E'][:,0] if jk else p0[-1]['E'][0].mean

        req['Zbot'] = {sm: None for sm in smearing}
        req['Z0']   = {sm: None for sm in smearing}

        for sm in smearing:
            pol = 'Bot' if mom!='000' else 'Unpol'
            if not jk:
                req['Zbot'][sm] = np.exp(p [-1][f'Z_{sm}_{pol}'][0].mean) * np.sqrt(2*p[-1]['E'][0]).mean
                req['Z0']  [sm] = np.exp(p0[-1][f'Z_{sm}_Unpol'][0].mean) * np.sqrt(2*p[-1]['E'][0]).mean
            else:
                req['Zbot'][sm] = np.exp(p [f'Z_{sm}_{pol}'][:,0]) * np.sqrt(2*p['E'][:,0])
                req['Z0']  [sm] = np.exp(p0[f'Z_{sm}_Unpol'][:,0]) * np.sqrt(2*p['E'][:,0])

        if mom!='000':
            if exists_analysis(readfrom,ens,'xfstpar',mom,type='3',jkfit=jk):
                tag = f'fit3pt_config_{ens}_xfstpar_{mom}'
                p = read_config_fit(tag, path=readfrom, jk=jk)
                xf = p['ratio'].reshape(p['ratio'].shape[-1]) if jk else p[-1]['ratio'][0].mean
                req['wrecoil'] = (1+xf**2)/(1-xf**2)
        else:
            req['E0'] = None
            req['m0'] = None
            req['wrecoil'] = None

    return req


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
        return f' # ------------- {self.name} -------------\n # ensemble = {self.ensemble}\n #    ratio = {self.ratio}\n # momentum = {self.momentum}\n #  binsize = {self.binsize}\n # filename = {self.filename}\n # ---------------------------------------'

class RatioIO:
    def __init__(self, _ens:str, _rat:str, _mom:str, PathToFile=None, PathToDataDir=None, name=None):
        dname = f'{_ens}_{_rat}_p{_mom}' if name is None else name

        self.info   = RatioInfo(dname,_ens,_rat,_mom)
        self.pmom   = [int(px) for px in self.info.momentum]
        self.mData  = FnalHISQMetadata.params(_ens)

        if PathToFile is not None:
            self.RatioFile2 = PathToFile
        elif PathToDataDir is not None:
            # path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File'])
            # if os.path.exists(path):
            #     self.RatioFile = path
            # else:
            #     raise FileNotFoundError(f'The file {path} has not been found')
            
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
            # ========================== B ---> D ==========================
            case 'XF':
                hStr   = cStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_V1_P5_','P5_V2_P5_','P5_V3_P5_']]
                nFacs  = [[1., 1., 1.]]
                dNames = ['P5_V4_P5_']
                
                self.specs  = dict(
                    source = 'D',
                    sink   = 'D'
                )  
            
            case 'QPLUS':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr                
                nNames = [['P5_V4_P5_']]
                nFacs  = [[1.]]
                dNames = ['P5_V4_P5_']

                self.specs  = dict(
                    source = 'B',
                    sink   = 'D'
                )                  

            case 'RMINUS':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_V1_P5_','P5_V2_P5_','P5_V3_P5_']]
                nFacs  = [[1.,1.,1.]]
                dNames = ['P5_V4_P5_']

                self.specs = dict(
                    source = 'B',
                    sink   = 'D' 
                )

            case 'RPLUS':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_V4_P5_'],['P5_V4_P5_']]
                nFacs  = [[1.],[1.]]
                dNames = [['P5_V4_P5_'],['P5_V4_P5_']]                

                self.specs = dict(
                    source = ['B','D'],
                    sink   = ['B','D']
                )

            # ========================== B ---> D* ==========================
            case 'RA1':
                hStr   = bStr
                lStr   = cStr
                qStr   = mStr
                nNames = [['P5_A2_V2_'], ['P5_A3_V3_']] # 'R'
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
                
                self.specs  = dict(
                    source = ['B','Dst'],
                    sink   = ['B','Dst']
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


            # ========================== Bs ---> Ds* ==========================
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
            files = [self.RatioFile2]
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
                    if self.info.momentum!='000':
                        raise Exception('Energy and rest mass must be provided')
                elif isinstance(E0, np.ndarray) and isinstance(m0, np.ndarray): 
                    perjk = True
                    if E0.shape!=m0.shape:
                        raise Exception(f'E0 ({E0.shape = }) and m0 ({m0.shape = }) must have compatible shapes ')
                    elif len(E0)!=AUX[0].shape[0]:
                        raise Exception(f'E0 ({E0.shape = }) and ratio data ({AUX[0].shape = }) must have compatible shapes ')
                elif isinstance(E0, float) and isinstance(E0, float):
                    perjk = False
                    if E0==0. and m0==0.:
                        Warning('E0 and m0 are set to 0')

                if self.info.momentum!='000':
                    expT  = np.exp((E0 - m0)*T)
                    expTp = np.exp((E0 - m0)*(T+1))
                    if perjk:
                        expT  = np.tile(expT,(T+1,1)).T
                        expTp = np.tile(expTp,(T+1,1)).T

                    PROCESSED[hss] = \
                        0.5*AUX[0][:,:T+1] * expT + \
                        0.25*AUX[1][:,:T+1] * expTp + \
                        0.25*np.roll(AUX[1], -1, axis=1)[:,:T+1] * expTp 

                else:
                    PROCESSED[hss] = \
                        0.5*AUX[0][:,0:T+1] +\
                        0.25*AUX[1][:,0:T+1] +\
                        0.25*np.roll(AUX[1], -1, axis=1)[:,0:T+1]

        return PROCESSED

class Ratio:
    def __init__(self, io:RatioIO, jkBin=None, smearing=None, verbose=False, datafiles=None, **kwargs):
        self.io           = io
        self.info         = io.info
        self.info.binsize = jkBin
        self.binsize      = jkBin

        E0 = kwargs.get('E0')
        m0 = kwargs.get('m0')

        self.E0           = E0
        self.m0           = m0

        self.data = io.ReadRatio(
            jkBin     = jkBin,
            E0        = E0,
            m0        = m0,
            datafiles = datafiles,
            verbose   = verbose,
            sms       = ['1S','RW'] if smearing is None else smearing,
        )
        self.specs = io.specs

        factor = {}
        if self.info.ratio in ['R0','R1','RA1'] and self.info.momentum!='000':
            Zpar = kwargs.get('Zpar')
            Zbot = kwargs.get('Zbot')
            if self.info.ratio in ['R0','R1']:
                if Zpar is None or Zbot is None:
                    raise Exception('For R0 ratio, Z_1S_Par, Z_1S_Bot, Z_d_Par, Z_d_Bot must be provided')
                else:
                    for sm in smearing:
                        smi = 'd' if sm=='RW' else sm
                        factor[sm] = np.sqrt(Zbot[smi]/Zpar[smi])

            elif self.info.ratio=='RA1':
                Z0      = kwargs.get('Z0')
                wrecoil = kwargs.get('wrecoil')
                if E0       is None or \
                    m0      is None or \
                    Zbot    is None or \
                    Z0      is None or \
                    wrecoil is None:
                    raise Exception('For RA1 ratio, rest mass, E0, Z_d_Bot, Z_d_Bot (at 0 and non-0 momentum) and recoil parameter must be provided')
                
                else:
                    T = self.io.mData['hSinks'][0]
                    for sm in smearing:
                        smi = 'd' if sm=='RW' else sm
                        factor[sm] = 1./wrecoil**2 * \
                            Zbot[smi]/np.sqrt( Z0[smi] * Z0['1S'] ) * \
                            np.exp(-(E0-m0)*T)    
                    
            for sm in smearing:
                if isinstance(factor[sm], np.ndarray):
                    factor[sm] = np.tile(factor[sm],(self.data[sm].shape[-1],1)).T


                self.data[sm] = self.data[sm] * factor[sm]

        elif self.info.ratio in ['ZRA1','ZRA1S']:
            Z0   = kwargs.get('Z0')
            Zbot = kwargs.get('Zbot')

            T = self.io.mData['hSinks'][0]
            for sm in smearing:
                smi = 'd' if sm=='RW' else sm
                factor[sm] = \
                    Zbot[smi]/np.sqrt( Z0[smi] * Z0['1S'] ) 

                if isinstance(factor[sm], np.ndarray):
                    factor[sm] = np.tile(factor[sm],(self.data[sm].shape[-1],1)).T

                self.data[sm] = self.data[sm] * factor[sm]


        self.smr = sorted(list(self.data.keys()))

        shapes = list({self.data[sm].shape for sm in self.data})
        assert len(shapes)==1

        self.nbins, tmax = shapes[0]
        self.timeslice = np.arange(tmax)

        return


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
            _e0 = self.E0.mean() if isinstance(self.E0, np.ndarray) else self.E0
            _m0 = self.m0.mean() if isinstance(self.m0, np.ndarray) else self.m0

            ydata_full = self.io.ReadRatio(jkBin=1,E0=_e0,m0=_m0)
            ally = {s: ydata_full[s][:,it] for s in smr}
        else:
            ally = None

        ydata = compute_covariance(
            {s: self.data[s][:,it] for s in smr},
            ally=ally, 
            **cov_kwargs
        )
        if flatten:
            ydata = np.concatenate([ydata[s] for s in smr])

        if alljk and not flatten:
            yjk = {s: self.data[s][:,it] for s in smr}
        elif alljk and flatten:
            yjk = np.hstack([self.data[s][:,it] for s in smr])

        return (xdata,ydata) if not alljk else (xdata,ydata,yjk)



def find_eps_cut(ratio:Ratio,trange,tol=1e+05,default=1E-12,**cov_specs):
    x,y, data = ratio.format(trange=trange,flatten=True,alljk=True,**cov_specs)    

    cov = np.cov(data.T) * (data.shape[0]-1)
    cdiag = np.diag(1./np.sqrt(np.diag(cov)))
    cor = cdiag @ cov @ cdiag

    eval,evec = np.linalg.eigh(cor)
    y = sorted(abs(eval))/max(eval)

    I=None
    for i,r in enumerate((y/np.roll(y,1))[1:]):
        if r>tol:
            I=i+1
            break

    return default if I is None else sorted(abs(eval))[I]







def main():
    ens = 'Coarse-1'
    rat = 'ZRA1'
    mom = '000'
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex'
    readfrom = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/report'

    reqs = ratio_prerequisites(
        ens      = ens,
        ratio    = rat,
        mom      = mom,
        readfrom = readfrom,
        jk       = True
    )

    io = RatioIO(ens,rat,mom,PathToDataDir=frm)
    # io.ReadRatio(jkBin=11, E0=0., m0=0.)
    
    ratio = Ratio(
        io,
        jkBin=11,
        smearing=['1S','RW'],
        # verbose=True,
        **reqs
    )

    # x,ydata = ratio.format()
    # plt.errorbar(x,gv.mean(ydata['1S']),gv.sdev(ydata['1S']),fmt='.')
    # plt.errorbar(x,gv.mean(ydata['RW']),gv.sdev(ydata['RW']),fmt='.')

    # plt.show()


    # # mom_list   = ['000','100','200','300','110','211']
    # mom_list   = ['000','100','200','300']#,'400']
    # ratio_list = ['RA1','ZRA1','ZRA1S','RA1S']#,'R0','R0S','R1','R1S','XV','XVS','XFSTPAR','XFSTBOT','XFSSTPAR','XFSSTBOT']

    # f,ax = plt.subplots(
    #     len(mom_list),len(ratio_list),
    #     layout="constrained",
    #     sharex=True,
    #     figsize=(20,10)
    # )

    # for i,ratio in enumerate(ratio_list):
    #     for j,mom in enumerate(mom_list):
    #         r = RatioIO(
    #             'Coarse-1',
    #             ratio,
    #             mom,
    #             PathToDataDir='/Users/pietro/code/data_analysis/BtoD/Alex'
    #         )

    #         try:
    #             d = r.ReadRatio(jkBin=11,E0=0,m0=0)
    #             print(r.specs)
    #         except Exception:
    #             d = None

    #         if d is not None:
    #             ydata = gv.gvar(
    #                 d['RW'].mean(axis=0),
    #                 np.cov(d['RW'],rowvar=False)*(d['RW'].shape[0]-1)
    #             )
    #             y  = gv.mean(ydata)
    #             ye = gv.sdev(ydata)
    #             x = np.arange(len(y)) 

    #             ax[j,i].errorbar(x,y,yerr=ye,fmt='.')
    #             ax[j,i].set_xticks([])
    #             ax[j,i].set_yticks([])
    #         else:
    #             ax[j,i].scatter([],[])
    #             ax[j,i].set_xticks([])
    #             ax[j,i].set_yticks([])

    #         if j==0:
    #             ax[0,i].set_title(ratio)
    #         if i==0:
    #             ax[j,0].set_ylabel(mom)

    #         print(mom,ratio)


    # plt.tight_layout()
    # # plt.savefig('/Users/pietro/Desktop/MediumCoarse.pdf')
    # plt.show()