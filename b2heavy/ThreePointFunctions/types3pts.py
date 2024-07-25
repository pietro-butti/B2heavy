import os
import h5py
import gvar   as gv
import numpy  as np
import xarray as xr
import pandas as pd

from copy import deepcopy

import matplotlib.pyplot as plt

from .. import FnalHISQMetadata
from ..TwoPointFunctions.utils   import jkCorr, compute_covariance

from .utils import read_config_fit, exists, exists_analysis
from ..FnalHISQMetadata import params


def ratiofmt(r:str):
    return r.replace('+','PLUS').replace('-','MINUS').upper()


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


def RatioSpecs(ratio, mData):
    bStr    = '_k' + mData['kBStr']
    cStr    = '_k' + mData['kDStr']
    sStr    = '_m' + mData['msStr']
    mStr    = '_m' + mData['mlStr']

    match ratiofmt(ratio):
        # ========================== B ---> D ==========================
        case 'XF':
            hStr   = cStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_V1_P5_']]
            nFacs  = [[1.]]
            dNames = [['P5_V4_P5_']]
            dFacs  = [[1.]]
            
            specs  = dict(
                source = 'D',
                sink   = 'D'
            )  

        case 'QPLUS':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr                
            nNames = [['P5_V4_P5_']]
            nFacs  = [[1.]]
            dNames = [['P5_V4_P5_']]
            dFacs  = [[1.]]

            specs  = dict(
                source = 'B',
                sink   = 'D'
            )                  

        case 'RMINUS':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_V1_P5_']]
            nFacs  = [[1.]]
            dNames = [['P5_V4_P5_']]
            dFacs  = [[1.]]

            specs = dict(
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
            dFacs  = [[1.],[1.]]

            specs = dict(
                num = ['B->D','D->B'],
                den = ['B->B','D->D']
            )



        # ========================== B ---> D* ==========================
        case 'RA1':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            # nNames = [['P5_A2_V2_'], ['P5_A2_V2_']] # 'R' # <--- Alex version
            # nFacs  = [[1., 1.], [ 1., 1.]]                # <--- Alex version    

            # nNames = [['P5_A2_V2_']]
            # nFacs  = [[1.]]   

            nNames = [['P5_A2_V2_','P5_A3_V3_']] #<--- should be the correct one 
            nFacs  = [[1., 1.]]       #<--- should be the correct one 
            # nFacs  = [[1., 1.], [ 1., 1.]]       #<--- should be the correct one 

            dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']         
            dFacs  = [[1.],[1.]]
            
            specs  = dict(
                source = ['B','Dst'],
                sink   = ['B','Dst']
            )

        case 'ZRA1':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            # nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
            # nFacs  = [[1., 1., 1.], [1., 1., 1.]]
            nNames = [['P5_A1_V1_'], ['V1_A1_P5_']]
            nFacs  = [[1.], [-1.]]
            dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
            dFacs  = [[1.],[1.]]
            
            specs  = dict(
                source = ['B','Dst'],
                sink   = ['B','Dst']
            )

        case 'R0':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_A4_V1_']]
            nFacs  = [[1.]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'B',
                sink   = 'Dst'
            )

        case 'R1':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_A1_V1_']]
            nFacs  = [[.5]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'B',
                sink   = 'Dst'
            )

        case 'XV':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_V3_V2_', 'P5_V2_V3_']]
            nFacs  = [[1., -1.]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            # dFacs  = [[-1.,-1.]]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'B',
                sink   = 'Dst'
            )

        case 'XFSTPAR':
            hStr   = cStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['V1_V1_V1_']]
            nFacs  = [[-1.]]
            dNames = [['V1_V4_V1_']]
            dFacs  = [[1.]]
            
            specs  = dict(
                source = 'Dst',
                sink   = 'Dst'
            )

        case 'XFSTBOT':
            hStr   = cStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['V1_V2_V1_', 'V1_V3_V1_']]
            nFacs  = [[1., 1.]]
            dNames = [['V1_V4_V2_', 'V1_V4_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'Dst',
                sink   = 'Dst'
            )


        # ========================== Bs ---> Ds* ==========================
        case 'RA1S':
            hStr   = bStr
            lStr   = cStr
            qStr   = sStr
            nNames = [['P5_A2_V2_', 'P5_A3_V3_'], ['V1_A1_P5_', 'V1_A2_P5_', 'V1_A3_P5_']] # 'R'
            nFacs  = [[1., 1.], [1., 1., 1.]]
            dNames = [['V1_V4_V1_'],['P5_V4_P5']] # 'V1_V4_V2_', 'V1_V4_V3_']
            dFacs  = [[1.],[1.]]
            
            specs  = dict(
                source = ['Bs','Dsst'],
                sink   = ['Bs','Dsst']
            )

        case 'ZRA1S':
            hStr   = bStr
            lStr   = cStr
            qStr   = mStr
            nNames = [['P5_A1_V1_', 'P5_A2_V2_', 'P5_A3_V3_']] # 'R'
            nFacs  = [[1., 1., 1.], [1., 1.]]
            dNames = [['V1_V4_V1_'],['P5_V4_P5_']] # 'V1_V4_V2_', 'V1_V4_V3_']
            dFacs  = [[1.],[1.]]
            
            specs  = {'num': None, 'den': None}

        case 'R0S':
            hStr   = bStr
            lStr   = cStr
            qStr   = sStr
            nNames = [['P5_A4_V1_']]
            nFacs  = [[1.]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'Bs',
                sink   = 'Dsst'
            )
        
        case 'R1S':
            hStr   = bStr
            lStr   = cStr
            qStr   = sStr
            nNames = [['P5_A1_V1_']]
            nFacs  = [[0.5]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'Bs',
                sink   = 'Dsst'
            )

        case 'XVS':
            hStr   = bStr
            lStr   = cStr
            qStr   = sStr
            nNames = [['P5_V3_V2_', 'P5_V3_V2_']]
            nFacs  = [[1., -1.]]
            dNames = [['P5_A2_V2_', 'P5_A3_V3_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
                source = 'Bs',
                sink   = 'Dsst'
            )

        case 'XFSSTPAR':
            hStr   = cStr
            lStr   = cStr
            qStr   = sStr
            nNames = [['V1_V1_V1_']]
            nFacs  = [[-1.]]
            dNames = [['V1_V4_V1_']]
            dFacs  = [[1.,1.]]
            
            specs  = dict(
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
            dFacs  = [[1.]]
            
            specs  = dict(
                source = 'Dsst',
                sink   = 'Dsst'
            )



    return { 
        'hStr'   : hStr  ,
        'lStr'   : lStr  ,
        'qStr'   : qStr  ,
        'nNames' : nNames,
        'nFacs'  : nFacs ,
        'dFacs'  : dFacs ,
        'dNames' : dNames,
        'specs'  : specs,
    }


def RatioFileString(name,ts,h,hss,q,l,p):
    return f'{name}T{ts}{h}_RW_{hss}_rot_rot{q}{l}_p{p}'


def RatioFileList(rstr,mom,mdata,sms=['RW','1S']):
    ratio = ratiofmt(rstr)

    specs = RatioSpecs(ratio,mdata)

    h = specs['hStr']
    q = specs['qStr']
    l = specs['lStr']

    tmp = []
    for smr in sms: 
        for t in sorted(mdata['hSinks']):
            rname = lambda n,h,q,l,sm: RatioFileString(n,t,h,sm,q,l,mom)
            
            nums = [[rname(n,h,q,l,smr) for n in s] for s in specs['nNames']]
            dens = [[rname(n,h,q,l,smr) for n in s] for s in specs['dNames']]
            facs = specs['nFacs']
            dfacs = specs['dFacs']

            # specific cases -----------------------------------
            h_or_l = lambda n: h if n.startswith('P5') else l
            if ratio in ['ZRA1','ZRA1S']: # switch heavy with light in names
                nums[1] = (
                    [rname(n,l,q,h,smr) for n in specs['nNames'][1]]
                )
                dens = [
                    [rname(n, h_or_l(n), q, h_or_l(n),'1S' if n.startswith('P5') else smr)  for n in s] 
                    for s in specs['dNames']
                ]

            if ratio in ['RA1','RA1S','QPLUS']: # at denom. mom must be 0
                rname0 = lambda n,h,q,l,sm: RatioFileString(n,t,h,sm,q,l,'000')

                dens = [
                    [rname0(n, h_or_l(n), q, h_or_l(n), '1S' if n.startswith('P5') else smr)  for n in s] 
                    for s in specs['dNames']
                ]
            
            if ratio=='RPLUS':
                nums = [
                    [RatioFileString('P5_V4_P5_',t,h,smr,q,l,'000')],
                    [RatioFileString('P5_V4_P5_',t,l,smr,q,h,'000')],
                ]
                dens = [
                    [RatioFileString('P5_V4_P5_',t,h,smr,q,h,'000')],
                    [RatioFileString('P5_V4_P5_',t,l,smr,q,l,'000')],
                ]

            if ratio in ['RMINUS','RMINUS2','RMINUS3']:
                dir = 1 if ratio=='RMINUS' else ratio[-1]
                nums = [[RatioFileString(f'P5_V{dir}_P5_',t,h,smr,q,l,mom)]]
                dens = [[RatioFileString('P5_V4_P5_'     ,t,h,smr,q,l,mom)]]

            if ratio in ['XF','XF2','XF3']:
                dir = 1 if ratio=='XF' else ratio[-1]
                nums = [[RatioFileString(f'P5_V{dir}_P5_',t,l,smr,q,l,mom)]]
                dens = [[RatioFileString('P5_V4_P5_'     ,t,l,smr,q,l,mom)]]

            if ratio=='QPLUS':
                nums = [[RatioFileString('P5_V4_P5_',t,h,smr,q,l,mom)]]
                dens = [[RatioFileString('P5_V4_P5_',t,h,smr,q,l,'000')]]
            # ---------------------------------------------------


            tmp.append({
                'smearing': smr,
                't_sink'  : t,
                'num'     : nums, 
                'den'     : dens,
                'facs'    : facs ,
                'dfacs'   : dfacs
            })

    return pd.DataFrame(tmp).set_index(['smearing','t_sink'])


def ratio_prerequisites(ens,ratio,mom,smearing=['1S','d'],readfrom=None,jk=False,w_from_corr3=True):
    req = dict(
        E0      = None,
        m0      = None,
        Z0      = {s: None for s in smearing},
        Zpar    = {s: None for s in smearing},
        Zbot    = {s: None for s in smearing},
        Zp      = {s: None for s in smearing},
        wrecoil = None,
        jk      = jk
    )

    rt = ratiofmt(ratio)
    if rt in ['XF','QPLUS','RPLUS','RMINUS']:
        mes = 'D'
    elif rt in ['RA1','ZRA1','XFSTPAR','XFSTBOT','XV','R0','R1']:
        mes = 'Dst'
    elif rt in ['RA1S','ZRA1S','XFSSTPAR','XFSSTBOT','XVS','R0S','R1S']:
        mes = 'Dsst'

    assert exists_analysis(readfrom,ens,mes,mom  ,type='2',jkfit=jk)
    assert exists_analysis(readfrom,ens,mes,'000',type='2',jkfit=jk)

    p2 = read_config_fit(
        f'fit2pt_config_{ens}_{mes}_{mom}',
    path=readfrom,jk=jk)
    p0 = read_config_fit(
        f'fit2pt_config_{ens}_{mes}_000',
    path=readfrom,jk=jk)

    req['E0'] = p2['dE'][:,0] if jk else p2[-1]['dE'][0].mean
    req['m0'] = p0['dE'][:,0] if jk else p0[-1]['dE'][0].mean


    for sm in smearing:
        if not jk:
            req['Z0'][sm] = (np.exp(p0[-1][f'Z.{sm}.Unpol'][0])**2).mean * 2*req['m0'] 
        else:
            req['Z0'][sm] = np.exp(p0[f'Z.{sm}.Unpol'][:,0])**2 * 2*p0['dE'][:,0]     

        if mes in ['Dst','Dsst'] and mom!='000':
            if not jk:
                req['Zpar'][sm] = (np.exp(p2[-1][f'Z.{sm}.Par'][0])**2).mean * 2*req['E0']
                req['Zbot'][sm] = (np.exp(p2[-1][f'Z.{sm}.Bot'][0])**2).mean * 2*req['E0']                
            else:
                req['Zpar'][sm] = np.exp(p2[f'Z.{sm}.Par'][:,0])**2 * 2*p2['dE'][:,0]
                req['Zbot'][sm] = np.exp(p2[f'Z.{sm}.Bot'][:,0])**2 * 2*p2['dE'][:,0]     

        elif mes=='D':
            if not jk:
                req['Zp'][sm] = (np.exp(p2[-1][f'Z.{sm}.Unpol'][0])**2).mean * 2*req['E0']
            else:
                req['Zp'][sm] = np.exp(p2[f'Z.{sm}.Unpol'][:,0])**2 * 2*p2['dE'][:,0]     

    if rt in ['RA1','RA1S']:
        if w_from_corr3:
            assert exists_analysis(readfrom,ens,'xfstpar',mom,type='3',jkfit=jk)
            tag = f'fit3pt_config_{ens}_xfstpar_{mom}'
            p3 = read_config_fit(tag, path=readfrom, jk=jk)
            xf = p3['ratio'].reshape(p3['ratio'].shape[-1]) if jk else p3[-1]['ratio'][0].mean
            req['wrecoil'] = (1+xf**2)/(1-xf**2)
            # req['wrecoil'] = req['E0']/req['m0']
        else:
            req['wrecoil'] = req['E0']/req['m0']
    else:
        req['wrecoil'] = req['E0']/req['m0']

    return req


def ratio_correction_factor(rstr,smearing=['RW','1S'],**req):
    ratio = ratiofmt(rstr)

    factor = {}
    for sm in smearing:
        smi = 'd' if sm=='RW' else sm

        ff = 1.
        match ratio:
            case ratio if ratio in ['R0','R1','R0S','R1S']:
                ff = 1. / np.sqrt(
                    req['Zbot'][smi]/req['Zpar'][smi]
                )

            case ratio if ratio in ['RA1','RA1S']:
                # ff = 1./ req['wrecoil']**2 * \
                ff = (req['m0']/req['E0'])**2 * \
                    req['Zbot'][smi]/np.sqrt( req['Z0'][smi] * req['Z0']['1S'] )

            case ratio if ratio in ['ZRA1','ZRA1S']:
                ff = req['Z0'][smi] / np.sqrt( req['Z0'][smi] * req['Z0']['1S'] ) 

            case 'QPLUS':
                ff = 1./ (
                    req['E0']/req['m0'] * \
                    np.sqrt(req['Z0'][smi]/req['Zp'][smi]) 
                )
                
        factor[sm] = 1./ff

    return factor if bool(factor) else None


def func(ratio):
    match ratiofmt(ratio):
        case ratio if ratio in ['ZRA1','ZRA1S']:
            tmp = dict(fnum=np.sum, fden=np.mean, reflect=False)
        case ratio if ratio in ['RA1','RA1S']:
            tmp = dict(fnum=np.mean, fden=np.sum, reflect=True)
        case ratio if ratio in ['XV','XVS','R1','R1S']:
            tmp = dict(fnum=np.mean, fden=np.mean, reflect=False)
        case _:
            tmp = dict(fnum=np.sum, fden=np.mean, reflect=False)

    return tmp



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
        if   _mom=='000' and _rat not in ['ZRA1','ZRA1S','RPLUS','RMINUS','XF','QPLUS']:
            raise NotImplementedError(f'At {_mom}, only ZRA1, ZRA1S, RPLUS are defined')
        elif _mom!='000' and _rat in ['ZRA1','ZRA1S','RPLUS']:
            raise NotImplementedError(f'{_rat} has not been defined for {_mom}>0')    

        dname = f'{_ens}_{_rat}_p{_mom}' if name is None else name

        rstr = _rat.replace('+','PLUS').replace('-','MINUS')

        self.info   = RatioInfo(dname,_ens,rstr,_mom)
        self.pmom   = [int(px) for px in self.info.momentum]
        self.mData  = FnalHISQMetadata.params(_ens)

        self.Ta, self.Tb = self.mData['hSinks']

        self.files = RatioFileList(self.info.ratio,_mom,self.mData)

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

        return


    def read(self, files=None , sms=['RW','1S'], jkBin=None, verbose=False, fnum=np.mean, fden=np.sum, reflect=False, readfrom=None):
        """
            This functions reads and build a raw ratio (without any renormalization or exp(E0-m0) correction factor).

            Given a dataframe `files`, output of `RatioFileList` function, it does the following for every smearing and every :
            - goes through the list of 3pf in the list numerators and reads them (multiply for `nFac`)
            - apply jackknife resampling
            - takes the product (effective in case of double ratio)
            - averages (or sum) over space directions (if vector meson case)
            - build the ratio
        """


        # Here we set the file list (if nothing is specified, we look at self.files)
        files = self.files if files is None else files
        if files is None:
            raise Exception(f'`RatioFileList` function must be called first')

        jk = 0 if jkBin is None else jkBin

        readfrom = [self.RatioFile] if readfrom is None else readfrom
        readhdf5 = lambda name: read_ratio_from_files_list(name,*readfrom,verbose=verbose)

        tmp = {}
        for sm in sms:
            tmp[sm] = {}
            for ts in sorted(self.mData['hSinks']):
                tmp[sm][ts] = {}

                # Here we stack 3pf for numerator
                nums = []
                for namelst, flst in zip(
                    files.loc[sm,ts]['num'],
                    files.loc[sm,ts]['facs']
                ):
                    num = np.array([
                        # jkCorr(readhdf5(name)*f, bsize=jk)[:,:(self.Ta+1)]
                        jkCorr(readhdf5(name)*f, bsize=jk)[:,:(ts+1)]
                        for name,f in zip(namelst,flst)
                    ])
                    nums.append(num)

                if reflect: # if RA1, timeslice must be reflected
                    aux = []
                    for x in nums:
                        num = np.flip(x,axis=-1)
                        aux.append(num)
                    nums = np.concatenate((nums,aux))


                # Here we stack 3pf for denominator
                dens = []
                for namelst,flst in zip(
                    files.loc[sm,ts]['den'],
                    files.loc[sm,ts]['dfacs']
                ):
                    den = [
                        jkCorr(readhdf5(name)*f, bsize=jk)[:,:(ts+1)]
                        for name,f in zip(namelst,flst)
                    ]
                    dens.append(den)


                nums = np.array(nums) # FIXME this works only if the double ratio is the product of 3pts with the same number of spatial component
                dens = np.array(dens)

                # Here we take product over dim=0 (only effective in case of multiple ratio)
                # dim=0 is different ratio types
                nums = nums.prod(axis=0)
                dens = dens.prod(axis=0)
                
                # Here we average or sum over directions
                nums = fnum(nums,axis=0)
                dens = fden(dens,axis=0)

                rt = nums/dens
                tmp[sm][ts] = rt[:,:(self.Ta+1)]

        return tmp


    def correct(self, raw, factor=None): # the first key of processed must be smearing, the second must be sink times
        corrected = {}

        if factor is None:
            corrected = raw
        elif not isinstance(factor, dict):
            raise TypeError(f'[factor] must be a dictionary with matching key as {raw = }')
        else:
            for sm in raw:
                corrected[sm] = {}
                for ts in raw[sm]:
                    ff = factor[sm]

                    if isinstance(ff,float):
                        corrected[sm][ts] = raw[sm][ts] * ff
                    elif isinstance(ff,np.ndarray):
                        corrected[sm][ts] = raw[sm][ts] *  np.tile(ff,(raw[sm][ts].shape[-1],1)).T

        return corrected


    def smooth(self, corrected, **kwargs):
        E0 = kwargs.get('E0')
        m0 = kwargs.get('m0')

        f1, f2 = 1.,1.
        if self.info.ratio in ['RA1','RA1S','QPLUS']:
            if E0 is None and m0 is None:
                raise Exception('Energy and rest mass must be provided')
            elif isinstance(E0, (int,float)) and isinstance(E0, (int,float)):
                perjk = False
            elif isinstance(E0, np.ndarray) and isinstance(m0, np.ndarray): 
                perjk = True
                if E0.shape!=m0.shape:
                    raise Exception(f'E0 ({E0.shape = }) and m0 ({m0.shape = }) must have compatible shapes ')

            if self.info.ratio in ['RA1','RA1S']:
                f1 = np.exp((E0 - m0)*self.Ta)
                f2 = np.exp((E0 - m0)*self.Tb)
                if perjk:
                    f1 = np.tile(f1,(self.Ta+1,1)).T
                    f2 = np.tile(f2,(self.Ta+1,1)).T

            elif self.info.ratio=='QPLUS':
                if not perjk:
                    f1 = np.exp((E0-m0) * np.arange(self.Ta+1))
                else:
                    tt = np.tile(np.arange(self.Ta+1),(len(E0),1))
                    dE = np.tile(E0-m0,(self.Ta+1,1)).T
                    f1 = np.exp(dE*tt)

                f2 = f1

        data = {}
        for sm in corrected:
            data[sm] = \
                f1/2. *   corrected[sm][self.Ta] + \
                f2/4. * ( 
                            corrected[sm][self.Tb] + \
                    np.roll(corrected[sm][self.Tb], -1, axis=1)
                )       
                # f1/2. *   corrected[sm][self.Ta][:,:self.Ta+1] + \
                # f2/4. * ( corrected[sm][self.Tb][:,:self.Ta+1] + \
                #     np.roll(corrected[sm][self.Tb], -1, axis=1)[:,:self.Ta+1]     
                # )        

        return data


    def build(self, smearing=['RW','1S'], jkBin=None, verbose=False, **reqs):
        # import 3pf from default datafiles, compute raw ratios
        fcts = func(self.info.ratio)
        
        raw = self.read(
            sms     = smearing,
            jkBin   = jkBin,
            verbose = verbose,
            **fcts
        )

        # correct ratio with apposite (re)normalizatio/kinematic factors
        ff = ratio_correction_factor(
            self.info.ratio,
            **reqs
        )
        corrected = self.correct(raw,factor=ff)

        # smoothen ratio 
        data = self.smooth(corrected, **reqs)
        return data


class Ratio:
    def __init__(self, io:RatioIO, jkBin=None, smearing=['1S','RW'], verbose=False, **requisites):
        self.io           = io
        self.info         = io.info
        self.info.binsize = jkBin
        self.binsize      = jkBin

        self.data = io.build(
            smearing = smearing,
            jkBin    = jkBin,
            verbose  = verbose,
            **requisites
        )
        self.smr = sorted(list(self.data.keys()))
        
        shapes = list({self.data[sm].shape for sm in self.data})
        assert len(shapes)==1

        self.nbins, tmax = shapes[0]
        self.timeslice = np.arange(tmax)

        # Check if some element in requisites is None *and* is a vector (jackknife)
        _req = deepcopy(requisites)
        if requisites.get('jk'):
            _req = {}
            for k,rq in requisites.items():
                if not k.startswith('jk') and rq is not None:
                    if isinstance(rq,dict):
                        _req[k] = {k: rq[k].mean() for k in rq if rq[k] is not None}
                    else:

                        _req[k] = rq.mean()
                else:
                    _req[k] = rq

        self.data_full = io.build(
            smearing = smearing,
            jkBin    = 1,
            verbose  = verbose,
            **_req
        )
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
            ally = {s: self.data_full[s][:,it] for s in smr}
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


    def plot(self, ax, **kwargs):
        x, y = self.format()
        for i,sm in enumerate(self.smr):
            ax.errorbar(
                x,
                gv.mean(y[sm]),
                yerr = gv.sdev(y[sm]),
                fmt    = 'o' , 
                ecolor = kwargs['col'][sm] if 'col' in kwargs else f'C{i}',
                color  = kwargs['col'][sm] if 'col' in kwargs else f'C{i}',
                capsize=2.5,
                mfc='w',
                label=sm
            )






def main():
    ens = 'Coarse-1'
    r   = 'XV'
    mom = '100'
    frm = '/Users/pietro/code/data_analysis/BtoD/Alex'
    readfrom = '/Users/pietro/Desktop/lattice24/0.25/corr2_3'

    req = ratio_prerequisites(ens,r,mom,readfrom=readfrom)

    io = RatioIO(ens,r,mom,PathToDataDir=frm)
    robj = Ratio(
        io,
        jkBin     = 11,
        smearing  = ['1S','RW'],
        **req
    )

    breakpoint()