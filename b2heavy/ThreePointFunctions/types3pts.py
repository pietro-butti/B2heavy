import os

from .. import FnalHISQMetadata


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

        match self.ratio:
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

    def RatioFileName(self,sm):
        specs = self.RatioSpecs()
        return f'{self.ratio}'


def test():
    io = RatioIO('Coarse-1','RA1','000',PathToDataDir='/Users/pietro/code/data_analysis/BtoD/Alex')

    print(io.mData['hSinks'])