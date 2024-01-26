import os
import h5py

import numpy  as np
import autograd 
from autograd import numpy as np

import xarray as xr
import gvar   as gv
import lsqfit
import sys

from ..     import FnalHISQMetadata
from .utils import jkCorr, covariance_shrinking, ConstantModel

ENSEMBLE_LIST = ['MediumCoarse', 'Coarse-2', 'Coarse-1', 'Coarse-Phys', 'Fine-1', 'SuperFine', 'Fine-Phys'] 
MESON_LIST    = ["Dsst", "Bs", "D", "Ds", "Dst", "Dsst", "K", "pi", "Bc", "Bst", "Bsst"]
MOMENTUM_LIST = ["000", "100", "200", "300", "400", "110", "211", "222"]

class CorrelatorInfo:
    """
        This is a basic structure to contain information for 2pt functions for the following given attributes:
        - ensemble
        - meson
        - momentum
    """
    def __new__(cls, _name:str, _ens:str, _mes:str, _mom:str):
        if _ens not in ENSEMBLE_LIST: 
            raise KeyError(f'{_ens} is not a valid ensemble.')
        if _mes not in MESON_LIST: # Check whether meson is recognized
            raise KeyError(f'{_mes} is not a valid meson.')
        if _mom not in MOMENTUM_LIST: # Check whether momentum is recognized
            raise KeyError(f'{_mom} is not a valid momentum.')
        return super().__new__(cls)
        # if _file is not None and not os.path.isfile(_file): # Check whether input file exist
        #     raise RuntimeError(f'{_file} has not been found.')

    def __init__(self, _name:str, _ens:str, _mes:str, _mom:str):
        self.name     = _name
        self.ensemble = _ens
        self.meson    = _mes
        self.momentum = _mom
        self.binsize  = None
        self.filename = None

    def __str__(self):
        return f' # ------------- {self.name} -------------\n # ensemble = {self.ensemble}\n #    meson = {self.meson}\n # momentum = {self.momentum}\n #  binsize = {self.binsize}\n # filename = {self.filename}\n # ---------------------------------------'

class CorrelatorIO:
    """
        This is the basic structure that deals with input/output of the correlator data.

        Data are read from a typical hdf5 archive, e.g. `l3264f211b600m00507m0507m628.hdf5` that can be provided by the additional argument `PathToFile`.
        
        If no other additional argument are specified, the location of the hdf5 archive is *automatically* inferred from `FnalHISQMetadata`. The location of the ensemble folder *must* be specified with `PathToDataDir`.
    """
    def __init__(self, _ens:str, _mes:str, _mom:str, PathToFile=None, PathToDataDir=None, name=None):
        dname = f'{_ens}_{_mes}_p{_mom}' if name is None else name

        self.info  = CorrelatorInfo(dname,_ens,_mes,_mom)
        self.pmom  = [int(px) for px in self.info.momentum]
        self.mData = FnalHISQMetadata.params(_ens)

        if PathToFile is not None:
            self.CorrFile = PathToFile
        elif PathToDataDir is not None:
            path = os.path.join(PathToDataDir,self.mData['folder'],self.mData['hdf5File2'])
            if os.path.exists(path):
                self.CorrFile = path
            else:
                raise FileNotFoundError(f'The file {path} has not been found')
        else:
            raise FileNotFoundError(f'Please specify as optional argument either data file in `PathToFile` or `PathToDataDir` to look at default data file.')

    def checkSmear(self, myStr):
        return '1S' if '1S' in myStr else 'rot' if 'rot' in myStr else 'd'

    def CorrFileList(self, sms=None):
        # Set informations for reading
        SMR = self.mData['sms'] if sms is None else sms

        meson = self.info.meson
        mom   = self.info.momentum

        # Infer information for file extraction
        B       = meson.startswith('B')
        D       = meson.startswith('D')

        charm   = 'c' in meson
        star    = meson.endswith('st')
        strange = (meson[-1]=='s') if not star else (meson.count('s')==2)

        channel = ['P5_P5'] if not star else ['V1_V1', 'V2_V2', 'V3_V3']
        heavy   = f'k{self.mData["kBStr"]}' if B else (f'k{self.mData["kDStr"]}' if D else None)
        light   = f'k{self.mData["kDStr"]}' if charm else (f'm{self.mData["msStr"]}' if strange else f'm{self.mData["mlStr"]}')

        # Read data from master file
        with h5py.File(self.CorrFile,'r') as d:
            data = d['data']

            # Cycle over all possible smearing, build filename and check if 
            cdict = {}
            for lss in SMR:
                for hss in SMR:
                    if meson=='K':
                        newfiles = [f'P5_P5_{lss}_{hss}_m{self.mData["msStr"]}_m{self.mData["mlStr"]}_p{mom}']
                        continue
                    elif meson=='pi':
                        newfiles = [f'P5_P5_{lss}_{hss}_m{self.mData["mlStr"]}_m{self.mData["mlStr"]}_p{mom}']
                    else:
                        newfiles = [f'{ch}_{lss}_{hss}_{heavy}_{light}_p{mom}' for ch in channel]
                    
                    aux = [f for f in newfiles if f in data]
                    
                    smr_string = f'{self.checkSmear(lss)}-{self.checkSmear(hss)}'
                    # clist.append((aux,smr_string)) if aux else 1
                    if aux:
                        cdict[smr_string] = aux

            return cdict

    def SmearingList(self,CrossSmearing=False):
        SMR = sorted(self.CorrFileList().keys())

        if not CrossSmearing:
            source = set(np.concatenate([s.split('-') for s in SMR]))
            cross = [f'{s1}-{s2}' for s1 in source for s2 in source if s1!=s2 and f'{s1}-{s2}' in SMR and f'{s2}-{s1}' in SMR]
        else:
            cross = []

        return SMR, cross

    def ReadCorrelator(self, CrossSmearing=False, jkBin=None, verbose=False):
        # Info about correlator
        T  = self.mData['T']//2
        collinear = self.pmom[1:]==[0,0] and not self.pmom[0]==0
        vector    = False

        CorrNameDict = self.CorrFileList() 
        SMR,cross    = self.SmearingList(CrossSmearing=CrossSmearing)    

        # Read and store correlator ---------------------------------------------
        PROCESSED = {}
        with h5py.File(self.CorrFile,'r') as f:
            RAW = f['data'] # read data from hdf5 archive

            for sm in SMR:
                # ------------------ Fold correlator ------------------
                corrs = []
                for corr_name in CorrNameDict[sm]:
                    corrs.append(
                        0.5 * ( RAW[corr_name][:,:T] + 
                            np.flip(np.roll(RAW[corr_name], -1, axis=1), axis=1)[:,:T] )
                    )
                corrs = np.array(corrs)

                # ------------------ Store correlator ------------------
                if len(corrs)==1: # scalar correlator 
                    PROCESSED[(sm,'Unpol')] = corrs[0]
                
                elif len(corrs)==3: # vector correlator
                    vector = True

                    if collinear:
                        PROCESSED[sm,'Par'] = corrs[0]
                        PROCESSED[sm,'Bot'] = corrs[1:].mean(axis=0) 
                    else:
                        PROCESSED[sm,'Unpol'] = corrs.mean(axis=0)
        

        # Prepare polarization list -----------------------------------------------
        POL = ['Par','Bot'] if collinear and vector else ['Unpol']

        if not CrossSmearing:
            # Average cross smearing --------------------------------------------------
            for sm in SMR:
                if sm in cross:
                    s1,s2 = sm.split('-')
                    for pol in POL:
                        PROCESSED[sm,pol] = 0.5*(
                            PROCESSED[sm,pol] + PROCESSED[f'{s2}-{s1}',pol]
                        )

            # Delete unnnecessary other
            # it is done like this because alex mantains d-1S and not 1S-d
            a = set()
            for sm in cross:
                s1,s2 = sorted(sm.split('-'))
                a.add(f'{s1}-{s2}')
            
            for crs in a:
                SMR.remove(crs)
                for pol in POL:
                    del PROCESSED[(crs,pol)]

        # Stack data --------------------------------------------------------------
        DATA = np.array([[jkCorr(PROCESSED[smr,pol],bsize=(jkBin if jkBin is not None else 0)) for pol in POL] for smr in SMR])    
        (Nsmr,Npol,Nconf,Nt) = DATA.shape

        xd = xr.DataArray(
            DATA,
            dims   = ['smearing','polarization','jkbin','timeslice'],
            coords = [SMR,POL,np.arange(Nconf),np.arange(Nt)],
            attrs  = {
                'ensemble': self.info.ensemble,
                'meson'   : self.info.meson,
                'momenutm': self.info.momentum,
                'list'    : CorrNameDict,
            },
            name   = self.info.name
        )
        

        if verbose:
            print(f'Correlators for ensemble {self.info.ensemble} for meson {self.info.meson} at fixed mom {self.info.momentum} read from')
            for sm,l in CorrNameDict.items():
                print(sm,'--->',l)


        return xd

class Correlator:
    """
        This is the basic structure for a correlator at fixed momentum. It contains data for all possible smearings and polarization.
    """
    def __init__(self, io:CorrelatorIO, jkBin=None, **kwargs):
        self.io   = io
        self.data = io.ReadCorrelator(jkBin=jkBin,**kwargs)
        self.Nt   = 2*self.data.timeslice.size
        self.info = io.info  
        self.info.binsize = jkBin

    def __str__(self):
        return f'{self.info}'

    def jack(self, bsize):
        self.info.binsize = bsize

        newshape = (
            self.data.shape[0],
            self.data.shape[1],
            self.data.shape[2]//bsize,
            self.data.shape[3],
        )
        MT = np.empty(newshape)

        for ism,sm in enumerate(self.data.smearing):
            for ipol,pol in enumerate(self.data.polarization):
                MT[ism,ipol,:,:] = jkCorr(self.data.loc[sm,pol,:,:].values, bsize)

        newcoords = [
            self.data.smearing.values,
            self.data.polarization.values,
            np.arange(newshape[2]),
            self.data.timeslice.values
        ]
        newattrs  = {
            'ensemble': self.info.ensemble,
            'meson'   : self.info.meson,
            'momenutm': self.info.momentum,
            'binsize' : bsize,
            'list'    : self.data.attrs['list'],
        }
        xd = xr.DataArray(
            MT,
            dims   = ['smearing','polarization','jkbin','timeslice'],
            coords = newcoords,
            attrs  = newattrs, 
            name   = self.info.name
        )
        self.data = xd
        
        return

    def format(self, trange=None, flatten=False, covariance=True, scale_covariance=False, shrink_covariance=False, smearing=None, polarization=None, alljk=False):
        """
            Preprocess correlator and returns argument `(xfit,yfit)` for `data` field in `lsqfit.nonlinear_fit`.

            Arguments
            ----------
            trange: tuple, optional
                A tuple containing `tmin` and `tmax` for the fit as `(tmin,tmax)`. 
                Default value set to `None` (`self.data` is untouched). If a tuple `(tmin,tmax)` is provided slicing is performed `self.data[:,:,:,range(tmin,tmax+1)]`.
            flatten: bool, optional
                Default value set to `False`: `xfit` and `yfit` will be given as dictionaries whose keys are tuples `(smr,pol)`. If `covariance` flag is set to `true`, data are flattened first, the **full** covariance matrix is calculated, then data are reshaped again.
                If set to `True`:  Data are flattened each timeslice **for each smearing and polarization** is considered as a separate variable. 
            covariance: bool, optional
                Default value set to `True`. Data are averaged and full covariance matrix is calculated and passed to `gvar`. If set to false, only diagonal covariance matrix is passed as standard deviation.
            scale_covariance: bool, optional
                Default value set to `False`. If `True`, covariance of the jackknife covariance matrix is scaled with the diagonal of the covariance matrix of the unbinned data.
            shrink_covariance: bool, optional
                Default value set to `False`. If `True`, covariance of the jackknife covariance matrix is shrunk according to (? FIXME).
            jknorm: bool, optional
                Default value set to `True`. Covariance matrix is normalized with `(Nb-1)/Nb`, `Nb` being `data.shape[2]`
            smearing: list, optional
                Default value set to `None`: all smearings are considered. If a list of `str` is provided only those smearings are considered.
            polarization: list, optional
                Default value set to `None`: all polarizations are considered. If a list of `str` is provided only those polarizations are considered.

            Returns
            ---------
            xfit: ndarray
                Array of floats with x-variables for fit
            yfit: 
                Array of gv.gvar variables for y-variables in fit
        """
        # Select smearing to be considered
        if not smearing==None:
            if False not in np.isin(smearing,self.data.smearing):
                SMR = sorted(smearing)
            else:
                raise ValueError(f'Smearing list {smearing} contain at least one item which is not contained in data.smearing')
        else:
            SMR = sorted(self.data.smearing.values)

        # Select polarization to be considered
        if not polarization==None:
            if False not in np.isin(polarization,self.data.polarization):
                POL = sorted(polarization)
            else:
                raise ValueError(f'Polarization list {polarization} contains at least one item which is not contained in data.polarization')
        else:
            POL = sorted(self.data.polarization.values)

        # Build a list of ordered keys (smearing, pol)
        keys = sorted([(smr,pol) for smr in SMR for pol in POL])

        # Slice data in time-range considered
        if trange is not None:
            Trange = self.data.timeslice.isin(np.arange(min(trange),max(trange)+1))
        else:
            Trange = self.data.timeslice.values
        X = self.data.timeslice[Trange].to_numpy()

        ydata = self.data.loc[SMR,POL,:,Trange]
        if scale_covariance and self.info.binsize is not None:
            ydata_full = self.io.ReadCorrelator(jkBin=1).loc[SMR,POL,:,Trange]
            # jkBin in this must be 1 or None?  

        # Flatten data temporarily to calculate covariances
        yaux = np.hstack([ydata.loc[smr,pol,:,:] for (smr,pol) in keys])

        if covariance:
            factor = ((yaux.shape[0]-1) if self.info.binsize is not None else 1./yaux.shape[0])
            cov = np.cov(yaux,rowvar=False) * factor
            
            if not scale_covariance:
                COV = cov
            else:
                yaux_full = np.hstack([ydata_full.loc[smr,pol,:,:] for (smr,pol) in keys])
                cov_full = np.cov(yaux_full,rowvar=False) * (yaux_full.shape[0]-1)
                scale = np.sqrt(np.diag(cov)/np.diag(cov_full))
                COV = cov_full * np.outer(scale,scale)

            if shrink_covariance:
                if scale_covariance:
                    COV = covariance_shrinking(yaux_full.mean(axis=0),COV,yaux_full.shape[0]-1)
                else:
                    COV = covariance_shrinking(yaux.mean(axis=0),COV,yaux.shape[0]-1)

            Y = gv.gvar(yaux.mean(axis=0),COV)
        
        else:
            Y = gv.gvar(yaux.mean(axis=0),yaux.std(axis=0) / np.sqrt(yaux.shape[0]-1) )

        # In case flatten flag is active, reshape 
        if flatten: 
            xfit = np.concatenate([X for _ in keys])
            yfit = Y
        else:
            xfit,yfit,yjk = {},{},{}
            for i,(smr,pol) in enumerate(keys):
                xfit[(smr,pol)] = X
                yfit[(smr,pol)] = Y[(len(X)*i):(len(X)*(i+1))]
                yjk[(smr,pol)]  = ydata.loc[smr,pol,:,:].values
        
        return (xfit,yfit) if not alljk else (xfit,yfit,yjk)

    def EffectiveMass (self, trange=None, variant='log', mprior=None, verbose=True, **kwargs):
        """
            Calls `format` and compute effective mass from the correlator as `log C(t)/C(t+2)`

            Arguments
            ---------
            fit: bool, optional
                Default value is set to `True`: returns also a dictionary with the effective masses obtained as
            kwargs: optional
                Same arguments that can be fed to `format`. `flatten` must *not* be specified.

            Returns
            ---------
            (x,y) 
                Same output of `format`
            MEFF
                Result of the fit
            prior

        """

        # Compute effective mass according to the definition
        (x,y) = self.format(flatten=False, **kwargs)
        meff = {}
        for k,c in y.items():
            if variant=='log':
                meff[k] = np.log(c/np.roll(c,-2))/2 
            elif variant=='arccosh':
                meff[k] = np.arccosh(((np.roll(c,-1) + np.roll(c,1))/c/2))/2

        xdic, ydic = {}, {}
        for k,m in meff.items():
            iifit = np.arange(len(m)) if trange is None else [i for i,x in enumerate(x[k]) if x>=min(trange) and x<=max(trange)]
            xdic[k] = x[k][iifit]
            ydic[k] = m[iifit]
        
        xfit = np.concatenate(xdic.values())
        yfit = np.concatenate(ydic.values())
        
        if mprior is None:
            aux = gv.mean(yfit).mean()
            pr = gv.gvar(aux,aux)
        else:
            pr = mprior

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data=(xfit,yfit),
            fcn=ConstantModel,
            prior={'const': pr}
        )
        MEFF = fit.p['const']

        if verbose:
            print(fit)

        return (x,meff), MEFF, pr

    def EffectiveCoeff(self, trange=None, variant='log', aprior=None, mprior=None, verbose=True, **kwargs):
        # Compute effective mass according to the definition
        (X,meff), MEFF, m_pr = self.EffectiveMass(trange=trange, variant=variant, verbose=verbose, mprior=mprior, **kwargs)
        (_,C) = self.format(flatten=False, **kwargs)

        apr = {}
        aeff = {}
        AEFF = {}
        for k,x in X.items():
            if variant=='log':
                Aeff = np.exp(MEFF.mean*x) * C[k]   #*Y[k][iok]
            elif variant=='arccosh':
                Aeff = C[k]/(np.exp(-MEFF.mean*C[k]) + np.exp(-MEFF.mean*(self.Nt-x)))
            aeff[k] = Aeff

            iok = np.arange(len(x)) if trange is None else [i for i,x in enumerate(x) if x>=min(trange) and x<=max(trange)]
            Aeff = Aeff[iok]

            if aprior is None:
                pr = {'const': gv.gvar(gv.mean(Aeff).mean(),gv.mean(Aeff).mean())}
            else:
                pr = {'const': aprior[k]}

            fit = lsqfit.nonlinear_fit(
                data=(x[iok],Aeff),
                fcn=ConstantModel,
                prior=pr
            )
            if verbose:
                print(k,fit)
        
            AEFF[k] = fit.p['const']
            apr[k] = pr['const']
        
        return (X,meff,aeff), MEFF,AEFF, m_pr, apr





def test():
    ens      = 'MediumCoarse'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex'
    meson    = 'Dsst'
    mom      = '000'
    binsize  = 13
    
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr =  Correlator(io,jkBin=binsize)

    corr.EffectiveMass(trange=(10,19),smearing=['d-d', '1S-1S', 'd-1S'])


if __name__ == "__main__":
    test()
