import os
import h5py

import matplotlib.pyplot as plt
import numpy  as np
import scipy
# import autograd 
# from autograd import numpy as np

from scipy.optimize import curve_fit

from warnings import warn

import xarray as xr
import gvar   as gv
import lsqfit
import sys

from ..     import FnalHISQMetadata
from .utils import jkCorr, covariance_shrinking, ConstantModel, ConstantFunc

ENSEMBLE_LIST = ['MediumCoarse', 'Coarse-2', 'Coarse-1', 'Coarse-Phys', 'Fine-1', 'SuperFine', 'Fine-Phys'] 
MESON_LIST    = ["Dsst", "Bs", "B", "D", "Ds", "Dst", "Dsst", "K", "pi", "Bc", "Bst", "Bsst"]
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

    def correlation_diagnostics(self, ysamples, verbose=True, plot=False):
        '''
            Analyze the correlation between data
        
            Parameters
            ----------
                - `ysamples`: array: Each column contains all the samples (MC chain) of a single observables.
        '''
        # Compute covariance of data
        ydata = gv.dataset.avg_data(ysamples,spread=True)
        cov = gv.evalcov(ydata)*(ydata.shape[0]-1 if self.info.binsize is not None else 1.)

        # Compute correlation matrix
        cdiag = np.diag(1./np.sqrt(np.diag(cov)))
        cor = cdiag @ cov @ cdiag

        # Compute condition number of the matrix.
        condn = np.linalg.cond(cor)
        if condn > 1e13:
            warn(f'Correlation matrix may be ill-conditioned, condition number: {condn:1.2e}', RuntimeWarning)
        if condn > 0.1 / np.finfo(float).eps:
            warn(f'Correlation matrix condition number exceed machine precision {condn:1.2e}', RuntimeWarning)

        # Plot correlation matrix
        if plot:
            plt.matshow(cor,vmin=-1,vmax=1,cmap='RdBu')
            plt.colorbar()
            plt.show()

        # SVD analysis from gvar
        svd = gv.dataset.svd_diagnosis(ysamples)
        svd.plot_ratio(show=plot)
        
        if verbose:
            print(f'Condition number of the correlation matrix is {condn:1.2e}')
            print(f'Advised svd cut parameter for is {svd.svdcut}')


        return svd.svdcut

    def compute_covariance(self, ysamples, diag=False, block=False, scale=False, ysamples_full=None, shrink=False, svdcut=None):
        tmp = np.hstack([ysamples[k] for k in ysamples])
        factor = (tmp.shape[0]-1) if self.info.binsize is not None else (1./tmp.shape[0])
        fullcov = np.cov(tmp,rowvar=False,bias=True) * factor

        if diag: # uncorrelated data
            cov = np.sqrt(np.diag(fullcov))

        elif block: # blocked covariance
            cov = np.asarray(scipy.linalg.block_diag(
                *[np.cov(ysamples[k],rowvar=False,bias=True) * factor for k in ysamples]
            ))
        
        elif scale and ysamples_full is not None: # scaled with full covariance matrix
            tmp_full = np.hstack([ysamples_full[k] for k in ysamples_full])
            cov_full = np.cov(tmp_full,rowvar=False,bias=True) * (tmp_full.shape[0]-1)
            scale = np.sqrt(np.diag(fullcov)/np.diag(cov_full))
            cov = cov_full * np.outer(scale,scale)  # TODO: to be checked

            if shrink: # scale + shrinking
                cov = covariance_shrinking(tmp_full.mean(axis=0),cov,tmp_full.shape[0]-1)

        elif shrink: # shrinking
            cov = covariance_shrinking(tmp.mean(axis=0),fullcov,tmp.shape[0]-1)

        else: # full covariance
            cov = fullcov

        yout = gv.gvar(tmp.mean(axis=0),cov)

        if svdcut is not None:
            yout = gv.svd(yout, svdcut=svdcut)

        return yout
    
    # def format(self, trange=None, flatten=False, covariance=True, scale_covariance=False, shrink_covariance=False, smearing=None, polarization=None, alljk=False):  
    def format(self, trange=None, flatten=False, smearing=None, polarization=None, alljk=False, **covariance_kwargs):  
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
        # ============================= DATA SELECTION =============================
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

        X          = self.data.timeslice[Trange].to_numpy()
        ydata      = self.data.loc[SMR,POL,:,Trange]
        ydata_full = self.io.ReadCorrelator(jkBin=1).loc[SMR,POL,:,Trange] # jkBin in this must be 1 or None?  
        # ==========================================================================

        # ======================== COMPUTE COVARIANCE MATRIX ========================
        ysamples      = {(smr,pol): ydata.loc[smr,pol,:,:].to_numpy()      for (smr,pol) in keys}
        ysamples_full = {(smr,pol): ydata_full.loc[smr,pol,:,:].to_numpy() for (smr,pol) in keys} if covariance_kwargs.get('shrink') else None  
        yformatted = self.compute_covariance(
            ysamples,
            ysamples_full=ysamples_full,
            **covariance_kwargs
        )
        # ===========================================================================


        # In case flatten flag is active, reshape 
        if flatten: 
            x = np.concatenate([X for _ in keys])
            y = yformatted

            if alljk:
                yjk = np.hstack([ydata.loc[smr,pol,:,:].values for smr,pol in keys])

        else:
            x,y,yjk = {},{},{}
            for i,(smr,pol) in enumerate(keys):
                x[(smr,pol)] = X
                y[(smr,pol)] = yformatted[(len(X)*i):(len(X)*(i+1))]
                yjk[(smr,pol)]  = ydata.loc[smr,pol,:,:].values
        
        return (x,y) if not alljk else (x,y,yjk)

    def EffectiveMass (self, trange=None, variant='log', mprior=None, verbose=False, **kwargs):
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

        nan = np.isnan(gv.mean(yfit))
        xfit = xfit[~nan]
        yfit = yfit[~nan]

        if mprior is None:
            aux = gv.mean(yfit).mean()
            pr = gv.gvar(aux,aux)
        else:
            pr = mprior

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data  = (xfit,yfit),
            fcn   = ConstantModel,
            prior = {'const': pr}
        )
        MEFF = fit.p['const']

        print(MEFF)

        if verbose:
            print(fit)

        return (x,meff), MEFF, pr

    def EffectiveCoeff(self, trange=None, variant='log', aprior=None, mprior=None, verbose=False, **kwargs):
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


def plot_effective_coeffs(trange,X,AEFF,aeff,Apr,MEFF,meff,mpr,Aknob=2.):
    NROWS = len(np.unique([k[1] for k in X]))+1
    
    # Effective coeffs -------------------------------------------------------------------------------
    for i,(k,x) in enumerate(X.items()):
        axj = plt.subplot(NROWS,3,i+1)

        # Points inside the fit
        iok = [j for j,x in enumerate(x) if x>=min(trange) and x<=max(trange)]
        xplot = x[iok]
        yplot = gv.mean(aeff[k][iok])
        yerr  = gv.sdev(aeff[k][iok])
        axj.scatter(xplot,yplot, marker='o', s=15 ,facecolors='none', edgecolors=f'C{i}')
        axj.errorbar(xplot,yplot, yerr=yerr,fmt=',',color=f'C{i}', capsize=2)

        # Points outside the fit
        iout = [j for j,x in enumerate(x) if x<min(trange) or x>max(trange)]
        xplot = x[iout]
        yplot = gv.mean(aeff[k][iout])
        yerr  = gv.sdev(aeff[k][iout])
        axj.scatter(xplot,yplot, marker='s', s=15 ,facecolors='none', edgecolors=f'C{i}' ,alpha=0.2)
        axj.errorbar(xplot,yplot, yerr=yerr, fmt=',',color=f'C{i}', capsize=2,alpha=0.2)
        
        # Prior
        axj.axhspan(Apr[k].mean+Apr[k].sdev,Apr[k].mean-Apr[k].sdev,color=f'C{i}',alpha=0.14)

        # Final result
        axj.axhspan(AEFF[k].mean+AEFF[k].sdev,AEFF[k].mean-AEFF[k].sdev,color=f'C{i}',alpha=0.3)#,label=AEFF[k])

        # Delimiter for timespan
        axj.axvline(min(trange),color='gray',linestyle=':')
        axj.axvline(max(trange),color='gray',linestyle=':')

        # # Delimiter on y axis
        # dispersion = abs(gv.mean(aeff[k][iok]) - gv.mean(aeff[k][iok]).mean()).mean()
        dispersion = Aknob*Apr[k].sdev
        axj.set_ylim(ymax=AEFF[k].mean+dispersion,ymin=AEFF[k].mean-dispersion)

        axj.grid(alpha=0.2)
        axj.title.set_text(k)
        axj.legend()

        axj.set_xlabel(r'$t/a$')
        if i%3==0:
            axj.set_ylabel(r'$\mathcal{C}(t)e^{tM_{eff}}$')

    # Effective mass -------------------------------------------------------------------------------
    ax = plt.subplot(NROWS,1,NROWS)
    for i,(k,y) in enumerate(meff.items()):
        mar = 's' if k[1]=='Unpol' else '^' if k[1]=='Par' else 'v'
        col = f'C{i}'

        # Plot point for the fit considered range
        iok = [j for j,x in enumerate(X[k]) if x>=min(trange) and x<=max(trange)]
        xplot = X[k][iok]
        yplot = gv.mean(y[iok])
        yerr  = gv.sdev(y[iok])
        ax.scatter(xplot+(-0.1 + 0.1*i), yplot, marker=mar, s=15 ,facecolors='none', edgecolors=col, label=f'({k[0]},{k[1]})')
        ax.errorbar(xplot+(-0.1 + 0.1*i),yplot, yerr=yerr, fmt=',' ,color=col, capsize=2)

        # Plot point outside considered range
        iout = [j for j,x in enumerate(X[k]) if x<min(trange) or x>max(trange)]
        xplot = X[k][iout]
        yplot = gv.mean(y[iout])
        yerr  = gv.sdev(y[iout])
        ax.scatter(xplot+0.1*i,yplot, marker=mar, s=15, facecolors='none', edgecolors=f'C{i}',color=col,alpha=0.2)
        ax.errorbar(xplot+0.1*i,yplot,yerr=yerr,fmt='.',color=f'C{i}',alpha=0.2, capsize=2)

    # Prior
    ax.axhspan(mpr.mean+mpr.sdev,mpr.mean-mpr.sdev,color=f'gray',alpha=0.2)

    # Final result
    ax.axhspan(MEFF.mean+MEFF.sdev,MEFF.mean-MEFF.sdev,color=f'gray',alpha=0.3)#,label=MEFF)

    # Delimiter for the timerange
    ax.axvline(min(trange),color='gray',linestyle=':')
    ax.axvline(max(trange),color='gray',linestyle=':')

    # Limit on y
    dispersion = (gv.mean(aeff[k]) - gv.mean(aeff[k]).mean()).mean()
    # ax.set_ylim(ymax=AEFF[k].mean+1.5*dispersion,ymin=AEFF[k].mean-1.5*dispersion)
    ax.set_ylim(ymin=1.45,ymax=1.6)

    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    ax.set_ylim(ymax=MEFF.mean+0.03,ymin=MEFF.mean-0.03)
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$M_{eff}(t)$')






def test():
    ens      = 'Coarse-1'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex'
    meson    = 'Dst'
    mom      = '200'
    binsize  = 11

    io   = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr = Correlator(io,jkBin=binsize)


    x,y = corr.format(
        trange   = (10,19),
        smearing = ['d-d','1S-1S','d-1S'],
        flatten  = True,
        **dict(scale=False,shrink=True, svdcut=None)
    )
    print(gv.evalcorr(y))

    x,y = corr.format(
        trange   = (10,19),
        smearing = ['d-d','1S-1S','d-1S'],
        flatten  = True,
        **dict(scale=False,shrink=True, svdcut=0.05)
    )
    print(gv.evalcorr(y))



if __name__ == "__main__":
    test()
