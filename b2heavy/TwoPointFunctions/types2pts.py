import os
import h5py
import sys

import matplotlib.pyplot as plt
import itertools
import numpy  as np
import xarray as xr
import gvar   as gv
import lsqfit
import scipy
import tomllib

import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from tqdm import tqdm

from ..     import FnalHISQMetadata
from .utils import jkCorr, compute_covariance, ConstantModel, ConstantDictModel, correlation_diagnostics

# ConstantModel, ConstantFunc

ENSEMBLE_LIST = ['MediumCoarse', 'Coarse-2', 'Coarse-1', 'Coarse-Phys', 'Fine-1', 'SuperFine', 'Fine-Phys'] 
MESON_LIST    = ["Dsst", "Bs", "B", "D", "Ds", "Dst", "Dsst", "K", "pi", "Bc", "Bst", "Bsst"]
MOMENTUM_LIST = ["000", "100", "200", "300", "400", "110", "211", "222"]



def effective_mass(corrd,variant='cosh'):
    tmp = {}
    for k,y in corrd.items():
        if variant=='cosh':
            tmp[k] = []
            for it in range(len(y)):
                try:
                    m = np.arccosh( (y[(it+1)%len(y)]+y[(it-1)%len(y)])/y[it]/2 )
                except ZeroDivisionError:
                    m = np.nan
                tmp[k].append(m)
            tmp[k] = np.array(tmp[k])

        elif variant=='log':
            tmp[k] = np.log( np.roll(y,2)/np.roll(y,-2) ) / 4
    return tmp

def effective_amplitude(corrd,e0,time,variant='cosh',Nt=None):
    tmp = {}
    for k,y in corrd.items():
        if variant=='cosh':
            tmp[k] = y / (np.exp(-e0*time) + np.exp(-e0*(Nt-time)))
        elif variant=='log':
            tmp[k] = y / np.exp(-e0*time)
    return tmp

def smeared_correlator(corrd,e0,time):
    yeff = {}
    for k,y in corrd.items():
        expt = np.exp(- e0 * time)
        tmp = y / expt
        yeff[k] = 0.25*expt * (tmp + 2*np.roll(tmp,-1) + np.roll(tmp,-2))
    return yeff



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
    def __init__(self, io:CorrelatorIO, smearing=None, polarization=None, jkBin=None, **kwargs):
        self.io           = io
        self.info         = io.info
        self.info.binsize = jkBin
        self.binsize      = jkBin

        data = io.ReadCorrelator(jkBin=jkBin,**kwargs)
        self.timeslice = np.asarray(data.timeslice.to_numpy()) 
        self.Nt   = 2*self.timeslice.size

        self.smr  = smearing     if smearing     is not None else data.smearing.values
        self.pol  = polarization if polarization is not None else data.polarization.values
        self.keys = sorted(itertools.product(self.smr,self.pol))
        self.data = data.loc[self.smr,self.pol,:,:]

    def __str__(self):
        return f'{self.info}'
    
    def format(self, trange=None, smearing=None, polarization=None, flatten=False, alljk=False, **cov_kwargs):
        '''
            Returns a pytree with formatted data whithin the time-range with given covariance properties.
        '''
        # Select smearings
        if smearing is None:
            smr = self.smr
        else:
            if set(smearing)<=set(self.smr):
                smr = smearing
            else:
                raise KeyError(f'Smearing list {smearing=} contain at least one item which is not contained in {self.smr=}')

        # Select polarizations
        if polarization is None:
            pol = self.pol
        else:
            if set(polarization)<=set(self.pol):
                pol = polarization
            else:
                raise KeyError(f'Polarization list {polarization=} contain at least one item which is not contained in {self.pol=}')

        # Select data in the timeranges
        it = self.timeslice if trange is None else np.arange(min(trange),max(trange)+1)
        xdata = self.timeslice[it]

        # Compute un-jk data
        if cov_kwargs.get('scale'):
            # ydata_full = self.io.ReadCorrelator(jkBin=1).loc[smr,pol,:,it]
            ydata_full = self.io.ReadCorrelator(jkBin=1)
            ally = {(s,p): jnp.asarray(ydata_full.loc[s,p,:,it].to_numpy()) for s,p in self.keys}
        else:
            ally = None

        ydata = compute_covariance(
            {(s,p): jnp.asarray(self.data.loc[s,p,:,it]) for s,p in self.keys},
            ysamples_full = ally,
            **cov_kwargs
        )

        if alljk:
            yjk = {(s,p): self.data.loc[s,p,:,it].to_numpy() for s,p in self.keys}


        if flatten:
            ydata = np.concatenate([ydata[k] for k in self.keys])

            if alljk:
                yjk = np.hstack([yjk[k] for k in self.keys])

        return (xdata, ydata) if not alljk else (xdata,ydata,yjk)

    def meff(self, trange=None, prior=None, verbose=False, plottable=False, variant='cosh', **cov_kwargs):
        xdata,ydata = self.format(trange=None, flatten=False, **cov_kwargs)

        # Compute effective masses
        meffs = effective_mass(ydata, variant=variant)

        # Slice in time-range and filter nans
        it = self.timeslice[1:-1] if trange is None else np.arange(min(trange),max(trange)+1)+1
        yfit = np.concatenate([meffs[k][it] for k in meffs])
        yfit = yfit[~np.isnan(gv.mean(yfit))]

        # Compute smeared_correlator
        aver = np.average(gv.mean(yfit),weights=1./gv.sdev(yfit)**2)
        ydata = smeared_correlator(ydata,xdata,aver)

        # Compute effective masses
        meffs = effective_mass(ydata, variant=variant)
        yfit = np.concatenate([meffs[k][it] for k in meffs])
        yfit = yfit[~np.isnan(gv.mean(yfit))]


        # Fit effective masses
        xfit = np.arange(len(yfit))

        aver = np.average(gv.mean(yfit),weights=1./gv.sdev(yfit)**2)
        disp = np.abs(gv.mean(yfit) - aver).mean()

        mpr = prior if prior is not None else gv.gvar(aver,disp)
        fit = lsqfit.nonlinear_fit(
            data   = (xfit,yfit),
            fcn    = ConstantModel,
            prior  = {'const': mpr },
        )
        Meff = fit.p['const']
        e0 = Meff.mean

        if verbose:
            print(fit)


        # Build effective coeffs ==============================================================
        aplt = effective_amplitude(ydata,xdata,e0,variant=variant,Nt=self.Nt)
        
        apr, xfit, yfit = {},{},{}
        for k in aplt:
            aver   = np.average(gv.mean(aplt[k][it]),weights=1./gv.sdev(aplt[k][it])**2)
            disp   = np.abs(gv.mean(aplt[k][it]) - aver).mean()

            apr[k] = gv.gvar(aver,disp)            
            xfit[k] = xdata[it]
            yfit[k] = aplt[k][it]

        afit = lsqfit.nonlinear_fit(
            data  = (xfit,yfit),
            fcn   = ConstantDictModel,
            prior = apr
        )
        Aeff = afit.p
        if verbose:
            print(afit)
        # # =====================================================================================

        if plottable:
            return {k: xdata for k in ydata}, Aeff, aplt, apr, Meff, meffs, mpr
        else:
            return Meff,Aeff

    def chiexp_meff(self, trange, variant, pvalue=False, Nmc=5000, **cov_kwargs):
        args = self.meff(trange=trange,plottable=True,variant=variant, **cov_kwargs)

        # Slice effective mass in trange and flatten
        meffs = [x[min(trange):(max(trange)+1)] for x in args[-2].values()]
        meffs = np.concatenate(meffs)

        # Filter nans
        iok = ~np.isnan(gv.mean(meffs))
        meffs = meffs[iok]

        # Evaluate cov
        fitcov = gv.evalcov(meffs)

        # Compute inverse covarince and chi2
        w = np.linalg.inv(fitcov)

        res = gv.mean(meffs) - args[4].mean
        chi2 = res.T @ w @ res

        # Compute projector
        s = w.sum(axis=1)
        proj = w - np.outer(s,s)/w.sum()
        
        
        # Compute full covariance matrix
        args = self.meff(trange=trange,plottable=True,variant=variant)
        meffs = [x[min(trange):(max(trange)+1)] for x in args[-2].values()]
        meffs = np.concatenate(meffs)
        iokn = ~np.isnan(gv.mean(meffs))
        assert np.all(iok==iokn)

        meffs = meffs[iokn]
        cov = gv.evalcov(meffs)

        # Compute chi2 expected
        chiexp = np.trace(proj*cov)



        if not pvalue:
            return chi2,chiexp 
        else:
            # Compute p-value
            l,evec = np.linalg.eig(cov)

            if np.any(np.real(l))<0:
                mask = np.real(l)>1e-12
                Csqroot = evec[:,mask] @ np.diag(np.sqrt(l[mask])) @ evec[:,mask].T
            else:
                Csqroot= evec @ np.diag(np.sqrt(l)) @ evec.T

            numat = Csqroot @ proj @ Csqroot
            l,_ = np.linalg.eig(numat)
            ls = l[l.real>=1e-14].real
            
            p = 0
            for _ in range(Nmc):
                ri = np.random.normal(0.,1.,len(ls))
                p += 1. if (ls.T @ (ri**2) - chi2)>=0 else 0.
            p /= Nmc
            
            return chi2, chiexp, p

    def tmax(self, threshold=0.25):
        xdata,ydata = self.format()
        rel = np.vstack([abs(gv.sdev(y)/gv.mean(y)) for y in ydata.values()]).mean(axis=0)
        Tmax = max([t for t,r in enumerate(rel) if r<=threshold])
        return Tmax


def plot_effective_coeffs(trange,X,AEFF,aeff,Apr,MEFF,meff,mpr,Aknob=10.):
    NROWS = len(np.unique([k[1] for k in X]))+1
    
    # Effective coeffs -------------------------------------------------------------------------------
    for i,(k,x) in enumerate(X.items()):
        axj = plt.subplot(NROWS,3,i+1)

        # Points inside the fit
        iok = np.array([min(trange)<=x<=max(trange) for x in X[k]])
        xplot = x[iok]
        yplot = gv.mean(aeff[k][iok])
        yerr  = gv.sdev(aeff[k][iok])
        axj.errorbar(xplot,yplot, yerr=yerr,fmt=',',color=f'C{i}', capsize=2)
        axj.scatter(xplot,yplot, marker='o', s=15 ,c=f'white', edgecolors=f'C{i}', label=k)

        # Points outside the fit
        xplot = x[~iok]
        yplot = gv.mean(aeff[k][~iok])
        yerr  = gv.sdev(aeff[k][~iok])
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
        # dispersion = Aknob*Apr[k].sdev
        # axj.set_ylim(ymax=AEFF[k].mean+dispersion,ymin=AEFF[k].mean-dispersion)
        axj.set_ylim(ymin=Apr[k].mean-Aknob*Apr[k].sdev,ymax=Apr[k].mean+Aknob*Apr[k].sdev)


        axj.grid(alpha=0.2)
        # axj.title.set_text(k)
        axj.legend()

        axj.set_xlabel(r'$t/a$')
        if i%3==0:
            axj.set_ylabel(r'$\frac{\mathcal{C}(t)}{e^{-M_{eff}t} + e^{-M_{eff}(N_t-t)}}$')

    # Effective mass -------------------------------------------------------------------------------
    ax = plt.subplot(NROWS,1,NROWS)
    for i,(k,y) in enumerate(meff.items()):
        mar = 's' if k[1]=='Unpol' else '^' if k[1]=='Par' else 'v'
        col = f'C{i}'

        # Plot point for the fit considered range
        iok = np.array([min(trange)<=x<=max(trange) for x in X[k][1:-1]])
        xplot = X[k][1:-1][iok]

        yplot = gv.mean(y[1:-1][iok])
        yerr  = gv.sdev(y[1:-1][iok])
        ax.errorbar(xplot+(-0.1 + 0.1*i),yplot, yerr=yerr, fmt=',' ,color=col, capsize=3)
        ax.scatter(xplot+(-0.1 + 0.1*i), yplot, marker=mar, s=20 ,facecolors='white', edgecolors=col, label=f'({k[0]},{k[1]})')

        # Plot point outside considered range
        xplot = X[k][1:-1][~iok]
        yplot = gv.mean(y[1:-1][~iok])
        yerr  = gv.sdev(y[1:-1][~iok])
        ax.errorbar(xplot+0.1*i,yplot,yerr=yerr,fmt=',',color=f'C{i}',alpha=0.2, capsize=3)
        ax.scatter(xplot+0.1*i,yplot, marker=mar, s=15, facecolors='white', edgecolors=f'C{i}',color=col,alpha=0.2)

    # # Prior
    ax.axhspan(mpr.mean+mpr.sdev,mpr.mean-mpr.sdev,color=f'gray',alpha=0.2)

    # # Final result
    ax.axhspan(MEFF.mean+MEFF.sdev,MEFF.mean-MEFF.sdev,color=f'gray',alpha=0.3)#,label=MEFF)

    # # Delimiter for the timerange
    ax.axvline(min(trange),color='gray',linestyle=':')
    ax.axvline(max(trange),color='gray',linestyle=':')

    # Limit on y
    # dispersion = (gv.mean(aeff[k]) - gv.mean(aeff[k]).mean()).mean()
    # ax.set_ylim(ymax=AEFF[k].mean+1.5*dispersion,ymin=AEFF[k].mean-1.5*dispersion)
    ax.set_ylim(ymin=mpr.mean-Aknob*mpr.sdev,ymax=mpr.mean+Aknob*mpr.sdev)

    ax.legend(loc='upper right')
    ax.grid(alpha=0.2)
    ax.set_xlabel(r'$t/a$')
    ax.set_ylabel(r'$M_{eff}(t)$')

def eff_coeffs(FLAG):
    ens      = 'Coarse-1'
    mes      = 'D'
    mom      = '000'
    binsize  = 13
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    smlist   = ['1S-1S','d-d','d-1S'] 


    io   = CorrelatorIO(ens,mes,mom,PathToDataDir=data_dir)
    stag = Correlator(
        io       = io,
        jkBin    = binsize,
        smearing = smlist
    )

    print(f'{mom = } {stag.tmax(threshold=0.3) = }')

    # ----------------------------- Correlation analysis -----------------------------
    if FLAG==1:
        tmin = 9
        tmax = 19
        trange = (tmin,tmax)
        tmaxe = stag.tmax(threshold=0.3)
        xdata,ydata,yjk = stag.format(trange=(tmin,tmax),alljk=True,flatten=True)

        print(f'-------- Correlation diagnostics whithin ({tmin},{tmax}) --------- ')
        print(f'{tmaxe = }')
        correlation_diagnostics(yjk,plot=True)
        plt.show()
    # --------------------------------------------------------------------------------


    # ----------------------------- Effective corr analysis -----------------------------
    elif FLAG==2:
        tmin = 13
        tmax = 20
        tmaxe = stag.tmax(threshold=0.3)
        trange = (tmin,tmax)

        print(tmaxe,trange)

        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True,
            shrink = True,
            cutsvd = 0.01,
        )

        args = stag.meff(trange=trange,verbose=True,plottable=True,variant='cosh', **cov_specs)
        chi2, chiexp,p = stag.chiexp_meff(trange=trange,variant='cosh',pvalue=True,**cov_specs)

        print('--------------------------')
        print(f'{ens = }, {mes = }, {mom = }')
        print(tmaxe,trange)
        print(f'meff = {args[-3]}')
        print(f'{trange = }')
        print(f'{chi2/chiexp = :.2f}')
        print(f'{p = }')
        print('--------------------------')
        print(f'{chi2/chiexp:.2f} {p:.2f}')
        plot_effective_coeffs(trange, *args)
        plt.show()
    # --------------------------------------------------------------------------------

    # ------------------------- Effective range analysis -----------------------------
    elif FLAG==3:
        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True,
            shrink = True,
            cutsvd = 0.01,
        )

        tmins = np.arange(9,15)
        # tmaxs = np.arange(19,22)
        tmaxs = [21]
        tranges = itertools.product(tmins,tmaxs)

        MEFF = []
        AEFF = []
        TIC = []
        Tranges = []
        for trange in tqdm(tranges):
            ydata, Aeff, aplt, apr, Meff, meffs, mpr = stag.meff(
                trange    = trange,
                plottable = True,
                variant   = 'cosh', 
                **cov_specs
            )

            try:
                chi2, chiexp = stag.chiexp_meff(
                    trange  = trange,
                    variant = 'cosh',
                    **cov_specs
                )
            except:
                continue

            MEFF.append(Meff)
            AEFF.append([Aeff[k] for k in stag.keys])
            TIC.append(chi2-2*chiexp)
            Tranges.append(trange)

        MEFF = np.array(MEFF)
        AEFF = np.array(AEFF)
        TIC = np.array(TIC)

        tic = np.exp(-TIC/2)
        tic = tic/tic.sum()

        tmin = np.array([min(t) for t in Tranges])
        tmax = np.array([max(t) for t in Tranges])

        print(f't_min = {np.sum(tmin*tic)}')
        print(f't_min = {np.sum(tmax*tic)}')
        print(f'E_eff = {np.sum(MEFF*tic)}')
        for i,k in enumerate(stag.keys):
            print(f'A_eff[{k}] = {np.sum(AEFF[:,i]*tic)}')
    # --------------------------------------------------------------------------------

def global_eff_coeffs(ens, mes, trange, chiexp=True, config_file='/Users/pietro/code/software/B2heavy/routines/2pts_fit_config.toml', **cov_specs):
    with open(config_file,'rb') as f:
        config = tomllib.load(f)

    mom_list = config['data'][ens]['mom_list']

    Meffs = {}
    xdata_meff = {}
    data_meff = {}
    fullcov_data = {}
    prior_meff = {}
    for mom in mom_list:
        io   = CorrelatorIO(ens,mes,mom,PathToDataDir=config['data'][ens]['data_dir'])
        stag = Correlator(
            io       = io,
            jkBin    = config['data'][ens]['binsize'],
            smearing = config['fit'][ens][mes]['smlist']
        )

        trange_eff = config['fit'][ens][mes]['mom'][mom]['trange_eff']
        # Compute effective masses and ampls. for priors calc
        Xdict, Aeff, aplt, apr, Meff, meffs, mpr = stag.meff(
            trange    = trange_eff,
            plottable = True,
            **cov_specs
        )
        _,_,_,_,_,fully,_ = stag.meff(trange=trange, plottable=True)

        # Slice and create fit data
        tmp,tmp1 = [],[]
        for k in meffs:
            inan = np.isnan(gv.mean(meffs[k][min(trange):(max(trange)+1)])) 
            tmp.append(
                meffs[k][min(trange):(max(trange)+1)][~inan]
            )
            tmp1.append(
                fully[k][min(trange):(max(trange)+1)][~inan]
            )
        data_meff[mom] = np.concatenate(tmp)
        xdata_meff[mom] = np.arange(len(data_meff[mom]))
        prior_meff[mom] = gv.gvar(Meff.mean,Meff.sdev)
        Meffs[mom] = Meff

        # Compute effective masses and ampls. for priors calc
        fullcov_data[mom] = np.concatenate(tmp1)

    # Perform fit
    fit = lsqfit.nonlinear_fit(
        data  = (xdata_meff,data_meff),
        fcn   = ConstantDictModel,
        prior = prior_meff
    )

    if not chiexp:
        return fit
    else:
        # Compute chiexp and p value -------------------------------------
        meffs = np.concatenate([data_meff[k] for k in mom_list])
        fitcov = gv.evalcov(meffs)
        w = np.linalg.inv(fitcov)
        res = np.concatenate(
            [gv.mean(ConstantDictModel(xdata_meff,fit.p)[k]) - gv.mean(data_meff[k]) for k in mom_list]
        )
        chi2 = res.T @ w @ res

        fullc = np.concatenate([fullcov_data[k] for k in mom_list])
        cov = gv.evalcov(fullc)

        # Compute jacobian
        jac = []
        for i,_ in enumerate(mom_list):
            tmp = []
            for j,p in enumerate(mom_list):
                tmp.append(
                    np.full_like(data_meff[p], 1. if i==j else 0.)
                )
            tmp = np.concatenate(tmp)
            jac.append(tmp)
        jac = np.array(jac).T

        # Calculate expected chi2
        Hmat = jac.T @ w @ jac
        Hinv = np.diag(1/np.diag(Hmat))
        wg = w @ jac
        proj = np.asfarray(w - wg @ Hinv @ wg.T)
        chiexp = np.trace(proj @ cov)

        l,evec = np.linalg.eig(cov)
        
        if np.any(np.real(l))<0:
            mask = np.real(l)>1e-12
            Csqroot = evec[:,mask] @ np.diag(np.sqrt(l[mask])) @ evec[:,mask].T
        else:
            Csqroot= evec @ np.diag(np.sqrt(l)) @ evec.T

        numat = Csqroot @ proj @ Csqroot

        l,_ = np.linalg.eig(numat)
        ls = l[l.real>=1e-14].real

        p = 0
        for _ in range(50000):
            ri = np.random.normal(0.,1.,len(ls))
            p += 1. if (ls.T @ (ri**2) - chi2)>=0 else 0.
        p /= 50000

        return fit, chi2, chiexp, p





def main():
    ens = 'Coarse-1'
    mes = 'Dst'
    mom = '000'

    io   = CorrelatorIO(ens,mes,mom,PathToDataDir="/Users/pietro/code/data_analysis/BtoD/Alex")
    print(io.CorrFile)

    # cov_specs = dict(
    #     shrink = True,
    #     scale  = True,
    #     svd = 0.05
    # )

    # trange = (10,19)
    # fit,chi2,chiexp,p = global_eff_coeffs(ensemble,meson,trange)

    # print(fit)
    # print(f'{chi2/chiexp = }')
    # print(f'{p = }')