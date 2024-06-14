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
from .utils import jkCorr, compute_covariance, ConstantModel, ConstantDictModel, correlation_diagnostics, p_value

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


def smeared_correlator(corrd,time,e0):
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

        Data are read from a typical hdf5 archive, e.g. `l3264f211b600m00507m0507m628-HISQscript.hdf5` that can be provided by the additional argument `PathToFile`.
        
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
                    tmp = 0.5 * ( 
                        RAW[corr_name][:,:T] + 
                        np.flip(np.roll(RAW[corr_name], -1, axis=1), axis=1)[:,:T] 
                    )
                    tmp = jkCorr(tmp,bsize=(0 if jkBin is None else jkBin))
                    corrs.append(tmp)

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
            for pol in POL:
                tmp = []
                for sm in sorted(SMR):
                    sm1,sm2 = sm.split('-')
                    s2 = f'{sm2}-{sm1}'
                    if sm1!=sm2 and (s2,pol) in PROCESSED:
                        if sm not in tmp:
                            tmp.append(s2)
                            PROCESSED[sm,pol] = 0.5*(
                                PROCESSED[sm,pol] + PROCESSED[s2,pol]
                            )                            
                            del PROCESSED[s2,pol]

            for rm_smr in tmp:
                SMR.remove(rm_smr)

            # change name 1S-d ---> d-1S
            for pol in POL:
                try:
                    PROCESSED['d-1S',pol] = PROCESSED['1S-d',pol]
                    del PROCESSED['1S-d',pol]
                    if '1S-d' in SMR:
                        SMR.remove('1S-d')
                        SMR.append('d-1S')
                except KeyError:
                    pass
        
        # Stack data --------------------------------------------------------------
        # DATA = np.array([[jkCorr(PROCESSED[smr,pol],bsize=(jkBin if jkBin is not None else 0)) for pol in POL] for smr in SMR])    
        DATA = np.array([[PROCESSED[smr,pol] for pol in POL] for smr in SMR])    
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


    def collect(self,smearing=['1S','d'], **kwargs):
        data      = self.ReadCorrelator(**kwargs)
        data_full = self.ReadCorrelator(jkBin=1)

        self.Nt    = 2*data.timeslice.size
        self.tdata = np.arange(self.Nt//2) 

        mes = self.info.meson
        mom = self.info.momentum

        self.data = {}
        self.data_full = {}
        for sm in data.smearing.values:
            for pol in data.polarization.values:
                s1,s2 = sm.split('-')
                if not (s1 in smearing and s2 in smearing):
                    continue
                tag = f'[{mes}]->[{mes}].{mom}.{s1}.{s2}.{pol}'
                self.data[tag]      = data.loc[sm,pol].values
                self.data_full[tag] = data_full.loc[sm,pol].values
        
        return


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


    def meff(self, trange=None, prior=None, verbose=False, plottable=False, variant='cosh', pvalue=False, **cov_kwargs):
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

        # calculate FScorrected p value
        if pvalue:
            chi_pr = 0.
            for k,_pr in fit.prior.items():
                dr = ((gv.mean(_pr) - fit.pmean[k])/gv.sdev(_pr))**2
                chi_pr += dr.sum()
            chi2red = fit.chi2 - chi_pr
            ndof  = len(fit.y) - 1
            aux = Correlator(io=self.io,jkBin=0)
            nconf = len(aux.data.jkbin)
            pvalue = p_value(chi2red,nconf,ndof)

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
            if pvalue:
                return Meff,Aeff,pvalue
            else:
                return Meff,Aeff


    def chiexp_meff(self, trange, variant, pvalue=False, Nmc=50000, **cov_kwargs):
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


    def tmax(self, threshold=0.25, criterion=max):
        xdata,ydata = self.format()
        # rel = np.vstack([abs(gv.sdev(y)/gv.mean(y)) for y in ydata.values()]).mean(axis=0)
        # Tmax = criterion([t for t,r in enumerate(rel) if r<=threshold])

        for Tmax in range(len(xdata)):
            err = min([abs(gv.sdev(y[Tmax])/gv.mean(y[Tmax])) for y in ydata.values()])
            if err>threshold:
                break

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


def find_eps_cut(corr:Correlator,trange,tol=1e+05,default=1E-12,**cov_specs):
    x,y, data = corr.format(trange=trange,flatten=True,alljk=True,**cov_specs)    

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
    ens = 'Coarse-Phys'
    mes = 'Dst'
    mom = '000'

    path = '/Users/pietro/code/data_analysis/BtoD/Alex/Ensembles/FnalHISQ/a0.12/'

    io  = CorrelatorIO(ens,mes,mom,PathToDataDir='/Users/pietro/code/data_analysis/BtoD/Alex/')
    stag = Correlator(io,jkBin=19,smearing=['d-1S'])