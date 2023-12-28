import numpy  as np
import autograd 
import gvar   as gv
import pandas as pd
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import lsqfit
import os
import sys
import h5py
from autograd import numpy as np

import FnalHISQMetadata



ENSEMBLE_LIST = ['MediumCoarse', 'Coarse-2', 'Coarse-1', 'Coarse-Phys', 'Fine-1', 'SuperFine', 'Fine-Phys', 'SuperFine'] 
MESON_LIST    = ["Dsst", "Bs", "D", "Ds", "Dst", "Dsst", "K", "pi", "Bc", "Bst", "Bsst"]
MOMENTUM_LIST = ["000", "100", "200", "300", "400", "110", "211", "222"]

def ConstantModel(x,p):
    return np.array([p['const']]*len(x))

def PeriodicExpDecay(Nt):
    return lambda t,E,Z: Z * ( np.exp(-E*t) + np.exp(-E*(Nt-t)) ) 

def NplusN2ptModel(Nstates,Nt,sm,pol):
    sm1,sm2 = sm.split('-')
    mix = sm1!=sm2

    def _(t,p):
        E0, E1 = p['E'][0], p['E'][0]+np.exp(p['E'][1])
        Z0 = np.exp(p[f'Z_{sm1}_{pol}'][0]) * np.exp(p[f'Z_{sm2}_{pol}'][0])
        Z1 = np.exp(p[f'Z_{sm1}_{pol}'][1]) * np.exp(p[f'Z_{sm2}_{pol}'][1])
        ans = PeriodicExpDecay(Nt)(t,E0,Z0) + (-1)**(t+1)*PeriodicExpDecay(Nt)(t,E1,Z1)

        for i in range(2,2*Nstates):
            E = (E0 if i%2==0 else E1) + np.exp(p['E'][i])
            Z = p[f'Z_{sm if mix else sm1}_{pol}'][i-2 if mix else i]**2
            ans += PeriodicExpDecay(Nt)(t,E,Z) * (-1)**(i*(t+1))
        return ans
    return _


def jkCorr(C,bsize):
    Nb  = C.shape[0]//bsize
    rem = C.shape[0]%bsize==0

    Cjk = []
    for k in range(Nb):
        Cjk.append(C[:Nb*bsize,:].reshape((Nb,bsize,C.shape[1])).mean(axis=1)[np.delete(range(Nb),k),:].mean(axis=0))

    return np.array(Cjk)



class CorrelatorInfo:
    """
        This is a basic structure to contain information for 2pt functions for the following given attributes:
        - ensemble
        - meson
        - momentum
    """
    # def __new__(cls, _name:str, _ens:str, _mes:str, _mom:str):
    #     if _ens not in ENSEMBLE_LIST: 
    #         raise KeyError(f'{_ens} is not a valid ensemble.')
    #     if _mes not in MESON_LIST: # Check whether meson is recognized
    #         raise KeyError(f'{_mes} is not a valid meson.')
    #     if _mom not in MOMENTUM_LIST: # Check whether momentum is recognized
    #         raise KeyError(f'{_mom} is not a valid momentum.')
    #     return super().__new__(cls)
    #     # if _file is not None and not os.path.isfile(_file): # Check whether input file exist
    #     #     raise RuntimeError(f'{_file} has not been found.')

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
        if sms is None:
            sms = self.mData['sms']
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
            clist = []
            for lss in sms:
                for hss in sms:
                    if meson=='K':
                        newfiles = [f'P5_P5_{lss}_{hss}_m{self.mData["msStr"]}_m{self.mData["mlStr"]}_p{mom}']
                        continue
                    elif meson=='pi':
                        newfiles = [f'P5_P5_{lss}_{hss}_m{self.mData["mlStr"]}_m{self.mData["mlStr"]}_p{mom}']
                    else:
                        newfiles = [f'{ch}_{lss}_{hss}_{heavy}_{light}_p{mom}' for ch in channel]
                    
                    aux = [f for f in newfiles if f in data]
                    
                    smr_string = f'{self.checkSmear(lss)}_{self.checkSmear(hss)}'
                    clist.append((aux,smr_string)) if aux else 1

            return clist

    def ReadCorrelator(self,sms=None):
        CorrList = self.CorrFileList(sms=sms)
        T  = self.mData['T']//2

        with h5py.File(self.CorrFile,'r') as d:
            data = d['data']

            allCorrs = []
            allSmear = []
            allPolar = set()
            for clist,smr in CorrList:
                aux = []
                for corr in clist:
                    Ci_t = np.array(data[corr])
                    Ci_t = 0.5*( # Fold correlators
                        Ci_t[:,:T] + 
                        np.flip( np.roll(Ci_t,-1,axis=1) ,axis=1)[:,:T]
                    ) 
                    aux.append(Ci_t)
                
                if len(aux)==1: # scalar correlator
                    allSmear.append(smr)
                    allPolar.add('Unpol')
                    allCorrs.append(aux)
                
                if len(aux)==3: # vector correlator
                    crPar = aux[0]*0.0
                    crBot = aux[0]*0.0

                    mom = self.info.momentum
                    if mom[1:]=='00' and mom!='000':
                        nBot = 0
                        for j,p in enumerate(mom):
                            pj = int(p)
                            crPar += aux[j]*pj
                            if pj==0:
                                crBot += aux[j]
                                nBot += 1
                        crPar /= np.array([int(x) for x in mom]).sum()

                        allSmear.append(smr)
                        allPolar.add(('Par','Bot'))
                        allCorrs.append([crPar,crBot])

                    else:
                        allSmear.append(smr)
                        allPolar.add('Unpol')
                        allCorrs.append([np.array(aux).mean(axis=0)])

        DATA = np.array(allCorrs)
        xd = xr.DataArray(
            DATA,
            dims   = ['smearing','polarization','jkbin','timeslice'],
            coords = [allSmear,list(*allPolar),np.arange(DATA.shape[-2]),np.arange(DATA.shape[-1])],
            attrs  = {
                'ensemble': self.info.ensemble,
                'meson'   : self.info.meson,
                'momenutm': mom,
                'list'    : CorrList,
            },
            name   = self.info.name
        )
        return xd

class Correlator:
    """
        This is the basic structure for a correlator at fixed momentum. It contains data for all possible smearings and polarization.
    """
    def __init__(self, io:CorrelatorIO, sms=None):
        self.data = io.ReadCorrelator(sms=sms)
        self.Nt   = 2*self.data.timeslice.size
        self.info = io.info  

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
        
        return 0

    def __str__(self):
        return self.info

    def format(self, trange=None, flatten=False, covariance=True, jknorm=True, smearing=None, polarization=None, alldata=False):
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

        keys = sorted([(smr,pol) for smr in SMR for pol in POL])

        if not alldata:
            DATA = self.data.loc[SMR,POL,:,:]
        else:
            DATA = self.alldata.loc[SMR,POL,:,:]


        # Slice data in the [Tmin,Tmax] window
        if trange==None:
            Trange = DATA.timeslice
            ydata  = DATA.loc[SMR,POL,:,:]
        else:
            Trange = DATA.timeslice.isin(np.arange(min(trange),max(trange)+1))
            ydata  = DATA.loc[SMR,POL,:,Trange]

        X = DATA.timeslice[Trange].to_numpy()

        # Flatten data temporarily to calculate covariances
        yaux = np.hstack([ydata.loc[smr,pol] for (smr,pol) in keys])
        if covariance:
            Y = gv.gvar(
                yaux.mean(axis=0),
                np.cov(yaux,rowvar=False,bias=False) * ((yaux.shape[0]-1) if jknorm else 1./yaux.shape[0])
            )
        else:
            Y = gv.gvar(yaux.mean(axis=0),yaux.std(axis=0))


        # In case flatten flag is active, return 
        if flatten: 
            xfit = np.concatenate([X for _ in keys])
            yfit = Y
        else:
            xfit,yfit = {},{}
            for i,(smr,pol) in enumerate(keys):
                xfit[(smr,pol)] = X
                yfit[(smr,pol)] = Y[(len(X)*i):(len(X)*(i+1))]

        return (xfit,yfit)

    def EffectiveMass(self, variant='log', trange=None, flatten=True, **kwargs):
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
            meff, dict
                Dictionary with the same keys of `x` and `y` containing the effective mass dep. on time.
            MEFF
                Dictionary with the same keys of `x` and `y` containing the values of the fitted effective mass. Returned only id `fit` flag is set to `True`.

        """

        # Compute effective mass according to the definition
        (x,y) = self.format(flatten=False, **kwargs)
        meff = {}
        for k,c in y.items():
            if variant=='log':
                meff[k] = np.log(c/np.roll(c,-2))/2 
            elif variant=='arccosh':
                meff[k] = np.arccosh(((np.roll(c,-2) + np.roll(c,2))/c/2))/2

        # Slice data in the time range
        iifit = np.arange(0,self.Nt//2) if trange is None else np.arange(min(trange),max(trange)+1)
        xfit, yfit = {}, {}
        for k,m in meff.items():
            xfit[k] = x[k][iifit]
            yfit[k] = m[iifit]

        pr = gv.mean(list(yfit.values()))

        # Perform fit
        fit = lsqfit.nonlinear_fit(
            data=(xfit,yfit),
            fcn=ConstantModel,
            prior={'const': gv.gvar(pr,pr)}
        )
        MEFF = fit.p['const']

        # (x,y) = self.format(flatten=False, **kwargs)

        # meff = {}
        # for k,v in y.items():
        #     if variant=='log':
        #         meff[k] = np.log(v/np.roll(v,-2))[:-2]/2        
        #     elif variant=='arccosh':
        #         meff[k] = np.arccosh(((np.roll(v,-2) + np.roll(v,2))/v/2)[2:-2])/2

        # if fit: 
        #     if not flatten:
        #         MEFF = {}
        #         for k,v in meff.items():
        #             Trange = np.arange(len(v)) if trange==None else np.arange(trange[0],min(trange[1],len(v)))
        #             fit = lsqfit.nonlinear_fit(
        #                 data=(x[k][Trange],v[Trange]),
        #                 fcn=ConstantModel,
        #                 prior={'const': gv.gvar(gv.mean(v[Trange]).mean(),gv.mean(v[Trange]).mean())}
        #             )
        #             MEFF[k] = fit.p['const']
        #     else:
        #         x = np.concatenate([v for k,v in x.items()])
        #         y = np.concatenate([v for k,v in y.items()])
        #         fit = lsqfit.nonlinear_fit(
        #             data=(x,y),
        #             fcn=ConstantModel,
        #             prior={'const': gv.gvar(gv.mean(v[trange]).mean(),gv.mean(v[trange]).mean())}
        #         )
        #         MEFF['all'] = fit.p['const']
        #     return x,y,meff,MEFF
        # else:
        #     return x,y,meff

        return meff, MEFF

    def EffectiveCoeff(self, variant='log', fitrange=None, **kwargs):
        (x,y,meff,MEFF) = self.EffectiveMass(fit=True, fitrange=fitrange, variant=variant)

        Aeff = {}
        for k,v in y.items():
            if variant=='log':
                Aeff[k] = np.exp(MEFF[k].mean*x[k])*v
            elif variant=='arccosh':
                Aeff[k] = v/(np.exp(-MEFF[k].mean*x[k]) + np.exp(-MEFF[k].mean*(self.Nt-x[k])))

        AEFF = {}
        for k,v in Aeff.items():
            trange = np.arange(fitrange[0],min(fitrange[1],len(v)))
            pr = {'const': gv.gvar(gv.mean(v[trange]).mean(),gv.mean(v[trange]).mean())}
            fit = lsqfit.nonlinear_fit(
                data=(x[k][trange],v[trange]),
                fcn=ConstantModel,
                prior=pr
            )
            AEFF[k] = fit.p['const']

        return x,y,meff,Aeff,MEFF,AEFF

class CorrFitter:
    """
        This class takes care of fitting correlators.
    """
    def __init__(self, corr:Correlator, smearing=None, polarization=None, **kwargs):
        SMR = smearing     if not smearing==None     else corr.data.smearing.to_numpy()
        POL = polarization if not polarization==None else corr.data.polarization.to_numpy()

        self.smearing = SMR
        self.polarization = POL 
        self.keys = sorted([(smr,pol) for smr in SMR for pol in POL])

        self.Nt = corr.Nt

        self.fits = {}


    def fit(self, corr:Correlator, Nstates, trange, priors,  p0=None, maxit=50000, svdcut=1e-12, debug=False, **kwargs):
        """
            This function perform a fit to 2pts oscillating functions.

            Arguments
            ---------
                corr: TwoPointFunctions.Correlator
                    An object of type `Correlator`. The function `format` will be called. It must be the same `Correlator` object used to instantiate the `CorrFitter` class.
                priors: dict
                    A dictionary containing all priors as `gv.gvar` variables.
                trange: tuple
                    A tuple containing `tmin` and `tmax` for timeslice selection.
                Nstates: int
                    Number of states to fit (fundamental+excited). E.g.: `Nstates=2` corresponds to '2+2 fits', i.e. 2 fundamental states (physical and oscillating) and 2 excited (physical and oscillating)   
                
        """
        if (Nstates,trange) in self.fits:
            print(f'Fit for {(Nstates,trange)} has already been performed. Returning...')
            return

        xfit,yfit = corr.format(trange=trange, smearing=self.smearing, polarization=self.polarization, **kwargs)
        yfit = np.concatenate([yfit[k] for k in self.keys])

        def model(x,p):
            return np.concatenate(
                [  NplusN2ptModel(Nstates,self.Nt,sm,pol)(x[(sm,pol)],p)   for sm,pol in self.keys ]
            )

        fit = lsqfit.nonlinear_fit(
            data   = (xfit,yfit),
            fcn    = model,
            prior  = priors,
            p0     = p0 if p0 is not None else gv.mean(priors),
            maxit  = maxit,
            svdcut = svdcut,
            debug  = debug,
        )

        self.fits[(Nstates,trange)] = fit
        return
    
    def chiexp(self, Nstates, trange):
        key = (Nstates,trange) 
        if key not in self.fits:
            raise KeyError(f'Fit for Nstates={Nstates} and trange={trange} has not been performed yet.')
        else:
            fit = self.fits[key]

        mix = lambda k: True if k.startswith('Z') and  '-' in k else False
        

        # ------------------- Compute gradient with AD ------------------ #
        keys   = [(k,exc) for exc in range(2*Nstates) for k in sorted(fit.p.keys()) if not (mix(k) and exc<2)]
        params = np.transpose([fit.p[k][exc-2 if mix(k) else exc] for (k,exc) in keys])

        lent = max(trange)+1-min(trange)

        def _model(t,E0,Z0a,Z0b, E1,Z1a,Z1b, *high):
            C_t =  np.exp(Z0a)*np.exp(Z0b) * ( np.exp(-E0*t) + np.exp(-E0*(self.Nt-t)))
            C_t += np.exp(Z1a)*np.exp(Z1b) * ( np.exp(-E1*t) + np.exp(-E1*(self.Nt-t)))
            for n in range(2*Nstates-2):
                C_t += high[2*n+1]**2 * ( np.exp(-high[2*n]*t) + np.exp(-high[2*n]*(self.Nt-t)))
            return C_t
    
        # Calculate gradient with pain and sorrow
        grad = np.zeros((lent*len(self.keys), len(params)))
        for ik,(sm,pol) in enumerate(self.keys):
            sm1,sm2 = sm.split('-')
            mix = sm1!=sm2

            # Collect parameters for fund_phys and fund_osc
            iipars = [
                keys.index(('E',0)),keys.index((f'Z_{sm1}_{pol}',0)),keys.index((f'Z_{sm2}_{pol}',0)),
                keys.index(('E',1)),keys.index((f'Z_{sm1}_{pol}',1)),keys.index((f'Z_{sm2}_{pol}',1))
            ]
            # Collect parameters for excited states
            for exc in range(2,2*Nstates):
                iipars.extend([keys.index(('E',exc)),keys.index((f'Z_{sm if mix else sm1}_{pol}',exc))])
            pars = params[iipars]
            park = [keys[i] for i in iipars]

            for I,k in enumerate(keys):
                delta = [ii for ii,el in enumerate(park) if el==k]
                for _i in delta:
                    grad[(ik*lent):((ik+1)*lent),I] += [autograd.grad(_model,_i+1)(tt,*gv.mean(pars)) for tt in fit.x[(sm,pol)]]
        # ----------------------------------------------------------------- #

        # Calculate enlarged covariance matrix as in (D.9) of (2209.14188v2) #
        COV  = gv.evalcov(fit.y)
        width = np.concatenate([fit.psdev[k] for k in sorted(fit.p.keys())])
        diagpr = np.diag(1./np.array(width)**2)
        COV = np.block(
            [[COV,np.zeros((COV.shape[0],len(width)))],[np.zeros((len(width),COV.shape[1])),diagpr]]
        )

        # Enlarge also gradient matrix with identity matrix 
        GRAD = np.hstack((grad.T, np.eye(len(params)) )).T

        W = np.linalg.inv(COV)
        Wg = W @ GRAD
        Hmat = GRAD.T @ W @ GRAD
        Hinv = np.linalg.inv(Hmat)
        proj = W - Wg @ Hinv @ Wg.T
        chi_exp = np.trace(proj.dot(COV))

        return chi_exp

    def weight(self, Nstates, trange, IC='AIC'):
        key = (Nstates,trange) 
        if key not in self.fits:
            raise KeyError(f'Fit for Nstates={Nstates} and trange={trange} has not been performed yet.')
        else:
            fit = self.fits[key]

        # Calculate augmented chi2
        chi2aug = fit.chi2
        for k,v in fit.prior.items():
            for i,p in enumerate(v):
                chi2aug += (fit.pmean[k][i]-p.mean)**2/(p.sdev**2)

        Npars = 6 + 6*(Nstates-1)
        Ncut  = self.Nt/2 - (max(trange)+1-min(trange))

        if IC=='AIC':
            ic = chi2aug + 2*Npars + 2*Ncut
        elif IC=='TIC':
            ic = fit.chi2 + 2*self.chiexp(Nstates,trange)
        
        return np.exp(-ic/2)





# def NplusN2ptModel(Nexc,Nt,sm,pol):
#     sm1,sm2 = sm.split('-')
#     mix = sm1!=sm2

#     if Nexc==2:
#         return lambda t,p: \
#                         PeriodicExpDecay(Nt)(t,p['E'][0]                                        , np.exp(p[f'Z_{sm1}_{pol}'][0]) * np.exp(p[f'Z_{sm2}_{pol}'][0])) + \
#             (-1)**(t+1)*PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][1])                    , np.exp(p[f'Z_{sm1}_{pol}'][1]) * np.exp(p[f'Z_{sm2}_{pol}'][1])) + \
#                         PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][2])                    , p[f'Z_{sm if mix else sm1}_{pol}'][2]**2) + \
#             (-1)**(t+1)*PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][1]) + np.exp(p['E'][3]), p[f'Z_{sm if mix else sm1}_{pol}'][3]**2)
#     else:
#         pass
