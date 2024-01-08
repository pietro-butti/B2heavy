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
from scipy.special import gammaincc as gammaQ
from scipy.special import gammaln as gammaLn
from scipy.special import gamma as gamma
from scipy.linalg import eigh

import FnalHISQMetadata



ENSEMBLE_LIST = ['MediumCoarse', 'Coarse-2', 'Coarse-1', 'Coarse-Phys', 'Fine-1', 'SuperFine', 'Fine-Phys'] 
MESON_LIST    = ["Dsst", "Bs", "D", "Ds", "Dst", "Dsst", "K", "pi", "Bc", "Bst", "Bsst"]
MOMENTUM_LIST = ["000", "100", "200", "300", "400", "110", "211", "222"]
mPhys  = { 
    'B'   : 5.280,
    'Bs'  : 5.366,
    'D'   : 1.870,
    'Dst' : 2.010,
    'Ds'  : 1.968,
    'Dsst': 2.112,
    'K'   : 0.496,
    'pi'  : 0.135 
}

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


        if _mes=='Dsst':
            self.GeV = self.mData['mDs']/mPhys['Ds'] * self.mData['aSpc']/self.mData['hbarc']


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
                    
                    smr_string = f'{self.checkSmear(lss)}-{self.checkSmear(hss)}'
                    clist.append((aux,smr_string)) if aux else 1

            return clist

    def ReadCorrelator(self,sms=None, CrossSmearing=False, verbose=False):
        CorrList = self.CorrFileList(sms=sms)
        T  = self.mData['T']//2

        with h5py.File(self.CorrFile,'r') as d: # open file and read
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
                        allPolar.add('Par')
                        allPolar.add('Bot')
                        allCorrs.append([crPar,crBot])

                    else:
                        allSmear.append(smr)
                        allPolar.add('Unpol')
                        allCorrs.append([np.array(aux).mean(axis=0)])
        
        if not CrossSmearing: # average cross-smearing correlator
            # detect which ones are the cross smearing and put them in a dict e.g. cross['1S_d']=[2,4]  
            cross = {}
            ssm = [sorted(s.split('-')) for s in allSmear]
            sst = [f'{s[0]}_{s[1]}' for s in ssm]
            for v in set(sst):
                if sst.count(v)>1:
                    cross[v] = [i for i,x in enumerate(sst) if x==v]

            # Cycle over all the crossed-smearing, average them and put the average in the first one, delete the second
            for v in cross.values():
                allCorrs[v[1]][0][:] = 0.5*(allCorrs[v[0]][0][:]+allCorrs[v[1]][0])[:]  
                allCorrs.pop(v[0])
                allSmear.pop(v[0])

        DATA = np.array(allCorrs)
        xd = xr.DataArray(
            DATA,
            dims   = ['smearing','polarization','jkbin','timeslice'],
            coords = [allSmear,list(allPolar),np.arange(DATA.shape[-2]),np.arange(DATA.shape[-1])],
            attrs  = {
                'ensemble': self.info.ensemble,
                'meson'   : self.info.meson,
                'momenutm': mom,
                'list'    : CorrList,
            },
            name   = self.info.name
        )

        if verbose:
            print(f'Correlators for ensemble {self.info.ensemble} for meson {self.info.meson} at fixed mom {mom} read from {CorrList}')

        return xd

class Correlator:
    """
        This is the basic structure for a correlator at fixed momentum. It contains data for all possible smearings and polarization.
    """
    def __init__(self, io:CorrelatorIO, sms=None, **kwargs):
        self.io   = io
        self.data = io.ReadCorrelator(sms=sms,**kwargs)
        self.Nt   = 2*self.data.timeslice.size
        self.info = io.info  

    def __str__(self):
        return self.info

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


    def format(self, trange=None, flatten=False, covariance=True, smearing=None, polarization=None, alldata=False):
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
                np.cov(yaux,rowvar=False,bias=False) * ((yaux.shape[0]-1) if self.info.binsize is not None else 1./yaux.shape[0])
            )
            # do I have to divide for the number of configurations? FIXME

        else:
            Y = gv.gvar(yaux.mean(axis=0),yaux.std(axis=0) * np.sqrt((yaux.shape[0]-1)) if self.info.binsize is not None else 1./yaux.shape[0])
            # do I have to divide for the number of configurations? FIXME

        # In case flatten flag is active, reshape 
        if flatten: 
            xfit = np.concatenate([X for _ in keys])
            yfit = Y
        else:
            xfit,yfit = {},{}
            for i,(smr,pol) in enumerate(keys):
                xfit[(smr,pol)] = X
                yfit[(smr,pol)] = Y[(len(X)*i):(len(X)*(i+1))]

        return (xfit,yfit)

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

class CorrFitter:
    """
        This class takes care of fitting correlators.
    """
    def __init__(self, corr:Correlator, smearing=None, polarization=None):
        self.corr = corr

        SMR = smearing     if not smearing==None     else list(corr.data.smearing.values)
        POL = polarization if not polarization==None else list(corr.data.polarization.values)

        self.smearing = SMR
        self.polarization = POL 
        self.keys = sorted([(smr,pol) for smr in SMR for pol in POL])

        self.Nt = corr.Nt

        self.fits = {}

    def set_priors_phys(self,Nstates, Meff=None):
        at_rest = self.corr.info.momentum=='000'
        p2 = sum([(2*np.pi*float(px)/self.corr.io.mData['L'])**2 for px in self.corr.info.momentum])


        dScale = self.corr.io.mData['mDs']/mPhys['Ds']
        bScale = self.corr.io.mData['mBs']/mPhys['Bs']
        aGeV   = self.corr.io.mData['aSpc']/self.corr.io.mData['hbarc']
        
        # SET ENERGIES ---------------------------------------------------------------------------
        priors = dict()
        if self.corr.info.meson=='Dsst':
            E = []
            if Meff is None:
                E.append(gv.gvar(mPhys[self.corr.info.meson],0.6/12) * dScale * aGeV) # fundamental physical state
            else:
                E.append(gv.gvar(Meff.mean,0.140*aGeV.mean))

            E.append(np.log(gv.gvar(0.350*bScale , 0.6/3*dScale) * aGeV)) # fundamental oscillating state

            for n in range(2,2*Nstates):
                E.append(np.log(gv.gvar(0.6,0.6)*dScale*aGeV)) # excited states
            priors['E'] = E

        if not at_rest: # non-zero mom
            priors['E'][0] = gv.gvar(
                np.sqrt(priors['E'][0].mean**2 + p2),
                np.sqrt(4*(priors['E'][0].mean**2)*(priors['E'][0].sdev**2) + p2*self.corr.io.mData['alphaS'])/(2*priors['E'][0].mean)
            )
        
        # SET OVERLAP FACTORS --------------------------------------------------------------------
        apEr = self.corr.io.mData["alphaS"]*p2

        lbl = set() # infer all smearing labels
        for smr,pol in self.keys:
            sm1,sm2 = smr.split('-')
            lbl.add(f'{sm1}_{pol}' if sm1==sm2 else f'{smr}_{pol}')         

        for smr in list(lbl):
            if len(self.corr.info.meson) > 2:
                if self.corr.info.meson[-2:] == 'st':
                    if 'd' in smr:
                        val  = (np.log((self.corr.io.mData['aSpc']*5.3 + 0.54)*self.corr.io.mData['aSpc'])).mean
                        err  = (np.sqrt((np.exp(val)*val*0.2)**2 + apEr**2)/np.exp(val))
                        bVal = gv.gvar(val, err)      

                        if smr.split('_')[-1]=='Par':
                            oVal = gv.gvar(-5.5,2.0)
                        else:
                            oVal = gv.gvar(-3.,1.5)  

                    else:
                        bVal = gv.gvar(-2.0, 2.0) - 0.5*np.log(priors['E'][0].mean)
                        oVal = gv.gvar(-9.5, 2.0) - 2.0*np.log(self.corr.io.mData['aSpc'].mean)                                                        
            else:
                bVal = gv.gvar(-1.5, 1.5)
                oVal = gv.gvar(-2.5, 1.5)  

            baseGv = gv.gvar( 0.0, 1.2) if '1S' in smr else gv.gvar(bVal.mean, bVal.sdev)
            osciGv = gv.gvar(-1.2, 1.2) if '1S' in smr else gv.gvar(oVal.mean, oVal.sdev)
            highGv = gv.gvar( 0.5, 1.5)

            priors[f'Z_{smr}'] = []

            if smr.split('-'): # mixed smearing case
                priors[f'Z_{smr}'].append(gv.gvar(baseGv.mean, baseGv.sdev)),
                priors[f'Z_{smr}'].append(gv.gvar(osciGv.mean, osciGv.sdev))
            
            for n in range(2,2*Nstates):
                priors[f'Z_{smr}'].append(gv.gvar(highGv.mean, highGv.sdev*(n//2)))

        return priors        

    def set_priors_eff(self, Nstates, Meff, Aeff):
        """
            This function returns a dictionary with priors, according to the following rules:

            Energies
            ---------
            - The prior for the **fundamental physical state at zero momentum** is set with the effective mass, while its width is set following the paper to $140\text{ MeV}$
            $$
                \tilde{E}_0(\mathbf{p}=0) = M_\text{eff} \pm 140\text{ MeV}
            $$ 
            - At **non-zero momentum** 
            $$
                \tilde{E}_0(\mathbf{p}\neq0) = \sqrt{M_\text{eff}^2+\mathbf{p}^2} \pm (140\text{ MeV} + \#\alpha_sa^2\mathbf{p}^2)
            $$ 
            - The energy priors for the energy differences are $\Delta \tilde E = 500(200) \text{ MeV}$

            Overlap factors
            ---------------
            We will use the following rules
            - For non-mixed smearing, physical fund. state, we use the plateaux of $\mathcal{C}(t)e^{tM_\text{eff}}=A_\text{eff}(t)$
            $$
            \tilde Z_0^{(\text{sm},\text{pol})} = \sqrt{A_\text{eff}} \, \pm (0.5 \,\mathtt{if\,[p=0]\,else}\,?)
            $$
            - For non-mixed smearing, oscillating fund. state, we use
            $$
            \tilde Z_1^{(\text{sm},\text{pol})} = \sqrt{A_\text{eff}} \, \pm (1.2 \,\mathtt{if\,[p=0]\,else}\,2.0)
            $$
            - For mixed smearing, excited states, we always use
            $$
            \tilde Z_{n\geq2}^{(\text{sm},\text{pol})} = 0.5 \pm 1.5
            $$
        """
        zeromom = self.corr.info.momentum=='000'
        ap2 = sum([2.*np.pi*float(px)/self.corr.io.mData['L']**2 for px in self.corr.info.momentum])

        priors = {}

        # Set energies ---------------------------------------------------------------------------
        priors['E'] = []    

        Erest = Meff.mean
        Eval   = np.sqrt(Erest**2+ap2**2)
        den = 0.140*self.corr.io.GeV.mean
        Ewidth = np.sqrt(4*Eval**2*den**2 + self.corr.io.mData['alphaS']*ap2)/2/Eval
        priors['E'].append(gv.gvar(Eval,Ewidth)) # fundamental physical state

        for n in range(1,2*Nstates): # excited states
            delta = gv.gvar('0.5(2)')*self.corr.io.GeV.mean
            priors['E'].append(np.log(delta))


        # Set overlap factors -------------------------------------------------------------------
        for (sm,pol),v in Aeff.items():
            smr = np.unique(sm.split('-'))
            if len(smr)==1:
                priors[f'Z_{smr[0]}_{pol}'] = [
                    np.log(gv.gvar(v.mean, 0.5 if zeromom else 2*self.corr.io.mData['alphaS']*ap2)), # 0.5 + 2*\alpha_s*p^2
                    np.log(gv.gvar(v.mean, 1.2 if zeromom else 2.0)), # ?
                ]
                for n in range(Nstates-1):
                    priors[f'Z_{smr[0]}_{pol}'].append(gv.gvar(0.5,1.5))
                    priors[f'Z_{smr[0]}_{pol}'].append(gv.gvar(0.5,1.5))
            else:
                priors[f'Z_{sm}_{pol}'] = []
                for n in range(1,Nstates):
                    priors[f'Z_{sm}_{pol}'].append(gv.gvar(0.5,1.5))
                    priors[f'Z_{sm}_{pol}'].append(gv.gvar(0.5,1.5))

        return priors

    def fit(self, Nstates, trange, priors,  p0=None, maxit=50000, svdcut=1e-12, debug=False, verbose=False, **kwargs):
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

        xfit,yfit = self.corr.format(trange=trange, smearing=self.smearing, polarization=self.polarization, **kwargs)
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
            debug  = debug
        )
        self.fits[(Nstates,trange)] = fit

        if verbose:
            print(fit)

        return

    # chiexp makes use of the fact that self.fits[...] is the ouptut of nonlinear_fit. This is not optimal
    # one should save only necessary informations FIXME
    def chiexp(self, Nstates, trange, covariance=True):
        """
            chiexp(self, Nstates, trange)

            Compute expected chi-square given by Eq. (2.13) of 2209.14188v2 for the case of fit with priors, i.e. ``augmenting'' the covariance matrix as in Eq. (D9)
        """
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
        if covariance:
            (x,y) = self.corr.format(trange=trange, flatten=True, covariance=True, smearing=self.smearing, polarization=self.polarization)
            COV  = gv.evalcov(y)
        else:
            COV = gv.evalcov(fit.y)
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

    def chiprior(self, Nstates, trange):
        """
            chiprior(self, Nstates, trange)

            Computes the prior chi-square of the fit.
        """
        f = self.fits[(Nstates,trange)]
        chi2pr = 0.
        for k,v in f.prior.items():
            for i,p in enumerate(v):
                chi2pr += (f.pmean[k][i]-p.mean)**2/(p.sdev**2)
        
        return chi2pr

    def weight(self, Nstates, trange, IC='AIC'):
        key = (Nstates,trange) 
        if key not in self.fits:
            raise KeyError(f'Fit for Nstates={Nstates} and trange={trange} has not been performed yet.')
        else:
            fit = self.fits[key]

        Ncut  = self.Nt/2 - (max(trange)+1-min(trange))
        Npars = 2*Nstates + 2*Nstates*2 + (2*Nstates-2) # this is for 2 smearing FIXME

        if IC=='AIC':
            ic = fit.chi2 + 2*Npars + 2*Ncut
        
        return np.exp(-ic/2)

    def fDist2(self, chi2, nConf, dof):
        """
            Compute the cumulative chi-square distribution with `dof` degrees of freedom, corrected for finite sample size `nConf` integrated from `chi2` to infinity.

            Its definition is taken from formula (B1) of PRD 93 113016.
        """
        return (np.exp(gammaLn(nConf/2.0) - gammaLn(dof/2.0) - gammaLn((nConf-dof)/2.0) + (-dof/2.0)*np.log(nConf) + ((dof-2.0)/2.0)*np.log(chi2) + (-nConf/2.0)*np.log(1.0+chi2/nConf)))

    def dpVal(self, chi2, nConf, dof):
        """
            Compute the p-value of the fit given by formula (B5) of of PRD 93 113016.
        """

        if dof <= 0:
            return 0.0

        if chi2 < 3.0*dof+2.0:
            nSteps = int(100.0*chi2/(2.0*np.sqrt(2.0*dof)))

            if nSteps < 20:
                nSteps = 20;
            eps = chi2/nSteps;
            sum = 0.0
            tChi2 = 0.5*eps

            while tChi2 < chi2:
                sum = sum + self.fDist2(tChi2, nConf, dof)
                tChi2 = tChi2 + eps

            if sum*eps < 0.9:
                return 1.0 - sum*eps;

        nSteps = 0;
        eps = (2.0*np.sqrt(2.0*dof))/100.0;

        y = self.fDist2(chi2, nConf, dof);

        sum = 0.0
        tChi2 = chi2+0.5*eps

        while True:
            tChi2 = tChi2 + eps
            nSteps = nSteps + 1
            z = self.fDist2(tChi2, nConf, dof);

            if nSteps > 1000 or z < 0.0001*y:
                break;

            sum += z;

        return sum*eps

    def model_average(self, keylist=None, IC='AIC', par='E0'):
        """
            model_average(self, keylist=None, IC='AIC', par='E0')

            Perform the model average of the results corresponding to the keys in `keylist` according to Eq. (15) of arXiv:2008.01069v3. Reutrns also systematic error according to Eq. (18)
        """
        if keylist is None:
            keys = self.fits.keys()
        else:
            # Check all keys in `keylist` are calculated
            for k in keylist:
                if k not in self.fits:
                    raise KeyError(f'Fit for {k} has not been calculated.')
            keys = keylist

        Ws = [self.weight(n,t,IC=IC) for (n,t) in keys] # Compute and gather all the weights
        Ws = Ws/sum(Ws)

        if par=='E0':
            P  = [self.fits[k].p['E'][0] for k in keys]
        else:
            pass # FIXME 
    
        syst = np.sqrt(gv.mean(sum(Ws*P*P) - (sum(Ws*P))**2))

        return sum(Ws*P), syst

    def GEVP(self, t0=None, smlist=['d','1S'], polarization=None, order=-1):
        """
            GEVP(self, t0=None, smlist=['d','1S'], polarization=None, order=-1)

            Extract the ground state mass by solving the GEVP. 
            
            Methodology
            -----------
            Given a list of smearing types, e.g. `['d','1S']`, a correlator matrix is built as
            $$ C(t) = 
                \begin{pmatrix}
                    (\text{d},\text{d}) & (\text{d},\text{1S}) \\
                    (\text{1S},\text{d}) & (\text{1S},\text{1S})
                \end{pmatrix}
            $$
            then the following GEVP is solved
            $$
                C(t)\cdot\vec{v}_{(n)}(t,t_0) = \lambda_{(n)}(t,t_0) C(t_0)\vec v_{(n)}(t,t_0)
            $$
            for each t. The ground state (effective) mass is extracted as
            $$
                E_n^{\text{eff}}(t,t_0) \equiv \log\frac{\lambda_{(n)}(t,t_0)}{\lambda_{(n)}(t+1,t_0)} = E_n + \mathcal{O}(e^{-t\Delta E_n })
            $$
            The asymptotic behavior is only insured in the regime $t\in[t_0,2t_0]$ as suggested [here](https://inis.iaea.org/collection/NCLCollectionStore/_Public/40/054/40054766.pdf).

            Arguments
            ---------
                t0: int
                    The reference time used to solve the GEVP. We remind that the convergence is ensured for $t\geq t_0/2$. The default value is set to `None`, in this case, when the GEVP is solved for each `t`, it is set to `t0=t//2`
                smlist: list
                    A list of smearing. From that the matrix is built.
                polarization: list
                    List of polarization to be considered
                order: int
                    Index of the eigenvalue to be returned, i.e. `sorted(eval)[order]` (`eval` being the first output of `eigh`).

        """
        # Construct the list with element of smearing matrix
        if len(np.unique(smlist))<2:
            raise KeyError('At least two types of smearings must be provided')
        else:
            smr_flat = [f'{sm1}-{sm2}' for sm1 in np.unique(smlist) for sm2 in np.unique(smlist)]

        Nt  = self.corr.data.timeslice.size
        Nc  = self.corr.data.jkbin.size
        Nsm = int(np.sqrt(len(smr_flat)))

        # Select polarization to be considered
        if polarization is not None:
            if False not in np.isin(polarization,self.data.polarization):
                POL = sorted(polarization)
            else:
                raise ValueError(f'Polarization list {polarization} contains at least one item which is not contained in data.polarization')
        else:
            POL = sorted(self.corr.data.polarization.values)

        # Reshape data and construct corr matrix
        C = np.moveaxis(self.corr.data.loc[smr_flat,POL,:,:].values,0,-1).reshape(Nc,Nt,Nsm,Nsm)

        # Iterate over time and solve the GEVP conf by conf
        eigs = []
        for jk in range(Nc):
            aux = []
            for t in range(Nt):
                eigval, eigvec = eigh(C[jk,t,:,:],b=C[jk,t0 if t0 is not None else t//2,:,:])
                aux.append(sorted(eigval)[order]) # sort eigenvalue and select the `order`-th
            eigs.append(aux)
        eigs = np.array(eigs)

        # Calculate effective mass
        aux = np.log(eigs/np.roll(eigs,-1,axis=1))
        Eeff = gv.gvar(
            aux.mean(axis=0),
            np.cov(aux,rowvar=False) * (eigs.shape[0]-1  if self.corr.info.binsize is not None else 1.)
        )

        return Eeff[:-1]

    def GEVPmass(self, trange=None, covariance=True, chiexp=True, pr=None, verbose=False, **kwargs):
        # Set time range
        if trange is None:
            iok = np.arange(self.corr.data.timeslice.size-1)
        else:
            iok = [i for i,t in enumerate(self.corr.data.timeslice) if t<=max(trange) and t>=min(trange)]

        # Call GEVP 
        meff = self.GEVP(**kwargs)[iok]

        # Set prior
        if pr is None:
            prior = gv.gvar(gv.mean(meff).mean(),gv.mean(meff).mean())
        else:
            prior = pr

        if covariance:
            # Perform a fully correlated fit
            fit = lsqfit.nonlinear_fit(
                data=(self.corr.data.timeslice[iok],meff),
                fcn=ConstantModel,
                prior={'const': prior}
            )

            if verbose:
                print(fit)

        else: # perform uncorrelated fit
            pass


        return fit.p['const']
    


def test():
    
    ens      = 'MediumCoarse'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex'
    meson    = 'Dsst'
    mom      = '100'
    binsize  = 13

    io   = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr = Correlator(io,CrossSmearing=True)
    corr.jack(binsize)
    
    trange = (14,20)
    smlist = ['1S-1S','d-d','d-1S']
    (X,meff,aeff), MEFF,AEFF, mpr, apr = corr.EffectiveCoeff(trange=trange,smearing=smlist)


    alex = {
        'E':          [
            gv.gvar('1.3716(38)'), 
            gv.gvar('-1.92(87)'), 
            gv.gvar('-1.0(1.0)'), 
            gv.gvar('-1.0(1.0)') 
        ] , 
        'Z_1S_Par':   [
            gv.gvar('0.0  (1.2)'), 
            gv.gvar('-1.2(1.2)'), 
            gv.gvar('0.5 (1.5)'), 
            gv.gvar('0.5 (1.5)') 
        ] , 
        'Z_1S_Bot':   [
            gv.gvar('0.0  (1.2)'), 
            gv.gvar('-1.2(1.2)'), 
            gv.gvar('0.5 (1.5)'), 
            gv.gvar('0.5 (1.5)') 
        ] , 
        'Z_d_Par':    [
            gv.gvar('-1.61 (33)'), 
            gv.gvar('-5.5(2.0)'), 
            gv.gvar('0.5 (1.5)'), 
            gv.gvar('0.5 (1.5)') 
        ] , 
        'Z_d_Bot':    [
            gv.gvar('-1.61 (33)'), 
            gv.gvar('-3.0(1.5)'), 
            gv.gvar('0.5 (1.5)'), 
            gv.gvar('0.5 (1.5)') 
        ] , 
        'Z_d-1S_Par': [
            gv.gvar('0.5  (1.7)'), 
            gv.gvar('0.5 (1.7)')                  
        ] , 
        'Z_d-1S_Bot': [
            gv.gvar('0.5  (1.7)'), 
            gv.gvar('0.5 (1.7)')
        ]                         
    }

    fitter = CorrFitter(corr,smearing=smlist)
    pr = fitter.set_priors_phys(2,Meff=MEFF)

    print(pr)


if __name__ == "__main__":
    test()






def old_NplusN2ptModel(Nexc,Nt,sm,pol):
    sm1,sm2 = sm.split('-')
    mix = sm1!=sm2

    if Nexc==2:
        return lambda t,p: \
                        PeriodicExpDecay(Nt)(t,p['E'][0]                                        , np.exp(p[f'Z_{sm1}_{pol}'][0]) * np.exp(p[f'Z_{sm2}_{pol}'][0])) + \
            (-1)**(t+1)*PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][1])                    , np.exp(p[f'Z_{sm1}_{pol}'][1]) * np.exp(p[f'Z_{sm2}_{pol}'][1])) + \
                        PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][2])                    , p[f'Z_{sm if mix else sm1}_{pol}'][2]**2) + \
            (-1)**(t+1)*PeriodicExpDecay(Nt)(t,p['E'][0] + np.exp(p['E'][1]) + np.exp(p['E'][3]), p[f'Z_{sm if mix else sm1}_{pol}'][3]**2)
    else:
        pass

