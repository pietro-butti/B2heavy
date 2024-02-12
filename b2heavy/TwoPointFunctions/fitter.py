import numpy as np
import gvar  as gv
import autograd 
import lsqfit
import sys
import os

import matplotlib.pyplot as plt

from scipy.linalg import eigh
from autograd import numpy as np

from .utils import load_toml, NplusN2ptModel, p_value, ConstantModel
from .types2pts import CorrelatorInfo, CorrelatorIO, Correlator


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

class CorrFitter:
    """
        This class takes care of fitting correlators.
    """
    def __init__(self, corr:Correlator, smearing=None, polarization=None, priors=None):
        self.corr = corr

        SMR = smearing     if not smearing==None     else list(corr.data.smearing.values)
        POL = polarization if not polarization==None else list(corr.data.polarization.values)

        self.smearing = SMR
        self.polarization = POL 
        self.keys = sorted([(smr,pol) for smr in SMR for pol in POL])

        self.Nt = corr.Nt

        self.prior_policy = priors

        self.fits = {}

    def set_priors_phys(self,Nstates, Meff=None, Aeff=None):
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
        at_rest = self.corr.info.momentum=='000'
        p2 = sum([(2*np.pi*float(px)/self.corr.io.mData['L'])**2 for px in self.corr.info.momentum])


        dScale = self.corr.io.mData['mDs']/mPhys['Ds']
        bScale = self.corr.io.mData['mBs']/mPhys['Bs']
        aGeV   = self.corr.io.mData['aSpc'].mean/self.corr.io.mData['hbarc']
        
        # SET ENERGIES ---------------------------------------------------------------------------
        if self.prior_policy==None:
            c = os.path.join(os.path.dirname(os.path.abspath(__file__)),'fitter_PriorPolicy.toml')
        else:
            c = self.prior_policy

        d = load_toml(c)[self.corr.info.meson]

        Scale    = dScale if d['scale']=='D' else bScale if d['scale']=='B' else 1.
        dE       = d['dE']      
        dG       = d['dG']       
        dE_E0err = d['dE_E0err']
        dE_E1err = d['dE_E1err']
        dE_Eierr = d['dE_Eierr'] if 'dE_Eierr' in d else 1.  # old was d['dE_E1err'] if 'dE_E1err'

        E = [None]*(2*Nstates)

        if Meff is None:
            E[0] = gv.gvar(mPhys[self.corr.info.meson],dE/dE_E0err) * Scale * aGeV # fundamental physical state
        else:
            # E[0] = gv.gvar(Meff.mean,dE/dE_E0err*Scale*aGeV)
            E[0] = Meff

        E[1] = np.log(gv.gvar(dG, dE/dE_E1err)*Scale*aGeV)

        for n in range(2,2*Nstates):
            E[n]= np.log(gv.gvar(dE,dE/dE_Eierr)*Scale*aGeV) # excited states

            if self.corr.info.meson=='Dst' and n==2:
                E1 = (2.630-mPhys['Dst'])*Scale
                E[2] = np.log(gv.gvar(
                    E1,
                    dE/dE_Eierr
                ) * aGeV )

        priors = {'E': E}

        if not at_rest and Meff is None: # non-zero mom
            priors['E'][0] = gv.gvar(
                np.sqrt(priors['E'][0].mean**2 + p2),
                np.sqrt(4*(priors['E'][0].mean**2)*(priors['E'][0].sdev**2) + (p2*self.corr.io.mData['alphaS'])**2)/(2*priors['E'][0].mean)
            )
        
        # SET OVERLAP FACTORS --------------------------------------------------------------------
        apEr = self.corr.io.mData["alphaS"]*p2

        lbl = [] # infer all smearing labels
        for smr,pol in self.keys:
            sm1,sm2 = smr.split('-')
            lbl.append(f'{sm1}_{pol}' if sm1==sm2 else f'{smr}_{pol}')         
        lbl = np.unique(lbl)

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

            if len(smr.split('-'))==1: # single smearing case
                priors[f'Z_{smr}'].append(gv.gvar(baseGv.mean, baseGv.sdev)),
                priors[f'Z_{smr}'].append(gv.gvar(osciGv.mean, osciGv.sdev))
            
            for n in range(2,2*Nstates):
                priors[f'Z_{smr}'].append(
                    gv.gvar(highGv.mean, highGv.sdev*(n//2)) if len(smr.split('-'))==1 else gv.gvar(0.5,1.7)
                )
        
        if Aeff is not None:
            for (sm,pol),v in Aeff.items():
                sm1,sm2 = sm.split('-')
                if sm1==sm2:
                    # priors[f'Z_{sm1}_{pol}'][0] = np.log(v)/2
                    v = np.log(v)/2
                    # priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(np.log(v.mean)/2,priors[f'Z_{sm1}_{pol}'][0].sdev)
                    priors[f'Z_{sm1}_{pol}'][0] = gv.gvar(v.mean,v.sdev*10)

        return priors        

    def fit(self, Nstates, trange, priors=None,  p0=None, maxit=50000, svdcut=1e-12, debug=False, verbose=False, pval=True, jkfit=False, override=False, **kwargs):
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
        if (Nstates,trange) in self.fits and not override:
            print(f'Fit for {(Nstates,trange)} has already been performed. Returning...')
            return

        xfit,yfit = self.corr.format(trange=trange, smearing=self.smearing, polarization=self.polarization, **kwargs)
        xfit = np.array([xfit[k] for k in self.keys])
        yfit = np.concatenate([yfit[k] for k in self.keys])
        fitdata = (xfit,gv.mean(yfit),gv.evalcov(yfit))


        def model(t,p):
            return np.concatenate(
                [NplusN2ptModel(Nstates,self.Nt,sm,pol)(t[i],p) for i,(sm,pol) in enumerate(self.keys)]
            )

        if verbose:
            print(f'---------- {Nstates}+{Nstates} fit in {trange} for mes: {self.corr.info.meson} of ens: {self.corr.info.ensemble} for mom: {self.corr.info.momentum} --------------')

        pr = priors if priors is not None else self.set_priors_phys(Nstates)
        p0 = p0 if p0 is not None else gv.mean(pr)
        fit = lsqfit.nonlinear_fit(
            data   = fitdata,
            fcn    = model,
            prior  = pr,
            p0     = p0,
            maxit  = maxit,
            svdcut = svdcut,
            debug  = debug
        )

        if pval: # Calculate p-value ------------------------------------------------
            Ndof = len(yfit)

            # de-augment chi2
            chi2 = fit.chi2
            for k,pvec in fit.prior.items():
                chi2 -= sum((fit.pmean[k]-gv.mean(pvec))**2/(gv.sdev(pvec))**2)
                Ndof -= len(pvec)
                Ndof += 1 if k=='E' else 0 # why so?

            # evaluate p-value
            fit.chi2red = chi2
            fit.pvalue = p_value(chi2, self.corr.io.mData['nConfs'], Ndof)

        self.fits[(Nstates,trange)] = fit

        if verbose:
            if pval:
                print(f' De-augmented chi2 = {fit.chi2red} with p-value = {fit.pvalue}')
            print(fit)

        if jkfit: # Repeat fit in each jacknife bin ------------------------------------------------
            fitjk = {'pval': []}
            for k,pvec in fit.p.items():
                fitjk[k] = []

            _,__,jk = self.corr.format(trange=trange, smearing=self.smearing, polarization=self.polarization, alljk=True, **kwargs)
            
            for ijk in range(len(self.corr.data.jkbin)):
                if verbose: print(f'{ijk} of {len(self.corr.data.jkbin)}')

                yfit_jk = gv.gvar(
                    np.concatenate([jk[k][ijk,:] for k in self.keys]),
                    gv.evalcov(yfit)
                )
                fjk = lsqfit.nonlinear_fit(
                    data=(xfit,yfit_jk), 
                    fcn=model, 
                    prior=pr, 
                    p0=p0, 
                    maxit=maxit, 
                    svdcut=svdcut, 
                    debug=debug
                )
                
                for k,pvec in fjk.p.items():
                    fitjk[k].append(pvec)                    

                if pval:
                    chi2 = fjk.chi2
                    for k,pvec in fjk.prior.items():
                        chi2 -= sum((fjk.pmean[k]-gv.mean(pvec))**2/(gv.sdev(pvec))**2)
                    fitjk['pval'].append(p_value(chi2, self.corr.io.mData['nConfs'], Ndof))

            for k in fitjk.keys():
                fitjk[k] = np.array(fitjk[k])

        return fitjk if jkfit else None

    def chiexp(self, Nstates, trange, covariance=True):
        # chiexp makes use of the fact that self.fits[...] is the ouptut of nonlinear_fit. This is not optimal
        # one should save only necessary informations FIXME
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

    def GEVP(self, t0=None, smlist=['d','1S'], polarization=None, order=None, jkbin=False):
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

        # Select polarization to be considered
        if polarization is not None:
            if False not in np.isin(polarization,self.corr.data.polarization):
                POL = sorted(polarization)
            else:
                raise ValueError(f'Polarization list {polarization} contains at least one item which is not contained in data.polarization')
        else:
            POL = sorted(self.corr.data.polarization.values)

        Nt   = self.corr.data.timeslice.size
        Nc   = self.corr.data.jkbin.size
        Nsm  = int(np.sqrt(len(smr_flat)))
        Npol = len(POL)

        # Reshape data and construct corr matrix
        C = np.moveaxis(self.corr.data.loc[smr_flat,POL,:,:].values,0,-1).reshape(Npol,Nc,Nt,Nsm,Nsm)

        # Iterate over time and solve the GEVP conf by conf
        eigs = {}
        for ipol,pol in enumerate(POL):
            eigs[pol] = []
            for jk in range(Nc):
                aux = []
                for t in range(Nt):
                    eigval, eigvec = eigh(C[ipol,jk,t,:,:],b=C[ipol,jk,t0 if t0 is not None else t//2,:,:])
                    if order is not None:
                        aux.append(sorted(eigval)[order]) # sort eigenvalue and select the `order`-th
                    else: 
                        aux.append(eigval)
                eigs[pol].append(aux)
            eigs[pol] = np.array(eigs[pol])

        if order is None:
            return eigs


        # Calculate effective mass
        aux = {}
        for pol in POL:
            aux[pol] = np.log(eigs[pol]/np.roll(eigs[pol],-1,axis=1))[:,:-1]

        if jkbin:
            return aux
        else:
            Eeff = {}
            for pol in POL:
                Eeff[pol] = gv.gvar(
                    aux[pol].mean(axis=0),
                    np.cov(aux[pol],rowvar=False) * (eigs[pol].shape[0]-1  if self.corr.info.binsize is not None else 1.)
                )[:-1]
            print(Eeff)
            return Eeff

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
    ens      = 'Coarse-1'
    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex'
    meson    = 'B'
    mom      = '100'
    binsize  = 11
    
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr =  Correlator(io,jkBin=binsize,CrossSmearing=True)


    trange = (10,19)
    nexc = 3
    smr = ['d-d','1S-1S','d-1S']

    fitter = CorrFitter(corr,smearing=smr)
    fitter.fit(nexc,trange,verbose=True)


if __name__ == "__main__":
    test()
