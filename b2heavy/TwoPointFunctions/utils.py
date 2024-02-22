import numpy as np
import gvar  as gv
import tomllib
import sys
import os

from scipy.special import gammaln as gammaLn

from ..Shrink import shrink


def ConstantModel(x,p):
    return np.array([p['const']]*len(x))

def ConstantFunc(x,c):
    if isinstance(x,np.ndarray) or isinstance(x,list):
        return np.array([c]*len(x))
    else:
        return c


def PeriodicExpDecay(Nt):
    return lambda t,E,Z: Z * ( np.exp(-E*t) + np.exp(-E*(Nt-t)) ) 

def NplusN2ptModel(Nstates,Nt,sm,pol):
    sm1,sm2 = sm.split('-')
    mix = sm1!=sm2

    def aux(t,p):
        ans = [0.] * len(t)

        E0, E1 = p['E'][0], p['E'][0]+np.exp(p['E'][1])
        Z0 = np.exp(p[f'Z_{sm1}_{pol}'][0]) * np.exp(p[f'Z_{sm2}_{pol}'][0])
        Z1 = np.exp(p[f'Z_{sm1}_{pol}'][1]) * np.exp(p[f'Z_{sm2}_{pol}'][1])
        ans += PeriodicExpDecay(Nt)(t,E0,Z0) + (-1)**(t+1)*PeriodicExpDecay(Nt)(t,E1,Z1)

        Es = [E0,E1]
        for i in range(2,2*Nstates):
            Ei = Es[i-2] + np.exp(p['E'][i])
            Z = p[f'Z_{sm if mix else sm1}_{pol}'][i-2 if mix else i]**2
            ans += PeriodicExpDecay(Nt)(t,Ei,Z) * (-1)**(i*(t+1))

            Es.append(Ei)
        return ans

    return aux

def jkCorr(data, bsize=1):

    if bsize==0:
        return data
    else:
        rData = (data.shape[0]//bsize)
        nData = rData*bsize
        dIn   = data[0:nData,:]

        avg   = dIn.mean(axis=0)

        base  = dIn.sum(axis=0) # avg * nData

        dOut  = dIn * 0.0

        for cf in range(rData):
            pvg = dIn[cf*bsize:(cf+1)*bsize,:].sum(axis=0)
            dOut[cf,:] = (base - pvg)/(nData - bsize)

        return dOut[:rData,:]

def chi2_distribution(chi2, nConf, dof):
    """
        Compute the cumulative chi-square distribution with `dof` degrees of freedom, corrected for finite sample size `nConf` integrated from `chi2` to infinity.

        Its definition is related to formula (B1) of PRD 93 113016.
    """
    return (np.exp(gammaLn(nConf/2.0) - gammaLn(dof/2.0) - gammaLn((nConf-dof)/2.0) + (-dof/2.0)*np.log(nConf) + ((dof-2.0)/2.0)*np.log(chi2) + (-nConf/2.0)*np.log(1.0+chi2/nConf)))

def p_value(chi2, nConf, dof):
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
            sum = sum + chi2_distribution(tChi2, nConf, dof)
            tChi2 = tChi2 + eps

        if sum*eps < 0.9:
            return 1.0 - sum*eps;

    nSteps = 0;
    eps = (2.0*np.sqrt(2.0*dof))/100.0;

    y = chi2_distribution(chi2, nConf, dof);

    sum = 0.0
    tChi2 = chi2+0.5*eps

    while True:
        tChi2 = tChi2 + eps
        nSteps = nSteps + 1
        z = chi2_distribution(tChi2, nConf, dof);

        if nSteps > 1000 or z < 0.0001*y:
            break;

        sum += z;

    return sum*eps

def covariance_shrinking(avg,cov,nEff):
    data = gv.gvar(avg, cov)
    corr = gv.evalcorr(data)

    # Decompose into eigenvalues
    eval, evec = np.linalg.eig(corr)  # (eigvals, eigvecs)

    # Sort in descending order
    evOr = np.argsort(eval)[::-1]
    eval  = eval[evOr]
    evec  = evec[:,evOr]

    # Shrink the eigenvalue spectrum
    eVsk = shrink.direct_nl_shrink(eval, nEff)

    # Reconstruct eigenvalue matrix: vecs x diag(vals) x vecs^T
    crSk = np.matmul(evec, np.matmul(np.diag(eVsk), evec.transpose()))
    nDta = gv.correlate(data, crSk)

    tCov = gv.evalcov(nDta)
    eV2 = np.linalg.eigvals(gv.evalcorr(nDta))  # Test

    return tCov

def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data


def test():
    print(f'Hello from {__file__}')
    pass

if __name__ == "__main__":
    test()