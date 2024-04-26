import sys
import itertools
import argparse
import numpy as np
import gvar  as gv

from matplotlib  import pyplot as plt
from tqdm        import tqdm

from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, Correlator, plot_effective_coeffs



def eff_coeffs(FLAG):
    ens      = 'Fine-1'
    mes      = 'Dst'
    mom      = '100'
    binsize  = 16
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
        tmax = 37
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
        tmin = 15
        tmax = 30
        tmaxe = stag.tmax(threshold=0.3)
        trange = (tmin,tmax)

        print(tmaxe,trange)

        cov_specs = dict(
            diag   = False,
            block  = False,
            scale  = True,
            shrink = True,
            # cutsvd = 0.01,
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

# def global_eff_coeffs(ens, mes, trange, chiexp=True, config_file='/Users/pietro/code/software/B2heavy/routines/2pts_fit_config.toml', **cov_specs):
#     with open(config_file,'rb') as f:
#         config = tomllib.load(f)

#     mom_list = config['data'][ens]['mom_list']

#     Meffs = {}
#     xdata_meff = {}
#     data_meff = {}
#     fullcov_data = {}
#     prior_meff = {}
#     for mom in mom_list:
#         io   = CorrelatorIO(ens,mes,mom,PathToDataDir=config['data'][ens]['data_dir'])
#         stag = Correlator(
#             io       = io,
#             jkBin    = config['data'][ens]['binsize'],
#             smearing = config['fit'][ens][mes]['smlist']
#         )

#         trange_eff = config['fit'][ens][mes]['mom'][mom]['trange_eff']
#         # Compute effective masses and ampls. for priors calc
#         Xdict, Aeff, aplt, apr, Meff, meffs, mpr = stag.meff(
#             trange    = trange_eff,
#             plottable = True,
#             **cov_specs
#         )
#         _,_,_,_,_,fully,_ = stag.meff(trange=trange, plottable=True)

#         # Slice and create fit data
#         tmp,tmp1 = [],[]
#         for k in meffs:
#             inan = np.isnan(gv.mean(meffs[k][min(trange):(max(trange)+1)])) 
#             tmp.append(
#                 meffs[k][min(trange):(max(trange)+1)][~inan]
#             )
#             tmp1.append(
#                 fully[k][min(trange):(max(trange)+1)][~inan]
#             )
#         data_meff[mom] = np.concatenate(tmp)
#         xdata_meff[mom] = np.arange(len(data_meff[mom]))
#         prior_meff[mom] = gv.gvar(Meff.mean,Meff.sdev)
#         Meffs[mom] = Meff

#         # Compute effective masses and ampls. for priors calc
#         fullcov_data[mom] = np.concatenate(tmp1)

#     # Perform fit
#     fit = lsqfit.nonlinear_fit(
#         data  = (xdata_meff,data_meff),
#         fcn   = ConstantDictModel,
#         prior = prior_meff
#     )

#     if not chiexp:
#         return fit
#     else:
#         # Compute chiexp and p value -------------------------------------
#         meffs = np.concatenate([data_meff[k] for k in mom_list])
#         fitcov = gv.evalcov(meffs)
#         w = np.linalg.inv(fitcov)
#         res = np.concatenate(
#             [gv.mean(ConstantDictModel(xdata_meff,fit.p)[k]) - gv.mean(data_meff[k]) for k in mom_list]
#         )
#         chi2 = res.T @ w @ res

#         fullc = np.concatenate([fullcov_data[k] for k in mom_list])
#         cov = gv.evalcov(fullc)

#         # Compute jacobian
#         jac = []
#         for i,_ in enumerate(mom_list):
#             tmp = []
#             for j,p in enumerate(mom_list):
#                 tmp.append(
#                     np.full_like(data_meff[p], 1. if i==j else 0.)
#                 )
#             tmp = np.concatenate(tmp)
#             jac.append(tmp)
#         jac = np.array(jac).T

#         # Calculate expected chi2
#         Hmat = jac.T @ w @ jac
#         Hinv = np.diag(1/np.diag(Hmat))
#         wg = w @ jac
#         proj = np.asfarray(w - wg @ Hinv @ wg.T)
#         chiexp = np.trace(proj @ cov)

#         l,evec = np.linalg.eig(cov)
        
#         if np.any(np.real(l))<0:
#             mask = np.real(l)>1e-12
#             Csqroot = evec[:,mask] @ np.diag(np.sqrt(l[mask])) @ evec[:,mask].T
#         else:
#             Csqroot= evec @ np.diag(np.sqrt(l)) @ evec.T

#         numat = Csqroot @ proj @ Csqroot

#         l,_ = np.linalg.eig(numat)
#         ls = l[l.real>=1e-14].real

#         p = 0
#         for _ in range(50000):
#             ri = np.random.normal(0.,1.,len(ls))
#             p += 1. if (ls.T @ (ri**2) - chi2)>=0 else 0.
#         p /= 50000

#         return fit, chi2, chiexp, p








if __name__=='__main__':
    prs = argparse.ArgumentParser()
    prs.add_argument('--do', type=int)

    args = prs.parse_args()

    eff_coeffs(args.do)
