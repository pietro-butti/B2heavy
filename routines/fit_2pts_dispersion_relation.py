# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_disp_rel.py --config   [file location of the toml config file]
                        --ensemble [ensemble analyzed]
                        --meson    [meson analyzed]
                        --momlist  [list of moments that have to be fitted to disp. relation]
                        --jk       [use jackknifed fit?]
                        --jkfit    [fit dispersion relation in every jkbin?]
                        --readfrom [*where* are the `config` analysis results?]
                        --plot     [do you want to plot?]
                        --show     [do you want to display plot?]
                        --saveto   [*where* do you want to save files?]

Examples
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import argparse
import pickle
import sys
import os
import numpy as np
import gvar as gv
import pandas as pd
import lsqfit
import tomllib
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from b2heavy.ThreePointFunctions.utils import read_config_fit, dump_fit_object
import fit_2pts_utils as utils

from b2heavy import FnalHISQMetadata
from b2heavy.FnalHISQMetadata import *


def extract_single_energy(tag,path=None,N=0,jk=False,key='dE'):
    ret = read_config_fit(tag,jk=jk,path=path)

    if not jk:
        return ret[1][key][N]
    else:
        return ret[key][:,N]

def mom_to_p2(mom,L=2*np.pi):
    return sum([(2*np.pi/L*float(px))**2 for px in mom])

def mom_to_pvec(mom,L=2*np.pi):
    return [(2*np.pi/L*float(px))**2 for px in mom]


def extract_energies(ensemble,meson,momlist=None,jk=False,readfrom='.',tag='fit2pt_config',key='dE',sort=True):
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    filename = f'{tag}_{ensemble}_{meson}_'

    E = {}
    if momlist is None:
        for file in os.listdir(readfrom):
            f = os.path.join(readfrom,file)

            file_end = '_jk_fit.pickle' if jk else '_fit.pickle'
            if file.startswith(filename) and file.endswith(file_end):
                if not jk and 'jk' in file:
                    continue

                name,_ = f.split(file_end)
                mom = name.split('_')[-1]

                tag = name.split('/')[-1]
                path = name.split(tag)[0]

                E[mom] = extract_single_energy(tag,path=path,N=0,jk=jk,key=key)
    else:
        for mom in momlist:
            f = os.path.join(readfrom,f'{filename}{mom}')
            E[mom] = extract_single_energy(f,N=0,jk=jk,key=key)

    if sort:
        psorted = list(E.keys())
        psorted.sort(key=mom_to_p2)
        E = {p: E[p] for p in psorted}

    return E

def dispersion_relation(p,M1,M2,M4,w4):
    p2  = sum(p**2)
    p22 = p2**2
    p4  = sum(p**4)

    return M1**2 + (M1/M2 * p2) + ((1/M2**2 - M1/M4**3)/4 * p22) - (M1*w4/3 * p4)

def dispersion_relation_vec(pvec,M1,M2,M4,w4):
    return np.asarray([dispersion_relation(p,M1,M2,M4,w4) for p in pvec])

def dispersion_relation_lsqfit(pveclist,d):
    M1,M2,M4,w4 = d['M1'],d['M2'],d['M4'],d['w4']

    res = []
    for pvec in pveclist:
        p2  = sum(pvec**2)
        p22 = p2**2
        p4  = sum(pvec**4)

        res.append(
            M1**2 + (M1/M2 * p2) + ((1/M2**2 - M1/M4**3)/4 * p22) - (M1*w4/3 * p4)
        )

    return np.array(res)

def disprel_priors(mp,error=1.):
    p = {'aInv': 1/mp['aSpc']}

    lbQcd = 0.6/p['aInv']
    lsQcd = 0.4/p['aInv']


    m0D = 0.5/mp['u0']*(1./mp['kappaD'] - 1/mp['kappaCr'])
    m0B = 0.5/mp['u0']*(1./mp['kappaB'] - 1/mp['kappaCr'])

    #   c^2 = M1/M2 Speed of light prior, only for D* meson
    c2_0 = m1_0(m0D)/m2_0(m0D)                      # Tree level
    c2_1 = y2*g2(m0D) + z2*h2(m0D)                  # 1-loop correction

    p['c2']   = c2_0 + als(mp['aSpc'])*c2_1 + lbQcd*l2(m0D)
    p['c2s']  = np.sqrt((y2s*g2(m0D)*als(mp['aSpc']))**2 + (z2s*h2(m0D)*als(mp['aSpc']))**2 + (lsQcd*l2(m0D))**2)

    #   A4' = M1 x W4 prior
    cq_0 = w4_0(m0D)*m1_0(m0D)                      # Tree level
    cq_1 = y4p*g4p(m0D) + z4p*h4p(m0D)              # 1-loop correction

    p['A4p']  = (-1./3.)*(cq_0 + als(mp['aSpc'])*cq_1 + lbQcd*l4p(m0D))
    p['A4ps'] = ( 1./3.)*np.sqrt((y4ps*g4p(m0D)*als(mp['aSpc']))**2 + (z4ps*h4p(m0D)*als(mp['aSpc']))**2 + (lsQcd*l4p(m0D))**2)   # Error l2 --> l4p


    #   A4 = cp prior (A4)
    cp_0 = 1/(m2_0(m0D)*m2_0(m0D)) - m1_0(m0D)*invm4_0_cu(m0D)  # Tree level
    cp_1 = y4*g4(m0D) + z4*h4(m0D)                              # 1-loop correction

    p['A4']   = 0.25*(cp_0 + als(mp['aSpc'])*cp_1 + lbQcd*l4(m0D))       # Error 1 --> als(p['aSpc'])*cp_1
    p['A4s']  = 0.25*np.sqrt((y4s*g4(m0D)*als(mp['aSpc']))**2 + (z4s*h4(m0D)*als(mp['aSpc']))**2 + (lbQcd*l4(m0D))**2)


    pr = {'M1': gv.gvar(0.5,error)}
    pr['M2'] = pr['M1']/np.sqrt(p['c2'])
    pr['M4'] = (pr['M1']/(1/pr['M2']**2 - 4*p['A4'])) ** (1/3)
    pr['w4'] = 3*p['A4p']/pr['M1']

    return pr




def fit_disp_rel(e0:dict, Lvol=1., priors=None, verbose=True):
    pv = [2*np.pi/Lvol*np.array([float(px) for px in m]) for m in e0]

    psort = list(e0.keys())
    psort.sort(key=mom_to_p2)

    e0vec = np.array([e0[p] for p in psort])
    yfit = e0vec**2

    if e0vec.ndim==2:
        yfit = gv.gvar(
            yfit.mean(axis=1),
            np.cov(yfit) * (yfit.shape[-1]-1)
        )

    fit = lsqfit.nonlinear_fit(
        data  = (pv,yfit),
        fcn   = dispersion_relation_lsqfit,
        prior = priors
    )

    if verbose:
        print(fit)

    return fit



def fit_disp_rel_jk(fit:lsqfit.nonlinear_fit, e0):
    psort = list(e0.keys())
    psort.sort(key=mom_to_p2)
    e0vec = np.array([e0[p] for p in psort])
    assert e0vec.ndim==2

    yfit = e0vec**2

    aux = []
    for jk in tqdm(range(yfit.shape[-1])):
        f = lsqfit.nonlinear_fit(
            data  = (fit.x,yfit[:,jk],gv.evalcov(fit.y)),
            fcn   = fit.fcn,
            prior = fit.prior 
        )
        aux.append(f.pmean)
    df = pd.DataFrame(aux)
    fitp = gv.gvar(
        df.mean().values,
        df.cov().values*(df.shape[0]-1)
    )
    
    return {c: fitp[i] for i,c in enumerate(df.columns)}



def plot_disp_rel(ax,fit,popt,Lvol,**kwargs):
    # breakpoint()
    p2 = np.array([sum(np.array(p)**2) for p in fit.x])
    ax.errorbar(p2,gv.mean(fit.y),gv.sdev(fit.y),**kwargs)

    plist = [np.sqrt([x/3,x/3,x/3]) for x in np.arange(0,max(p2)+0.1,0.01)]
    xplot = [sum(p**2) for p in plist]
    fitpar = [popt[k] for k in ['M1','M2','M4','w4']]
    fplot = [dispersion_relation(p,*fitpar) for p in plist]
    ax.fill_between(xplot,gv.mean(fplot)-gv.sdev(fplot),gv.mean(fplot)+gv.sdev(fplot),alpha=0.2)

    dp = p2[1]-p2[0]
    ax.set_xlim(xmax=max(p2)+dp)

    de2 = abs((fit.y[-1]-fit.y[-2]).mean)
    ax.set_ylim(ymax=max(gv.mean(fit.y))+de2, ymin = min(gv.mean(fit.y))-de2)

    return


prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson', type=str)
prs.add_argument('-mm','--momlist', type=str, nargs='+', default=None)
prs.add_argument('--jk', action='store_true')
prs.add_argument('--jkfit', action='store_true')
prs.add_argument('--readfrom', type=str, default='./')
prs.add_argument('--saveto',   type=str, default='./')
prs.add_argument('--override', action='store_true')
prs.add_argument('--plot', action='store_true')
prs.add_argument('--show', action='store_true')




def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ens = args.ensemble
    mes = args.meson
    tag = f'{ens}_{mes}'

    mdata = FnalHISQMetadata.params(ens)
    Lvol   = mdata['L']
    alphas = mdata['alphaS'] 


    readfrom = f'{DEFAULT_ANALYSIS_ROOT}/' if args.readfrom=='default' else args.readfrom
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    saveto = f'{DEFAULT_ANALYSIS_ROOT}/' if args.saveto=='default' else args.saveto
    if not os.path.exists(saveto):
        raise NameError(f'{saveto} is not an existing location')
    saveplot = f'{DEFAULT_ANALYSIS_ROOT}' if args.saveto=='default' else args.saveto

    tag = f'{ens}_{mes}'


    es = extract_energies(ens,mes,jk=args.jk,readfrom=readfrom,sort=True,momlist=args.momlist)

    # priors = disprel_priors(mdata)
    # priors = gv.gvar(dict(M1='0.8(5.)',M2='0.8(5.)',M4='0.8(5.)',w4='0.8(5.)'))
    priors = dict(
        M1 = gv.gvar(0.5,1.5),
        M2 = gv.gvar(0.5,1.5),
        M4 = gv.gvar(0.5,1.5),
        w4 = gv.gvar(0.,2.5)
    )



    fit = fit_disp_rel(es, Lvol=Lvol, priors=priors)
    if args.jkfit:
        popt = fit_disp_rel_jk(fit,es)        
    else:
        popt = fit.p


    # Save data -----------------------------------------------------------------
    gv.dump(fit.p,f'{saveto}/fit2pts_dispersion_relation_{tag}.pickle')



    # Plot fit ------------------------------------------------
    if args.plot:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12

        plt.figure(figsize=(4,3))
        ax = plt.subplot(1,1,1)

        plot_disp_rel(ax, fit, popt, Lvol=Lvol, fmt='o', ecolor='C0', mfc='w', capsize=2.5)

        ax.set_ylabel(r'$(aE(\mathbf{p}))^2$')
        ax.set_xlabel(r'$a^2\mathbf{p}^2$')

        ax.grid(alpha=0.2)

        plt.title(tag)
        plt.tight_layout()
        plt.savefig(f'{saveto}/fit2pts_dispersion_relation_{tag}.pdf')

        if args.show:
            plt.show()




    # Plot discretization errors ------------------------------------------------
    if args.plot:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        
        p2 = np.sum(np.array(fit.x)**2,axis=1)
        denominator = p2 + fit.p['M1']**2
        yplot = fit.y/denominator


        fig,ax = plt.subplots(2,1,figsize=(4, 4),sharex=True)

        # Discr. errors on energies

        ax[0].errorbar(p2,gv.mean(yplot),gv.sdev(yplot),fmt='o', ecolor='C0', mfc='w', capsize=2.5)
        ax[0].axhline(1.,color='gray',alpha=0.5,linestyle=':')

        endp = max(p2)+p2[1]
        xcone = np.arange(0,endp,0.01)
        ax[0].fill_between(xcone, alphas*xcone+1,-alphas*xcone+1,alpha=0.1)

        ax[0].set_xlim(xmin=-0.01,xmax=endp-0.01)

        ax[0].set_ylabel(r'$\frac{E^2(\mathbf{p})}{\mathbf{p}^2 + M_1^2}$')
        ax[0].set_xlabel(r'$a^2\mathbf{p}^2$')

        ax[0].grid(alpha=0.2)


        # # Discr. errors on coefficients
        # Z1S = []
        # Zd  = []
        # for p in psort:
        #     fname = f'fit2pt_config_{ens}_{mes}_'
        #     f = os.path.join(readfrom,f'{fname}{p}')
        #     collinear = p.endswith('00') and not p.startswith('0')

        #     k1s_1  = 'Z_1S_Par' if (collinear and mes=='Dst') else 'Z_1S_Unpol'      
        #     z1s_1 = extract_single_energy(f,N=0,jk=JK,key=k1s_1)
        #     z1s_1 = np.exp(z1s_1)*np.sqrt(2*E[p])

        #     k1s_2  = 'Z_1S_Bot' if (collinear and mes=='Dst') else 'Z_1S_Unpol'      
        #     z1s_2 = extract_single_energy(f,N=0,jk=JK,key=k1s_2)
        #     z1s_2 = np.exp(z1s_2)*np.sqrt(2*E[p])

        #     Z1S.append((z1s_1+z1s_2)/2)


        #     kd_1  = 'Z_d_Par' if (collinear and mes=='Dst') else 'Z_d_Unpol'      
        #     zd_1 = extract_single_energy(f,N=0,jk=JK,key=kd_1)
        #     zd_1 = np.exp(zd_1)*np.sqrt(2*E[p])

        #     kd_2  = 'Z_d_Bot' if (collinear and mes=='Dst') else 'Z_d_Unpol'      
        #     zd_2 = extract_single_energy(f,N=0,jk=JK,key=kd_2)
        #     zd_2 = np.exp(zd_2)*np.sqrt(2*E[p])

        #     Zd.append((zd_1+zd_2)/2) 

        # Z1S = np.array(Z1S)
        # Zd  = np.array(Zd)

        # Z1S = Z1S/Z1S[0]; Z1S[0] = np.ones_like(Z1S[0]) if JK else gv.gvar(1.,0)
        # Zd  = Zd/Zd[0];   Zd [0] = np.ones_like(Z1S[0]) if JK else gv.gvar(1.,0)


        # if JK:
        #     Z1S = gv.gvar( Z1S.mean(axis=1), np.cov(Z1S) * Z1S.shape[-1] )
        #     Zd  = gv.gvar( Zd.mean(axis=1) , np.cov(Zd)  * Zd.shape[-1]  )


        # # ax[1].errorbar(p2,gv.mean(Z1S),gv.sdev(Z1S),fmt='o', ecolor='C0', mfc='w', capsize=2.5)
        # ax[1].errorbar(p2,gv.mean(Zd ),gv.sdev(Zd ),fmt='o', ecolor='C0', mfc='w', capsize=2.5)
        # ax[1].axhline(1.,color='gray',alpha=0.5,linestyle=':')

        # endp = max(p2)+p2[1]
        # xcone = np.arange(0,endp,0.01)
        # ax[1].fill_between(xcone, alphas*xcone+1,-alphas*xcone+1,alpha=0.1)

        # ax[1].set_xlim(xmin=-0.01,xmax=endp-0.01)

        # ax[1].set_ylabel(r'$\frac{Z(\mathbf{p})}{Z(0)}$')
        # ax[1].set_xlabel(r'$a^2\mathbf{p}^2$')

        # ax[1].grid(alpha=0.2)



        ax[0].set_title(tag)
        plt.tight_layout()
        plt.savefig(f'{saveto}/fit2pts_discretization_errors_{tag}.pdf')


        if args.show:
            plt.show()



    return



if __name__ =="__main__":
    main()