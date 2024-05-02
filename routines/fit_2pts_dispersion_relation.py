# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_disp_rel.py --config   [file location of the toml config file]
                        --ensemble [ensemble analyzed]
                        --meson    [meson analyzed]
                        --momlist  [list of moments that have to be fitted to disp. relation]
                        --jkfit    [use jackknifed fit]
                        --readfrom [*where* are the `config` analysis results?]
                        --plot     [do you want to plot?]
                        --showfig  [do you want to display plot?]
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
import lsqfit
import tomllib

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from b2heavy.ThreePointFunctions.utils import read_config_fit, dump_fit_object
import fit_2pts_utils as utils

from b2heavy import FnalHISQMetadata


def extract_single_energy(tag,path=None,N=0,jk=False):
    ret = read_config_fit(tag,jk=jk,path=path)

    if not jk:
        return ret[1]['E'][N]
    else:
        return ret['E'][:,N]


def extract_energies(ensemble,meson,momlist=None,jk=False,readfrom='.',tag='fit2pt_config'):
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    filename = f'{tag}_{ensemble}_{meson}_'

    E = {}
    if momlist is None:
        for file in os.listdir(readfrom):
            f = os.path.join(readfrom,file)

            file_end = 'fit.pickle' if not jk else '_jk_fit.pickle'
            if file.startswith(filename) and file.endswith(file_end):
                name,_ = f.split(file_end)
                mom = name.split('_')[-1]

                tag = name.split('/')[-1]
                path = name.split(tag)[0]
                E[mom] = extract_single_energy(tag,path=path,N=0,jk=jk)
    else:
        for mom in momlist:
            f = os.path.join(readfrom,f'{filename}{mom}')
            E[mom] = extract_single_energy(f,N=0,jk=jk)

    return E


def mom_to_p2(mom,L=2*np.pi):
    return sum([(2*np.pi/L*float(px))**2 for px in mom])


def dispersion_relation(p,M1,M2,M4,w4):
    p2  = sum(p**2)
    p22 = p2**2
    p4  = sum(p**4)

    return M1**2 + (M1/M2 * p2) + ((1/M1**2 - M1/M4**3)/4 * p22) - (M1*w4/3 * p4)


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
            M1**2 + (M1/M2 * p2) + ((1/M1**2 - M1/M4**3)/4 * p22) - (M1*w4/3 * p4)
        )

    return np.array(res)


def plot_dispersion_relation(ax,mom,p2,E0,fitpar=None,chi2=None):
    xfit = p2
    yfit = E0**2

    ax.scatter(xfit,gv.mean(yfit),marker='s',facecolors='none')
    ax.errorbar(xfit,gv.mean(yfit),yerr=gv.sdev(yfit),fmt='.',capsize=2)

    if fitpar is not None and chi2 is not None:
        plist = [np.sqrt([x/3,x/3,x/3]) for x in np.arange(0,max(xfit)+1.,0.2)]
        xplot = [sum(p**2) for p in plist]
        yplot = [dispersion_relation(p,*fitpar) for p in plist]
                
        ax.fill_between(xplot,gv.mean(yplot)-gv.sdev(yplot),gv.mean(yplot)+gv.sdev(yplot),alpha=0.2)


def plot_discretization_errors(ax,mom,p2,E0,pars,alphas,L=1):
    xplot = p2

    p2M2 = p2 + pars[0]
    yplot = E0**2/p2M2

    ax.errorbar(xplot,gv.mean(yplot),gv.sdev(yplot),fmt='o', ecolor='C0', mfc='w', capsize=2.5)

    ax.axhline(1.,color='gray',alpha=0.5,linestyle=':')

    xcone = np.arange(0,max(xplot),0.01)
    ax.fill_between(xcone, alphas*xcone+1,-alphas*xcone+1,alpha=0.1)


prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson', type=str)
prs.add_argument('-mm','--momlist', type=str, nargs='+', default=[])
prs.add_argument('--jkfit', action='store_true')
prs.add_argument('--readfrom', type=str, default='./')
prs.add_argument('--saveto',   type=str, default='./')
prs.add_argument('--override', action='store_true')
prs.add_argument('--plot', action='store_true')
prs.add_argument('--showfig', action='store_true')




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

    JK = True if args.jkfit else False

    E = extract_energies(
        ensemble = ens,
        meson    = mes,
        momlist  = None if not args.momlist else args.momlist,
        jk       = JK,
        readfrom = readfrom  
    )
    

    # Create vector with fitting points ---------------------------------------
    psort = list(E.keys())
    psort.sort(key=lambda x: mom_to_p2(x,L=Lvol))


    pv = [2*np.pi/Lvol*np.array([float(px) for px in mom]) for mom in psort]
    p2 = [sum(np.array([float(px)*2*np.pi/Lvol for px in mom])**2) for mom in psort]

    if not JK:
        E0 = np.asarray([E[kp] for kp in psort])    
    else:
        E0 = []
        for mom in psort:
            E0.append(E[mom])
        E0 = np.array(E0).T
        E0 = gv.gvar(
            E0.mean(axis=0),
            np.cov(E0,rowvar=False,bias=True) * (E0.shape[0]-1)
        )

    print(p2,psort)


    # Perform fit --------------------------------------------------------------
    priors = dict(
        M1 = gv.gvar(0.5,1.5),
        M2 = gv.gvar(0.5,1.5),
        M4 = gv.gvar(0.5,1.5),
        w4 = gv.gvar(0.5,1.5)
    )
    fit = lsqfit.nonlinear_fit(
        data  = (pv,E0**2),
        fcn   = dispersion_relation_lsqfit,
        prior = priors
    )

    # Save data -----------------------------------------------------------------
    gv.dump(fit.p,f'{saveto}/fit2pts_dispersion_relation_{tag}.pickle')


    # Plot discretization errors ------------------------------------------------
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12
    
    m1 = fit.p['M1']
    denominator = np.array(p2) + m1**2
    yplot = E0**2/denominator

    plt.figure(figsize=(6, 2))
    ax = plt.subplot(1,1,1)

    ax.errorbar(p2,gv.mean(yplot),gv.sdev(yplot),fmt='o', ecolor='C0', mfc='w', capsize=2.5)
    ax.axhline(1.,color='gray',alpha=0.5,linestyle=':')

    endp = max(p2)+p2[1]
    xcone = np.arange(0,endp,0.01)
    ax.fill_between(xcone, alphas*xcone+1,-alphas*xcone+1,alpha=0.1)

    ax.set_xlim(xmin=-0.01,xmax=endp-0.01)

    ax.set_ylabel(r'$\frac{E^2(\mathbf{p})}{\mathbf{p}^2 + M_1^2}$')
    ax.set_xlabel(r'$a^2\mathbf{p}^2$')

    plt.tight_layout()
    plt.title(tag)
    plt.savefig(f'{saveto}/fit2pts_dispersion_relation_{tag}.pdf')

    plt.show()




    return



if __name__ =="__main__":
    main()