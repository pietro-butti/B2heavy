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

from .DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

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


def extract_single_energy(filename,N=0,jk=False):
    with open(filename,'rb') as f:
        dfit = pickle.load(f)

    if not jk:
        return dfit['p']['E'][N]
    else:
        return gv.mean(dfit['E'][:,N])

def extract_energies(ensemble,meson,momlist=None,jk=False,readfrom='.',tag='fit2pt_config'):
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    filename = f'{tag}_{ensemble}_{meson}_' if not jk else f'{tag}_jk_{ensemble}_{meson}_'

    E = {}
    if momlist is None:
        for file in os.listdir(readfrom):
            f = os.path.join(readfrom,file)
            if file.startswith(filename) and file.endswith('.pickle'):
                name,_ = f.split('.pickle')
                mom = name.split('_')[-1]
                E[mom] = extract_single_energy(f,N=0,jk=jk)
    else:
        for mom in momlist:
            f = os.path.join(readfrom,f'{filename}{mom}.pickle')
            E[mom] = extract_single_energy(f,N=0,jk=jk)

    return E

def mom_to_p2(mom):
    return sum([float(px)**2 for px in mom])

def format_energies(E,jk=False):
    psort = list(E.keys())
    psort.sort(key=lambda x: mom_to_p2(x))

    xfit = np.array([mom_to_p2(mom) for mom in psort])
    yfit = np.array([E[mom] for mom in psort])

    if jk:
        yfit =  gv.gvar(
            yfit.mean(axis=1),
            np.cov(yfit) * (yfit.shape[1]-1)
        )

    return psort,xfit,yfit

def dispersion_relation(pvec,M1,M2,M4,w4):
    px,py,pz = pvec
    p2 = px**2 + py**2 + pz**2
    p22 = p2**2
    p4  = px**4 + py**4 + pz**4

    return M1**2 + (M1/M2 * p2) + ((1/M1**2 - M1/M4**3)/4 * p22) - (M1*w4/3 * p4)

def model(pvec,M1,M2,M4,w4):
    return [dispersion_relation(p,M1,M2,M4,w4) for p in pvec]

def fit_dispersion_relation(momlist,E0):
    # Define fit points
    xfit = [[float(px) for px in mom] for mom in momlist]
    yfit = E0**2

    # Fit function
    popt,pcov = curve_fit(
        model,
        xfit, gv.mean(yfit),
        sigma = gv.evalcov(yfit),
    )
    pars = gv.gvar(popt,pcov)

    # Calculate chi2
    r = gv.mean(yfit) - model(xfit,*popt)
    chisq = r.T @ np.linalg.inv(gv.evalcov(yfit)) @ r

    return pars,chisq

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




prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('--ensemble', type=str)
prs.add_argument('--meson', type=str)
prs.add_argument('--momlist', type=str, nargs='+', default=[])
prs.add_argument('--jkfit', action='store_true')
prs.add_argument('--read_from', type=str, default='./')
prs.add_argument('--saveto',   type=str, default='./')
prs.add_argument('--override', action='store_true')
prs.add_argument('--plot', action='store_true')
prs.add_argument('--showfig', action='store_true')




def main():
    args = prs.parse_args()

    ens = args.ensemble
    mes = args.meson

    tag = f'{ens}_{mes}'
    

    readfrom = f'{DEFAULT_ANALYSIS_ROOT}/{ens}' if args.saveto=='default' else args.read_from
    if not os.path.exists(readfrom):
        raise NameError(f'{readfrom} is not an existing location')

    saveto = f'{DEFAULT_ANALYSIS_ROOT}/{ens}' if args.saveto=='default' else args.saveto
    if not os.path.exists(saveto):
        raise NameError(f'{saveto} is not an existing location')
    saveplot = f'{DEFAULT_ANALYSIS_ROOT}/PLOTS' if args.saveto=='default' else args.saveto

    JK = True if args.jkfit else False

    E = extract_energies(
        ensemble = ens,
        meson    = mes,
        momlist  = None if not args.momlist else args.momlist,
        jk       = JK,
        readfrom = readfrom  
    )

    mom,p2,E0 = format_energies(E,jk=JK)
    pars,chi2 = fit_dispersion_relation(mom,E0)
    tosave = dict(pars=pars,chi2=chi2)


    file = f'{saveto}/fit2pt_disp_rel_{tag}.pickle'
    with open(file,'wb') as handle:
        pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if args.plot:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(6, 4))
        ax = plt.subplot(1,1,1)

        plot_dispersion_relation(ax,mom,p2,E0,fitpar=pars,chi2=chi2)

        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_ylabel(r'$a^2 E^2(\mathbf{p})$')
        ax.set_xlabel(r'$a^2\mathbf{p}^2$')

        ax.set_xlim(xmin=-0.1)

        plt.title(f'{tag}')
        plt.tight_layout()
        plt.savefig(f'{saveplot}/fit2pt_disp_rel_{tag}.pdf')


        if args.showfig:
            plt.show()



def test():
    JK = True
    PLOT = True
    MOMLIST = [
        "000", 
        "100", 
        "110", 
        "200", 
        "211", 
        "300", 
        # "222",
        "400" 
    ]


    E = extract_energies(
        ensemble='MediumCoarse',
        meson='Dsst',
        momlist=MOMLIST,
        jk=JK,
        readfrom='./PROCESSED_DATA'
    )

    mom,p2,E0 = format_energies(E,jk=JK)

    pars,chi2 = fit_dispersion_relation(mom,E0)

    if PLOT:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(6, 4))
        ax = plt.subplot(1,1,1)

        plot_dispersion_relation(ax,mom,p2,E0,fitpar=pars,chi2=chi2)

        ax.legend()
        ax.grid(alpha=0.2)

        ax.set_ylabel(r'$a^2 E^2(\mathbf{p})$')
        # ax.set_xlabel(r'$a|\mathbf{p}|$')
        ax.set_xlabel(r'$a^2\mathbf{p}^2$')

        ax.set_xlim(xmin=-0.1)

        plt.tight_layout()
        plt.show()

    return



if __name__ =="__main__":
    main()