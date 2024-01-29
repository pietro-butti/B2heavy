# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_stability_test.py --config        [file location of the toml config file]
                                  --ensemble     [which ensemble?]                       
                                  --meson        [which meson?]                          
                                  --mom          [which momentum?]                       
                                  --Nstates      [list of N for (N+N) fit (listed without comas)]
                                  --tmins        [list of tmins (listed without commas)]
                                  --tmaxs        [list of tmaxs (listed without commas)]
                                  --read_from    [name of the .pickle file of previous analysis]
                                  --saveto       [where do you want to save the analysis?]
                                  --not_average  [list of tmins that do not have to be taken in model average]
                                  --showfig      [do you want to display the plot with plt.show()?]
                                  --plot         [do you want to plot data?]
                                  --plot_ymax    [set maximum y in the plot]
                                  --plot_ymin    [set minimum y in the plot]
                                  --plot_AIC     [do you want to plot also the AIC weight?]
Examples
python 2pts_fit_stability_test.py --ensemble MediumCoarse --meson Dst --mom 000 --Nstates 1 2 3 --tmins  6 7 8 9 10 --tmaxs 15 --plot --showfig --saveto . --plot_AIC
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import pickle
import numpy as np
import gvar  as gv
import tomllib
import argparse
import os
import matplotlib.pyplot as plt

from b2heavy.TwoPointFunctions.types2pts  import CorrelatorIO, Correlator
from b2heavy.TwoPointFunctions.fitter import CorrFitter


def stability_test_fit(ens,meson,mom,data_dir,binsize,smslist,nexcrange,tminrange,tmaxrange,saveto='./'):
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr = Correlator(io)
    corr.jack(binsize)

    fitter = CorrFitter(corr, smearing=smslist)

    for nstates in nexcrange:
        for tmin in tminrange:
            for tmax in tmaxrange:
                trange = (tmin,tmax)

                _,MEFF,_ = corr.EffectiveMass(trange=trange,smearing=smslist)

                try:
                    fitter.fit(
                        Nstates=nstates,
                        trange=trange,
                        verbose=True,
                        pval=True,
                        priors = fitter.set_priors_phys(nstates,Meff=MEFF)
                    )
                except ValueError:
                    print(nstates,tmin)

    aux = {}
    for k,fit in fitter.fits.items():
        aux[k] = dict(
            x       = fit.x,
            y       = fit.y,
            p       = fit.p,
            pvalue  = fit.pvalue,
            chi2    = fit.chi2,
            chi2red = fit.chi2red
        )

    with open(saveto,'wb') as handle:
        pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return aux
            
def stability_test_model_average(fits,Nt,not_average):
    Ws = []
    E0  = []
    for Nstates,(tmin,tmax) in fits:
        if tmin not in not_average:
            Ncut = Nt//2 - (tmax+1-tmin)
            Npars = 2*Nstates + 2*Nstates*2 + (2*Nstates-2)
            ic = fits[Nstates,(tmin,tmax)]['chi2'] + 2*Npars + 2*Ncut
            Ws.append(np.exp(-ic/2))
            E0.append(fits[Nstates,(tmin,tmax)]['p']['E'][0])
    E0 = np.array(E0)
    Ws = np.array(Ws)/sum(Ws)

    stat = sum(Ws*E0)
    syst = np.sqrt(gv.mean(sum(Ws*E0*E0) - (sum(Ws*E0))**2))

    return stat,syst

def stability_test_plot(ax,fits,Nstates,modav):
    pnorm = sum([fit['pvalue'] for fit in fits.values()])

    # Energy plot for different tmin and Nexc
    # for nexc in np.arange(1,Nmax+1):
    for nexc in Nstates:
        E0 = [f['p']['E'][0] for k,f in fits.items() if k[0]==nexc]
        pvaln = [f['pvalue'] for k,f in fits.items() if k[0]==nexc]/pnorm*500
        xplot = np.array([k[1][0] for k,f in fits.items() if k[0]==nexc])
        yplot = gv.mean(E0)
        yerr  = gv.sdev(E0)

        ax.scatter( xplot+(-0.1 + 0.1*(nexc-1)), yplot, marker='s', s=pvaln ,facecolors='none', edgecolors=f'C{nexc-1}')
        ax.errorbar(xplot+(-0.1 + 0.1*(nexc-1)), yplot, yerr=yerr, fmt=',' ,color=f'C{nexc-1}', capsize=2)
        ax.errorbar([], [], yerr=[], fmt='s' ,color=f'C{nexc-1}', capsize=2, label=f'{nexc}+{nexc}')
    
    # Plot model average
    e0,syst = modav 
    ax.axhspan(e0.mean+e0.sdev,e0.mean-e0.sdev,color='gray',alpha=0.2,label=r'Model average of $E_0$')
    ax.axhspan(e0.mean+syst,e0.mean-syst,color='gray',alpha=0.2)

    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$t_{min}/a$')
    ax.set_ylabel(r'$E_0$')
    ax.legend()

    # ax.title(title)

    return

def stability_test_plot_AIC(a1,Nt,fits):
    a1.grid(alpha=0.2)

    icall = {}
    for (Nstates,trange),fit in fits.items():
        Ncut  = Nt/2 - (max(trange)+1-min(trange))
        Npars = 2*Nstates + 2*Nstates*2 + (2*Nstates-2)
        ic = fit['chi2'] + 2*Npars + 2*Ncut
        icall[Nstates,trange] = np.exp(-ic/2) 

    norm = sum(list(icall.values()))

    for nexc in [1,2,3]:
        x = [min(trange) for (n,trange),f in fits.items() if n==nexc]
        w = [icall[n,t]/norm for (n,t),f in fits.items() if n==nexc]
        a1.scatter(x,w)
        a1.plot(x,w,alpha=0.2)

    a1.set_ylim(ymin=0,ymax=1)


def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config', type=str,  default='./2pts_fit_config.toml')
prs.add_argument('--ensemble', type=str)
prs.add_argument('--meson'   , type=str)
prs.add_argument('--mom'     , type=str)
prs.add_argument('--Nstates' , type=int, nargs='+')
prs.add_argument('--tmins'   , type=int, nargs='+')
prs.add_argument('--tmaxs'   , type=int, nargs='+')

prs.add_argument('--saveto'   , type=str,  default=None)
prs.add_argument('--read_from', type=str,  default=None)

prs.add_argument('--not_average', type=int, nargs='+', default=[])

prs.add_argument('--plot',      action='store_true')
prs.add_argument('--plot_ymax', type=float, default=None)
prs.add_argument('--plot_ymin', type=float, default=None)
prs.add_argument('--plot_AIC', action='store_true')
prs.add_argument('--showfig', action='store_true')

def main():
    args = prs.parse_args()

    ens = args.ensemble
    mes = args.meson   
    mom = args.mom     

    config_file = args.config
    config = load_toml(config_file)
    tag = config['fit'][ens][mes]['mom'][mom]['tag']
    

    READFROM = f'{DEFAULT_ANALYSIS_ROOT}/fit2pts_stability_test_{tag}.pickle' if args.read_from=='default' else args.read_from
    SAVETO = DEFAULT_ANALYSIS_ROOT if args.saveto=='default' else args.saveto
    if args.read_from is not None: 
        with open(READFROM,'rb') as f:
            fits = pickle.load(f)

    # Check if there is an existing former analysis
    elif os.path.exists(f'{SAVETO}/fit2pt_stability_test_{tag}.pickle'):
        print(f'Reading from {SAVETO}/fit2pt_stability_test_{tag}.pickle')
        with open(f'{SAVETO}/fit2pt_stability_test_{tag}.pickle','rb') as f:
            fits = pickle.load(f)
    
    else:
        if not os.path.isdir(SAVETO):
            raise NameError(f'{SAVETO} is not a directory')
        else:
            saveto = f'{SAVETO}/fit2pt_stability_test_{tag}.pickle'
        
        fits = stability_test_fit(
            ens       = args.ensemble,
            meson     = args.meson,
            mom       = args.mom,
            data_dir  = config['data'][ens]['data_dir'],
            binsize   = config['data'][ens]['binsize'],
            smslist   = config['fit'][ens][mes]['smlist'],
            nexcrange = args.Nstates,
            tminrange = args.tmins,
            tmaxrange = args.tmaxs,
            saveto    = saveto
        ) 


    e0,syst = stability_test_model_average(
        fits, config['data'][ens]['Nt'], args.not_average
    )


    if args.plot:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12

        if not args.plot_AIC:
            f, ax = plt.subplots(1, 1)
        else:
            f, (ax, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

        states = np.unique([n for n,(mi,ma) in fits])
        stability_test_plot(ax,fits,states,[e0,syst])
        ax.set_ylim(ymax=args.plot_ymax,ymin=args.plot_ymin)
        ax.set_title(tag)


        if args.plot_AIC:
            stability_test_plot_AIC(a1,config['data'][ens]['Nt'],fits)

        saveplot = f'{SAVETO}/PLOTS/fit2pt_stability_test_{tag}.pdf' if args.saveto=='default' else f'{args.saveto}/PLOTS/fit2pt_stability_test_{tag}.pdf'
        plt.savefig(saveplot)

        if args.showfig:
            plt.show()
        







if __name__ == "__main__":
    main()

