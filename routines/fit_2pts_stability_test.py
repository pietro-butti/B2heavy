# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_stability_test.py --config       [file location of the toml config file]
                                  --ensemble     [which ensemble?]                       
                                  --meson        [which meson?]                          
                                  --mom          [which momentum?]                       
                                  --prior_trange [trange for effective mass priors]
                                  --Nstates      [list of N for (N+N) fit (listed without comas)]
                                  --tmins        [list of tmins (listed without commas)]
                                  --tmaxs        [list of tmaxs (listed without commas)]
                                  --saveto       [where do you want to save the analysis?]
                                  --readfrom    [name of the .pickle file of previous analysis]
                                  --override     [ ]
                                  --not_average  [list of tmins that do not have to be taken in model average]
                                  --show         [do you want to display the plot with plt.show()?]
                                  --plot         [do you want to plot data?]
                                  --plot_ymax    [set maximum y in the plot]
                                  --plot_ymin    [set minimum y in the plot]
                                  --plot_AIC     [do you want to plot also the AIC weight?]
Examples
python 2pts_fit_stability_test.py --ensemble MediumCoarse --meson Dst --mom 000 --Nstates 1 2 3 --tmins  6 7 8 9 10 --tmaxs 15 --plot --show --saveto . --plot_AIC
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT

import pickle
import tomllib
import argparse
import os
import datetime
import jax 
import jax.numpy         as jnp
jax.config.update("jax_enable_x64", True)
import numpy             as np
import gvar              as gv
import pandas            as pd
import matplotlib.pyplot as plt

from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter    import StagFitter

import fit_2pts_utils as utils

def stability_test_fit(ens,meson,mom,data_dir,binsize,smslist,nexcrange,tminrange,tmaxrange,prior_trange,saveto='./',**cov_specs):
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smslist
    )

    effm,effa = stag.meff(prior_trange,**cov_specs)

    aux = {}
    for nstates in nexcrange:
        for tmin in tminrange:
            for tmax in tmaxrange:
                trange = (tmin,tmax)
            
                pr = stag.priors(nstates,Meff=effm,Aeff=effa)
                stag.fit(
                    Nstates = nstates,
                    trange  = trange,
                    priors  = pr,
                    # verbose = True,
                    **cov_specs
                )
                d = stag.fit_result(nstates,trange,priors=pr)

                fit = d['fit']
                aux[nstates,trange] = dict(
                    x       = fit.x,
                    y       = fit.y,
                    p       = fit.p,
                    pvalue  = d['pvalue'],
                    chi2    = fit.chi2,
                    chi2red = d['chi2'],
                    chiexp  = d['chiexp']
                )

    if saveto is not None:
        with open(saveto,'wb') as handle:
            pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'{stag.tmax(threshold=0.3) = }')

    return aux
            
def read_results_stability_test(dumped, ic='TIC', show=True):
    if type(dumped) is str: 
        with open(dumped,'rb') as f:
            d = pickle.load(f)
    else:
        d = dumped

    # Calculate weight normalization
    sumw = sum([np.exp(-(d[k]['chi2red']-2*d[k]['chiexp'])/2) for k in d])

    # Format to dataframe
    ks = ['Nexc','tmin','tmax','chi2/chiexp','pvalue','E_0','weight']
    df = {k: [] for k in ks}
    for k in d:
        nexc,(tmin,tmax) = k
        df['Nexc'].append(nexc)
        df['tmin'].append(tmin)
        df['tmax'].append(tmax)
        df['chi2/chiexp'].append(f'{d[k]["chi2red"]/d[k]["chiexp"]:.2f}')
        df['pvalue'].append(f'{d[k]["pvalue"]:.3f}')
        df['E_0'].append(d[k]['p']['E'][0])
        df['weight'].append(f'{np.exp(-(d[k]["chi2red"]-2*d[k]["chiexp"])/2)/sumw:.1e}')

    df = pd.DataFrame(df)
    df.set_index(['Nexc','tmin','tmax'],inplace=True)

    if show:
        print(df)

    return df

def stability_test_model_average(fits,Nt,not_average,IC='tic'):
    Ws = []
    E0  = []
    for (Nstates,(tmin,tmax)),fit in fits.items():
        if (Nstates,(tmin,tmax)) not in not_average:
            Ncut = Nt//2 - (tmax+1-tmin)
            Npars = 2*Nstates + 2*Nstates*2 + (2*Nstates-2)

            if IC=='aic':
                ic = fit['chi2red'] + 2*Npars + 2*Ncut
            elif IC=='tic':
                ic = fit['chi2red'] - 2*fit['chiexp']

            Ws.append(np.exp(-ic/2))
            E0.append(fit['p']['E'][0])

    E0 = np.array(E0)

    sumw = sum(Ws)
    Ws = np.array(Ws)/sumw
    stat = sum(Ws*E0)
    syst = np.sqrt(gv.mean(sum(Ws*E0*E0) - (sum(Ws*E0))**2))

    return stat,syst,sumw

def stability_test_plot(ax,fits,Nstates,modav,tmax=False):
    pnorm = sum([fit['pvalue'] for fit in fits.values()])/500
    # pnorm = sum([fit['chi2red']/fit['chiexp'] for fit in fits.values()])/500

    # Energy plot for different tmin and Nexc
    for nexc in Nstates:
        E0 = [f['p']['E'][0] for k,f in fits.items() if k[0]==nexc]
        # pvaln = np.array([f['chi2red']/f['chiexp'] for k,f in fits.items() if k[0]==nexc])/pnorm
        pvaln = np.array([f['pvalue'] for k,f in fits.items() if k[0]==nexc])/pnorm
        xplot = np.array([k[1][1 if tmax else 0] for k,f in fits.items() if k[0]==nexc])
        yplot = gv.mean(E0)
        yerr  = gv.sdev(E0)

        ax.errorbar(xplot+(-0.1 + 0.1*(nexc-1)), yplot, yerr=yerr, fmt=',' ,color=f'C{nexc-1}', capsize=2)
        ax.scatter( xplot+(-0.1 + 0.1*(nexc-1)), yplot, marker='s', s=pvaln , facecolors='w', edgecolors=f'C{nexc-1}')
        ax.errorbar([], [], yerr=[], fmt='s' ,color=f'C{nexc-1}', capsize=2, label=f'{nexc}+{nexc}')

        ttitle = np.unique([t0 if tmax else t1 for n,(t0,t1) in fits])[-1]

    # Plot model average
    e0,syst = modav 
    tstr = r'$t_{max}$=' if not tmax else r'$t_{min}$=' + f'{ttitle}'
    ax.axhspan(e0.mean+e0.sdev,e0.mean-e0.sdev,color='gray',alpha=0.2,label=r'Model average of $E_0$ ('+f't={ttitle})')
    ax.axhspan(e0.mean+syst,e0.mean-syst,color='gray',alpha=0.2)

    ax.grid(alpha=0.5)
    ax.set_xlabel(r'$t_{min}/a$' if not tmax else r'$t_{max}/a$')
    ax.set_ylabel(r'$E_0$')
    ax.legend()
    return

def stability_test_plot_AIC(a1,Nt,fits, sum_ws, IC='tic'):
    a1.grid(alpha=0.2)
    a2 = a1.twinx()

    icall = {}
    for (Nstates,trange),fit in fits.items():
        if IC=='aic':
            Ncut  = Nt/2 - (max(trange)+1-min(trange))
            Npars = 2*Nstates + 2*Nstates*2 + (2*Nstates-2)
            ic = fit['chi2'] + 2*Npars + 2*Ncut
        elif IC=='tic':
            ic = fit['chi2red'] - 2*fit['chiexp']

        icall[Nstates,trange] = np.exp(-ic/2) 
    

    for i,nexc in enumerate(sorted(np.unique([n for (n,t) in fits]))):
        x = [min(trange) for (n,trange),f in fits.items() if n==nexc]
        w = [icall[n,t]/sum_ws for (n,t),f in fits.items() if n==nexc]
        a1.scatter(x,w,color=f'C{i}')
        # a1.plot(x,w,alpha=0.2,color=f'C{i}')


        p = [f['pvalue'] for (n,t),f in fits.items() if n==nexc]
        a1.plot(x,p,alpha=0.2,color=f'C{i}')

    a1.set_ylabel(r'$w$')

    a1.scatter([],[],color='gray',label='TIC')
    a1.plot([],[],alpha=0.2,color='gray',label=r'$p$')
    a1.set_ylim(ymin=0,ymax=1)
    a1.legend()

    a2.set_ylabel(r'$p$-value')




def log(tag,ens,meson,mom,prior_trange,Nstates,tmins,tmaxs,not_average):
    st = f'''
# ================================== ({tag}) ==================================
# fit_2pts_stability_test from {__file__} called at {datetime.datetime.now()} with
#        ens          = {ens         }
#        meson        = {meson       }
#        mom          = {mom         }
#        prior_trange = {prior_trange}
#        Nstates      = {Nstates     }
#        tmins        = {tmins       }
#        tmaxs        = {tmaxs       }
#        not_average  = {not_average }
# =============================================================================
''' 

    logg = {
        'prior_trange':prior_trange,
        'Nstates':Nstates,
        'tmins':tmins,
        'tmaxs':tmaxs
    }
    return st, logg



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config', type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson'   , type=str)
prs.add_argument('-mm','--mom'     , type=str)
prs.add_argument('--prior_trange' , type=int, nargs='+')
prs.add_argument('--Nstates' , type=int, nargs='+')
prs.add_argument('--tmins'   , type=int, nargs='+')
prs.add_argument('--tmaxs'   , type=int, nargs='+')

prs.add_argument('--shrink', action='store_true')
prs.add_argument('--scale' , action='store_true')
prs.add_argument('--diag'  , action='store_true')
prs.add_argument('--block' , action='store_true')
prs.add_argument('--svd'   , type=float, default=None)


prs.add_argument('--saveto'  , type=str,  default=None)
prs.add_argument('--readfrom', type=str,  default=None)
prs.add_argument('--override', action='store_true')

prs.add_argument('--not_average', type=int, nargs='+', default=[])

prs.add_argument('--plot',      action='store_true')
prs.add_argument('--plot_ymax', type=float, default=None)
prs.add_argument('--plot_ymin', type=float, default=None)
prs.add_argument('--plot_AIC', action='store_true')
prs.add_argument('--show', action='store_true')











def main():
    args = prs.parse_args()

    ens = args.ensemble
    mes = args.meson   
    mom = args.mom     

    config_file = args.config
    config = utils.load_toml(config_file)
    tag = config['fit'][ens][mes]['mom'][mom]['tag']
    
    # READFROM = f'{DEFAULT_ANALYSIS_ROOT}/fit2pt_stability_test_{tag}.pickle' if args.readfrom=='default' else args.readfrom
    READFROM = DEFAULT_ANALYSIS_ROOT if args.readfrom == 'default' else args.readfrom
    SAVETO   = DEFAULT_ANALYSIS_ROOT if args.saveto   == 'default' else args.saveto
    
    fits = None
    if args.readfrom is not None and not args.override: 
        with open(f'{READFROM}/fit2pt_stability_test_{tag}.pickle','rb') as f:
            fits = pickle.load(f)

    elif os.path.exists(f'{SAVETO}/fit2pt_stability_test_{tag}.pickle') and not args.override:
        # Check if there is an existing former analysis with same specifics
        
        logfile = f'{SAVETO}/fit2pt_stability_test_{tag}_d.log'
        if os.path.exists(logfile):
            with open(logfile,'rb') as f:
                logdata = pickle.load(f)

                if logdata['prior_trange']==args.prior_trange and \
                    logdata['Nstates']==args.Nstates and \
                    logdata['tmins']==args.tmins and \
                    logdata['tmaxs']==args.tmaxs :

                    print(f'Reading from {SAVETO}/fit2pt_stability_test_{tag}.pickle')
                    with open(f'{SAVETO}/fit2pt_stability_test_{tag}.pickle','rb') as f:
                        fits = pickle.load(f)


    if SAVETO is not None and not os.path.isdir(SAVETO):
        raise NameError(f'{SAVETO} is not a directory')
    elif SAVETO is not None:
        saveto = f'{SAVETO}/fit2pt_stability_test_{tag}.pickle'
    else:
        saveto = None



    if fits is None:
        cov_specs = dict(
            shrink = args.shrink ,
            scale  = args.scale  ,
            diag   = args.diag   ,
            block  = args.block  ,
            cutsvd = args.svd   
        )

        fits = stability_test_fit(
            ens          = args.ensemble,
            meson        = args.meson,
            mom          = args.mom,
            data_dir     = config['data'][ens]['data_dir'],
            binsize      = config['data'][ens]['binsize'],
            smslist      = config['fit'][ens][mes]['smlist'],
            prior_trange = tuple(args.prior_trange),
            nexcrange    = args.Nstates,
            tminrange    = args.tmins,
            tmaxrange    = args.tmaxs,
            saveto       = saveto,
            **cov_specs
        ) 

    df = read_results_stability_test(fits, show=True)




    e0,syst,sum_ws = stability_test_model_average(
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
            stability_test_plot_AIC(a1,config['data'][ens]['Nt'],fits,sum_ws)

        saveplot = f'{SAVETO}/fit2pt_stability_test_{tag}.pdf' if args.saveto=='default' else f'{args.saveto}/fit2pt_stability_test_{tag}.pdf'
        plt.savefig(saveplot)

        if args.show:
            plt.show()


    
    # LOG analysis =======================================================================
    if SAVETO is not None:
        string,data = log(tag,ens,mes,mom,args.prior_trange,args.Nstates,args.tmins,args.tmaxs,args.not_average)
        
        logfile = f'{SAVETO}/fit2pt_stability_test_{tag}.log'
        with open(logfile,'w') as f:
            f.write(string)
            f.write(df.to_string())
        
        # logdata = f'{SAVETO}/fit2pt_stability_test_{tag}_d.log'
        # with open(logdata,'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)







if __name__ == "__main__":
    main()
    # read_results_stability_test(
    #     '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/2024-03-12/fit2pt_stability_test_Coarse-1_Dst_000.pickle'
    # )
