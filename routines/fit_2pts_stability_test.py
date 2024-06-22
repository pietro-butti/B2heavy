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
                                  --obs          [which observable to observe]
                                  --diag         [True/False] default is False
                                  --block        [True/False] default is False
                                  --shrink       [True/False] default is False
                                  --scale        [True/False] default is False
                                  --svd          [True/None/float]  default is None
                                  --nochipr      [compute chi2exp with priors?]
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
from b2heavy.TwoPointFunctions.types2pts import CorrelatorIO, plot_effective_coeffs, find_eps_cut
from b2heavy.TwoPointFunctions.fitter    import StagFitter

from b2heavy.FnalHISQMetadata            import params

from b2heavy.ThreePointFunctions.utils import read_config_fit, dump_fit_object
import fit_2pts_utils as utils

def stability_test_fit(ens,meson,mom,data_dir,binsize,smslist,nexcrange,tminrange,tmaxrange,prior_trange,chipr=True,saveto='./',**cov_specs):
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    stag = StagFitter(
        io       = io,
        jkBin    = binsize,
        smearing = smslist
    )

    svd = cov_specs.get('cutsvd')
    if isinstance(svd,float) or svd is None:
        find_epsilon = False
    else:
        find_epsilon = True

    effm,effa = stag.meff(prior_trange,**cov_specs)

    aux = {}
    for nstates in nexcrange:
        for tmin in tminrange:
            for tmax in tmaxrange:
                if np.floor(tmax)==tmax:
                    tmax = int(tmax)
                
                trange = (tmin,tmax)

                if find_epsilon:
                    covs = {k: cov_specs[k] for k in cov_specs}
                    covs['cutsvd'] = None
                    eps = find_eps_cut(stag,trange,**covs)
                    cov_specs['cutsvd'] = eps

            
                pr = stag.priors(nstates,Meff=effm,Aeff=effa)
                stag.fit(
                    Nstates = nstates,
                    trange  = trange,
                    priors  = pr,
                    # verbose = True,
                    **cov_specs
                )
                d = stag.fit_result(nstates,trange,priors=pr if chipr else None)

                fit = d['fit']
                aux[nstates,trange] = dict(
                    x       = fit.x,
                    y       = fit.y,
                    p       = fit.p,
                    pstd    = d['pstd'],
                    pexp    = d['pexp'],
                    chi2red = d['chi2red'],
                    chi2aug = d['chi2aug'],
                    chiexp  = d['chiexp']
                )

    if saveto is not None:
        with open(saveto,'wb') as handle:
            pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'{stag.tmax(threshold=0.3) = }')

    return aux


def read_results_stability_test(dumped, Nt=30, show=True, obs='dE', n_states=0):
    if type(dumped) is str: 
        with open(dumped,'rb') as f:
            dd = pickle.load(f)
    else:
        dd = dumped

    # Calculate weight normalization
    sumw = sum([np.exp(-(dd[k]['chi2aug']-2*dd[k]['chiexp'])/2) for k in dd])

    # How many polarization are there?
    ls = [[k.split('.')[-1] for k in dd[fk]['p'] if k.startswith('Z') and not k.endswith('o')] for fk in dd]
    ls = np.unique(np.concatenate(ls))
    unpol = True if len(ls)==1 else False
    Zs = ['Z.1S.Unpol','Z.d.Unpol'] if unpol else ['Z.1S.Par','Z.1S.Bot','Z.d.Par','Z.d.Bot']

    df = []
    for k in dd:
        nexc,(tmin,tmax) = k

        e0 = dd[k]['p'][obs][n_states]

        chiexp  = dd[k]['chiexp']
        chi2aug = dd[k]['chi2aug']

        tic = chi2aug - 2*chiexp

        ncut  = Nt - (tmax-tmin)
        npars = len(np.concatenate([v for k,v in dd[k]['p'].items()]))
        aic = chi2aug + 2*ncut + 2*npars

        tmp = {
                'Nexc': nexc,
                'trange': (tmin,tmax),
                'E0': e0,
                'chiexp': chiexp,
                'chi2aug': chi2aug,
                'aug/exp': chi2aug/chiexp,
                'TIC': np.exp(-tic/2)/sumw,
                'AIC': np.exp(-aic/2),
                'pexp': dd[k]['pexp'],
                'pstd': dd[k]['pstd'],
            }

        for zk in Zs:
            z = np.exp(dd[k]['p'][zk][0])**2 * (2*e0)
            tmp[zk] = z

        df.append(tmp)

    df = pd.DataFrame(df)
    df.set_index(['Nexc','trange'],inplace=True)
    df['AIC'] /= df['AIC'].sum()


    if show:
        print(df)

    return df


def stability_test_plot(ax, df, IC='TIC', label_mod_av='mod. av.', obs='E0', afm=1.):
    Nstates = np.unique([n for n,tr in df.index])

    # Plot observable
    # obs = df.columns[0]
    for nexc in Nstates:
        idx = [i for i in df.index if i[0]==nexc]

        xplot = [i[1][0] for i in idx]
        yplot = gv.mean(df[df.index.isin(idx)][obs].values)
        yerr  = gv.sdev(df[df.index.isin(idx)][obs].values)

        ax.errorbar((xplot+(-0.1 + 0.1*(nexc-1)))*(afm), yplot, yerr=yerr, fmt='o' ,color=f'C{nexc-1}', capsize=2)
        # ax.scatter( xplot+(-0.1 + 0.1*(nexc-1)), yplot, marker='o', facecolors='w', edgecolors=f'C{nexc-1}')
        ax.errorbar([], [], yerr=[], fmt='o' ,color=f'C{nexc-1}', capsize=2, label=f'{nexc}+{nexc}')

    # Plot model average
    ws = df[IC]
    modav = (ws * df[obs]).sum()
    # ax.axhline(modav.mean, color='gray', alpha=0.2)
    ax.axhspan(modav.mean-modav.sdev, modav.mean+modav.sdev, color='gray', alpha=0.1, label=label_mod_av)
    syst = np.sqrt((ws * df['E0']**2).sum() - modav**2).mean
    ax.axhline(modav.mean+modav.sdev+syst, color='gray', linestyle=':', alpha=0.1)
    ax.axhline(modav.mean-modav.sdev-syst, color='gray', linestyle=':', alpha=0.1)

    # Plot other model average
    IC2 = 'AIC' if IC=='TIC' else 'TIC'
    ws2 = df[IC2]
    modav2 = (ws2 * df[obs]).sum()    
    ax.axhline(modav2.mean, color='tan', alpha=0.2, linestyle='--', label=f'{IC2} mod. av.')

    return


def stability_test_plot_AIC(ax, df, IC='TIC', pkey='pstd', legend=True, afm=1., **kwargs):
    Nstates = np.unique([n for n,tr in df.index])

    ax2 = ax.twinx()

    for nexc in Nstates:
        idx = [i for i in df.index if i[0]==nexc]
        xplot = [i[1][0] for i in idx]

        # IC
        yplot1 = df[df.index.isin(idx)][IC]
        ax.plot(np.array(xplot)*afm, yplot1, color=f'C{nexc-1}', alpha=0.35)
        
        # pvalue
        yplot2 = df[df.index.isin(idx)][pkey]
        ax2.scatter(np.array(xplot)*afm, yplot2, color=f'C{nexc-1}')
        ax2.plot(np.array(xplot)*afm, yplot2, color=f'C{nexc-1}',alpha=0.1,linestyle="--")

    if legend:
        ax.scatter([], [], color='gray', label=pkey)
        ax.plot([], [], color=f'gray', alpha=0.35, label=IC)
        ax.legend()

    
    return ax2


def log(tag,ens,meson,mom,prior_trange,Nstates,tmins,tmaxs):
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
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('-e','--ensemble', type=str)
prs.add_argument('-m','--meson'   , type=str)
prs.add_argument('-mm','--mom'    , type=str)

prs.add_argument('--prior_trange' , type=int, nargs='+')
prs.add_argument('--Nstates'      , type=int, nargs='+')
prs.add_argument('--tmins'        , type=int, nargs='+')
prs.add_argument('--tmaxs'        , type=float, nargs='+')
prs.add_argument('--obs'          , type=str, default='E0')

prs.add_argument('--shrink', action='store_true')
prs.add_argument('--scale' , action='store_true')
prs.add_argument('--diag'  , action='store_true')
prs.add_argument('--block' , action='store_true')
prs.add_argument('--svd'   , type=float, default=None)

prs.add_argument('--nochipr', action='store_false')
# prs.add_argument('--not_average', type=int, nargs='+', default=[])


prs.add_argument('--saveto'  , type=str,  default=None)
prs.add_argument('--readfrom', type=str,  default=None)
prs.add_argument('--override', action='store_true')


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
        try:
            with open(f'{READFROM}/fit2pt_stability_test_{tag}.pickle','rb') as f:
                fits = pickle.load(f)
        except FileNotFoundError:
            FileNotFoundError(f'fit2pt_stability_test_{tag}.pickle cannot be found in {args.readfrom}')

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
            chipr        = args.nochipr,
            **cov_specs
        ) 

    df = read_results_stability_test(
        fits, 
        show     = True, 
        Nt       = config['data'][ens]['Nt'],
        n_states = 0
    )


    if args.plot:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12

        if not args.plot_AIC:
            f, ax = plt.subplots(1, 1)
        else:
            # f, (ax, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=(6,6), sharex=True)
            f, (ax, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(6,6), sharex=True)


        alat = params(ens)['aSpc'].mean

        stability_test_plot(ax,df, obs=args.obs, afm=1)
        ax.set_ylim(ymax=args.plot_ymax,ymin=args.plot_ymin)
        ax.set_title(tag)
        ax.grid(alpha=0.2)

        ax.axvline(config['fit'][ens][mes]['mom'][mom]['trange'][0],color='gray',alpha=0.6,linestyle=":")

        ax.set_ylabel(r'$E_0$' if args.obs=='dE' else '')
        ax.legend()
        # ax.set_xlabel(r'$t_{min}/a$')


        if args.plot_AIC:
            twin_ax = stability_test_plot_AIC(a1,df,IC='TIC',pkey='pstd', legend=True)
            a1.grid(alpha=0.2)
            a1.set_ylabel(r'$w$')
            twin_ax.set_ylabel(r'$p$-value')

            a1.set_xlabel(r'$t_{min}$ [fm]')

            xtk = np.array(a1.get_xticks())
            ntk = [f'{x:.2f}' for x in xtk*alat]
            a1.set_xticks(xtk, labels=ntk)

            # breakpoint()

        else:
            ax.set_xlabel(r'$t_{min}$ [fm]')

            # twin_ax = stability_test_plot_AIC(a2,df,IC='AIC',pkey='pexp', legend=True)
            # a2.grid(alpha=0.2)
            # a2.set_ylabel(r'$w$')
            # twin_ax.set_ylabel(r'$p$-value')

        plt.tight_layout()
        saveplot = f'{SAVETO}/fit2pt_stability_test_{tag}.pdf' if args.saveto=='default' else f'{args.saveto}/fit2pt_stability_test_{tag}.pdf'
        plt.savefig(saveplot)

        if args.show:
            plt.show()


    
    # LOG analysis =======================================================================
    if SAVETO is not None:
        string,data = log(tag,ens,mes,mom,args.prior_trange,args.Nstates,args.tmins,args.tmaxs)
        
        logfile = f'{SAVETO}/fit2pt_stability_test_{tag}.log'
        with open(logfile,'w') as f:
            f.write(string)
            f.write(df.to_string())
        
        logdata = f'{SAVETO}/fit2pt_stability_test_{tag}_d.log'
        with open(logdata,'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)







if __name__ == "__main__":
    main()
