# THIS ROUTINE DEALS WITH THE FIRST ANALYSIS OF 2PTS FUNCTIONS
# -------------------------------- usage --------------------------------
usage = '''
python 2pts_fit_config.py --config [file location of the toml config file]         
                   --ensemble [list of ensembles analyzed]                    
                   --meson    [list of meson analyzed]                        
                   --mom      [list of momenta considered]                    
                   --jkfit         [repeat same fit inside each bin. Default: false]               
                   --saveto        [*where* do you want to save files.]             
                   --logto         [Log file name]                                 
                   --override      [do you want to override pre-existing analysis?]
                   --scale         [rescale the covariance matrix with the diagonal]
                   --shrink        [shrink covariance matrix]

                   --plot_eff
                   --plot_fit
                   --plot_show

Examples
python 2pts_fit_config.py --only_ensemble MediumCoarse --only_meson Dsst --only_mom 000 100 --saveto .     
'''

from DEFAULT_ANALYSIS_ROOT import DEFAULT_ANALYSIS_ROOT


import argparse
import pickle
import sys
import tomllib
import os
import gvar as gv

import matplotlib.pyplot as plt
import numpy as np

import datetime

from b2heavy.TwoPointFunctions.types2pts  import CorrelatorIO, Correlator, plot_effective_coeffs
from b2heavy.TwoPointFunctions.fitter import CorrFitter

import fit_2pts_utils as utils


def plot_corr_fit(fitter,Nstates,trange):
    Npol = len(set([pol for (sm,pol) in fitter.keys]))
    Nsmr = len(set([sm for (sm,pol) in fitter.keys]))

    fit = fitter.fits[Nstates,trange]

    yteo = fit.fcn(fit.x,fit.p)
    ydata = (fit.y-gv.mean(yteo))/gv.mean(yteo)
    ydata = np.reshape(ydata,(Npol*Nsmr,len(ydata)//(Npol*Nsmr)))

    xx,yy = fitter.corr.format(smearing=fitter.smearing)
    xtot = list(xx.values())
    ytot = (np.concatenate([yy[k] for k in fitter.keys]) - fit.fcn(xtot,fit.pmean))/fit.fcn(xtot,fit.pmean)
    ytot = np.reshape(ytot,(Npol*Nsmr,len(ytot)//(Npol*Nsmr)))

    res = fit.fcn(xtot,fit.p)
    yshade = gv.sdev( (np.concatenate([gv.mean(yy[k]) for k in fitter.keys])-res)/res)
    yshade = np.reshape(yshade,(Npol*Nsmr,len(yshade)//(Npol*Nsmr)))

    for i,(sm,pol) in enumerate(fitter.keys):
        axi = plt.subplot(Npol,Nsmr,i+1)

        xplot = fit.x[i]
        yplot = gv.mean(ydata[i])
        yerr = gv.sdev( ydata[i])
        axi.scatter(xplot,yplot, marker='o', s=15 ,facecolors='none', edgecolors=f'C{i}')
        axi.errorbar(xplot,yplot, yerr=yerr,fmt=',',color=f'C{i}', capsize=2)

        iok = [i for i,xi in enumerate(xx[sm,pol]) if xi<min(trange) or xi>max(trange)]
        xall = xtot[i][iok]
        yall = ytot[i][iok]
        yaplot = gv.mean(yall)
        yaerr = gv.sdev( yall)
        axi.scatter( xall,yaplot, marker='o', s=15 ,facecolors='none', edgecolors=f'C{i}',alpha=0.2)
        axi.errorbar(xall,yaplot, yerr=yaerr,fmt=',',color=f'C{i}', capsize=2,alpha=0.2)

        axi.fill_between(xall,yshade[i][iok],-yshade[i][iok],color=f'C{i}',alpha=0.3)

        axi.grid(alpha=0.2)

        axi.set_ylim(ymin=-.5,ymax=.5)
        axi.set_xlim(xmin=0.)
        axi.set_title(f'({sm},{pol})')



def fit_2pts_single_corr(
    ens, meson, mom, 
    data_dir, 
    binsize, 
    smslist, 
    nstates, 
    trange, 
    saveto = None, 
    meff   = False, 
    aeff   = False, 
    jkfit  = False,
    scale  = False,
    shrink = False,
):
    """
        This function perform a fit to 2pts correlation function

        Arguments
        ---------

            ens: str
                Ensemble name, e.g. `MediumCoarse`
            meson: str
                Meson name, e.g. `Dsst`
            mom: str
                Momentum string, e.g. `200`
            data_dir: str
                Location of the `Ensemble` directory
            binsize: int
                Bin size for jackknife error
            smslist: list
                List of smearing to be considered
            nstates: int
                Number of states to be fitted. E.g. set to 2 for "2+2" fits.
            saveto: str
                Path of the location where to save the results
            meff: bool
    """
    
    # initialize structures
    io = CorrelatorIO(ens,meson,mom,PathToDataDir=data_dir)
    corr = Correlator(io,jkBin=binsize)

    if meff:
        _,MEFF,_ = corr.EffectiveMass(trange=trange,smearing=smslist)
    if aeff:
       (X,m_eff,a_eff), MEFF, AEFF, Mpr, Apr = corr.EffectiveCoeff(trange,smearing=smslist)


    fitter = CorrFitter(corr,smearing=smslist)
    priors = fitter.set_priors_phys(nstates,Meff=MEFF if meff else None, Aeff=AEFF if aeff else None)
    fit = fitter.fit(
        Nstates           = nstates,
        trange            = trange,
        verbose           = True,
        pval              = True,
        jkfit             = jkfit,
        priors            = priors,
        scale_covariance  = scale,
        shrink_covariance = shrink,
    )


    if saveto is not None:
        name = f'{saveto}_fit.pickle'
        if jkfit:
            with open(name, 'wb') as handle:
                pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            f = fitter.fits[nstates,trange]
            utils.dump_fit_object(f,saveto)

    aux = (
        # (corr,fitter.fits[nstates,trange],fitter.keys),
        fitter,
        ((X,m_eff,a_eff),MEFF,AEFF,Mpr,Apr) if aeff else 0
    )
    return aux

def log(tag,ens,meson,mom,data_dir,smlist,trange,saveto,JKFIT,shrink,scale):
    return f'''
# ================================== ({tag}) ==================================
# fit_2pts_single_corr from {__file__} called at {datetime.datetime.now()} with
#        ens      = {ens}                                                           
#        meson    = {meson}                                                       
#        mom      = {mom}                                                        
#        data_dir = {data_dir}                                                      
#        smlist   = {smlist}                                                      
#        trange   = {trange}                                                       
#        saveto   = {saveto}                                                       
#        jkfit    = {JKFIT}                                                        
#        meff     = True,
#        aeff     = True,                                                           
#        shrink   = {shrink},
#        scale    = {scale}                          
# =============================================================================
''' 



prs = argparse.ArgumentParser(usage=usage)
prs.add_argument('-c','--config'  , type=str,  default='./2pts_fit_config.toml')
prs.add_argument('--ensemble', type=str,  nargs='+',  default=None)
prs.add_argument('--meson'   , type=str,  nargs='+',  default=None)
prs.add_argument('--mom'     , type=str,  nargs='+',  default=None)
prs.add_argument('--saveto'       , type=str,  default=None)
prs.add_argument('--jkfit'        , action='store_true')
prs.add_argument('--override'     , action='store_true')
prs.add_argument('--logto'        , type=str, default=None)
prs.add_argument('--debug'        , action='store_true')
prs.add_argument('--scale'        , action='store_true')
prs.add_argument('--shrink'       , action='store_true')
prs.add_argument('--verbose'       , action='store_true')
prs.add_argument('--plot_eff'       , action='store_true')
prs.add_argument('--plot_fit'       , action='store_true')
prs.add_argument('--show'           , action='store_true')

def main():
    args = prs.parse_args()

    config_file = args.config
    config = utils.load_toml(config_file)

    ENSEMBLE_LIST = args.ensemble if args.ensemble is not None else config['ensemble']['list']
    MESON_LIST    = args.meson    if args.meson    is not None else config['meson']['list']
    MOM_LIST      = args.mom      if args.mom      is not None else []

    print(f'# B2heavy AUTO: ------------- RUNNING 2pt fits for -------------')
    print(f'# B2heavy AUTO: ------------- {ENSEMBLE_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MESON_LIST} -------------')
    print(f'# B2heavy AUTO: ------------- {MOM_LIST} -------------')

    JKFIT  = True if args.jkfit else False

    for ens in ENSEMBLE_LIST:
        for meson in MESON_LIST:
            for mom in (MOM_LIST if MOM_LIST else config['fit'][ens][meson]['mom'].keys()):

                tag = config['fit'][ens][meson]['mom'][mom]['tag']
                data_dir = config['data'][ens]['data_dir']
                binsize  = config['data'][ens]['binsize'] 
                smlist   = config['fit'][ens][meson]['smlist'] 
                nstates  = config['fit'][ens][meson]['nstates'] 
                trange   = tuple(config['fit'][ens][meson]['mom'][mom]['trange']) 

                SAVETO = DEFAULT_ANALYSIS_ROOT if args.saveto=='default' else args.saveto
                saveto = None
                if SAVETO is not None:
                    if not os.path.exists(SAVETO):
                        raise Exception(f'{SAVETO} is not an existing location.')
                    else:
                        if not JKFIT:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}' if args.saveto is not None else None
                        else:
                            saveto = f'{SAVETO}/fit2pt_config_{tag}_jk' if args.saveto is not None else None
                    
                        # Check if there is an already existing analysis =====================================
                        if os.path.exists(f'{saveto}_fit.pickle'):
                            if args.override:
                                print(f'Already existing analysis for {tag}, but overriding...')
                            else:
                                print(f'Analysis for {tag} already up to date')
                                continue


                # Perform analysis ===================================================================
                # try:
                fitter,eff = fit_2pts_single_corr(
                    ens, meson, 
                    mom, 
                    data_dir, 
                    binsize, 
                    smlist, 
                    nstates, 
                    trange, 
                    saveto  = saveto, 
                    jkfit   = JKFIT,
                    meff    = True, 
                    aeff    = True,
                    shrink  = args.shrink,
                    scale   = args.scale,
                )
                # except Exception:
                #     pass

                # LOG analysis and PLOTS =======================================================================
                if SAVETO is not None:
                    logfile = f'{saveto}.log' if args.logto==None else args.logto
                    with open(logfile,'w') as f:
                        f.write(log(tag,ens,meson,mom,data_dir,smlist,trange,saveto,JKFIT,args.shrink,args.scale))

                (X,m_eff,a_eff),MEFF,AEFF,Mpr,Apr = eff
                if args.plot_eff:
                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['font.size'] = 12
                    plt.figure(figsize=(12, 6 if len(np.unique([k[1] for k in X]))==1 else 8))

                    plot_effective_coeffs(trange,X,AEFF,a_eff,Apr,MEFF,m_eff,Mpr,Aknob=1.)

                    plt.tight_layout()
                    if SAVETO is not None:
                        plt.savefig(f'{SAVETO}/PLOTS/fit2pt_config_{tag}_eff.pdf')
                        print(f'Effective mass and coeff plot saved to {SAVETO}/PLOTS/fit2pt_config_{tag}_eff.pdf')
                    if args.show:
                        plt.show()

                if args.plot_fit:
                    plt.rcParams['text.usetex'] = True
                    plt.rcParams['font.size'] = 12
                    plt.figure(figsize=(12, 6))

                    plt.title(tag)
                    plot_corr_fit(fitter,nstates,trange)

                    # plt.tight_layout()
                    if SAVETO is not None:
                        plt.savefig(f'{SAVETO}/PLOTS/fit2pt_config_{tag}_fit.pdf')
                        print(f'Effective mass and coeff plot saved to {SAVETO}/PLOTS/fit2pt_config_{tag}_fit.pdf')
                    if args.show:
                        plt.show()              



if __name__ == "__main__":
    main()


    





    