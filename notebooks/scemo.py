import numpy             as np
import gvar              as gv
import matplotlib.pyplot as plt
import pandas            as pd
import lsqfit
import itertools
from tqdm import tqdm

 
from b2heavy.FnalHISQMetadata import params as mData

from b2heavy.ThreePointFunctions.types3pts  import Ratio, RatioIO, ratio_prerequisites, ratio_correction_factor
from b2heavy.ThreePointFunctions.fitter3pts import RatioFitter, phys_energy_priors
from b2heavy.ThreePointFunctions.utils      import read_config_fit, dump_fit_object

 
DATA_DIR = '/Users/pietro/code/data_analysis/BtoD/Alex/'
# DATA_DIR = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/CORRELATORS'

# DATA_2PT = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/report'
DATA_2PT = '/Users/pietro/Desktop/lattice24/0.25/corr2_3'


BINSIZE  = {
    'MediumCoarse':13,
    'Coarse-2':    16,
    'Coarse-1':    11,
    'Coarse-Phys': 19,
    'Fine-1':      16,
    'Fine-Phys':   16,
    'SuperFine':   22
}
momlist = {
    'MediumCoarse':['100','200','300','400'],
    'Coarse-2':    ['100','200','300'],
    'Coarse-1':    ['100','200','300'],
    'Coarse-Phys': ['100','200','300','400'],
    'Fine-1':      ['100','200','300','400'],
    'Fine-Phys':   ['100','200','300','400'],
    'SuperFine':   ['100','200','300','400']
}

smslist = {
    # 'RPLUS'  : ['1S','RW'],  
    'RMINUS' : ['1S','RW'],  
    'QPLUS'  : ['1S','RW'],  
    'XF'     : ['1S','RW'],  
    'XFSTPAR': ['1S','RW'], 
    'R0'     : ['1S','RW'], 
    'R1'     : ['1S','RW'], 
    'XV'     : ['1S','RW'],
    'RA1'    : ['1S'],
}


['XFSTPAR','R0','R1','XV','RA1']


for ENSEMBLE in momlist:
    # for RATIO in smslist:
    for RATIO in ['RA1']:

        if RATIO in ['XFSTPAR','R0','R1','XV','RA1']:
            if ENSEMBLE not in ['Coarse-1','Coarse-2','Coarse-Phys','Fine-1']:
                continue


        SMSLIST  = smslist[RATIO]
        MOMLIST  = momlist[ENSEMBLE]
        
        if RATIO in ['RMINUS','RPLUS','QPLUS','XF']:
            meson = 'D'
        else:
            meson = 'Dst'

        
        TMIN    = 2 if ENSEMBLE=='SuperFine' else 1
        NSTATES = 2

        JK = False
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.size'] = 12

        f, ax = plt.subplots(1,1,figsize=(5,5))

        fits = []
        f0s = []
        for i,mom in enumerate(MOMLIST):
            requisites = ratio_prerequisites(
                ens      = ENSEMBLE,
                ratio    = RATIO,
                mom      = mom,
                readfrom = DATA_2PT,
                jk       = JK
            )

            io = RatioIO(ENSEMBLE,RATIO,mom,PathToDataDir=DATA_DIR)
            robj = RatioFitter(
                io,
                jkBin     = BINSIZE[ENSEMBLE],
                smearing  = SMSLIST,
                # readfrom  = DATA_2PT,
                **requisites
            )

            trange = (TMIN, robj.Ta-TMIN)

            cov_specs = dict(
                diag   = False,
                block  = False,
                scale  = True,
                shrink = True,
                cutsvd = 1E-12
            )

            dE_src = phys_energy_priors(ENSEMBLE,meson,mom,NSTATES,readfrom=DATA_2PT, error=0.5)
            dE_snk = phys_energy_priors(ENSEMBLE,'B'  ,mom,NSTATES,readfrom=DATA_2PT, error=0.5)
            x,ydata = robj.format(trange,flatten=True)
            pr = robj.priors(NSTATES, dE_src=dE_src, dE_snk=dE_snk)

            fit = robj.fit(
                Nstates = NSTATES,
                trange  = trange,
                priors  = pr,
                verbose = False,
                **cov_specs
            )

            res = robj.fit_result(
                Nexc   = NSTATES,
                trange = trange,
                priors = pr 
            )

            fits.append({
                'ensemble': ENSEMBLE,
                'ratio'   : RATIO,
                'momentum': mom,
                'f0'      : fit.p['ratio'][0],
                'chiexp'  : res['chiexp'],
                'pstd'    : res['pstd']
            })


            alpha = 0.2 if mom=='400' else 1.
            robj.plot_fit(ax,NSTATES,trange,color=f'C{i}',color_res=f'C{i}',alpha=alpha,minus=False)

            ax.scatter([],[], marker='o', color=f'C{i}', label=mom)


            f0s.append(fit.p['ratio'][0])

        ax.errorbar([],[],[],fmt='o', ecolor='gray', mfc='w', color='gray', capsize=2.5, label='1S')
        ax.errorbar([],[],[],fmt='^', ecolor='gray', mfc='w', color='gray', capsize=2.5, label='d')

        # ax.set_xlim(-0.5,13)
        # ax.set_ylim(ymax=0.35,ymin=0.)
        ax.grid(alpha=0.2)
        ax.legend()

        ax.set_xlabel(r'$t_{min}/a$')

        ax.set_title(f'{ENSEMBLE} {RATIO}')

        # ax.set_ylabel(r'$Q_+$')


        plt.savefig(f'/Users/pietro/Desktop/ratios/{ENSEMBLE}_{RATIO}_multimom.pdf')

        
        pd.DataFrame(fits).set_index(['ensemble','ratio','momentum'])

        