import toml
import numpy  as np
import gvar   as gv
import pandas as pd

from matplotlib  import pyplot as plt
from tqdm        import tqdm

from b2heavy.FnalHISQMetadata            import params as mData
from b2heavy.TwoPointFunctions.utils     import correlation_diagnostics

from b2heavy.ThreePointFunctions.fitter3pts import RatioFitter, RatioIO
from b2heavy.ThreePointFunctions.utils     import read_config_fit, dump_fit_object
from b2heavy.ThreePointFunctions.types3pts import ratio_prerequisites, ratiofmt




tmin2 = 0.15
tmin1 = 0.30




def main():
    ens_list = [
        'MediumCoarse',
        'Coarse-2',
        'Coarse-1',
        'Coarse-Phys',
        'Fine-1',
        'Fine-Phys',
        'SuperFine',
    ]
    binSizes  = {
        'MediumCoarse': 13,
        'Coarse-2'    : 16,
        'Coarse-1'    : 11,
        'Coarse-Phys' : 19,
        'Fine-1'      : 16,
        'Fine-Phys'   : 16,
        'SuperFine'   : 22
    }
    mom_list = {
        'MediumCoarse' : ['000','100','200','300','400'],
        'Coarse-2'     : ['000','100','200','300'],
        'Coarse-1'     : ['000','100','200','300'],
        'Coarse-Phys'  : ['000','100','200','300','400'],
        'Fine-1'       : ['000','100','200','300','400'],
        'Fine-Phys'    : ['000','100','200','300','400'],
        'SuperFine'    : ['000','100','200','300','400'],        
    }



    mes = 'D'
    ratio_list = {
        'D': ['xf','r+','r-','q+']
    }

    data_dir = '/Users/pietro/code/data_analysis/BtoD/Alex/'
    frm = '/Users/pietro/code/data_analysis/data/QCDNf2p1stag/B2heavy/report'

    smlist   = ['1S','RW'] 
    # smlist   = ['1S'] 
    config = {'fit': {}}

    aux = []
    for ens in ens_list:
        config['fit'][ens] = {}
        for rstr in ratio_list[mes]:
            ratio = ratiofmt(rstr)
            
            config['fit'][ens][ratio] = {'smlist':smlist, 'mom': {}}

            for mom in mom_list[ens]:
                if mom=='000' and ratio!='RPLUS':
                    continue
                elif mom!='000' and ratio=='RPLUS':
                    continue
                        
                config['fit'][ens][ratio]['mom'][mom] = {}

                print(f'-------- {ens,ratio,mom} --------')
                a_fm = mData(ens)['aSpc'].mean


                req = ratio_prerequisites(ens,ratio,mom,readfrom=frm)
                try:
                    io   = RatioIO(ens,ratio,mom,PathToDataDir=data_dir)
                except NotImplementedError:
                    continue

                robj = RatioFitter(
                    io       = io,
                    jkBin    = binSizes[ens],
                    smearing = smlist,
                    **req
                )

                # choose tmin
                tmin = int(tmin2/a_fm)
                tmax = robj.Ta - tmin

                config['fit'][ens][ratio]['mom'][mom] = {}
                config['fit'][ens][ratio]['mom'][mom]['nstates']    = 1
                config['fit'][ens][ratio]['mom'][mom]['tag']        = f'{ens}_{ratio}_{mom}' # 'Coarse-Phys_B_211'
                config['fit'][ens][ratio]['mom'][mom]['trange']     = [tmin,tmax]

                d = {
                    'ensemble'   : ens,
                    'ratio'      : ratio,
                    'momentum'   : mom,
                    'tmin'       : tmin,
                    'tmax'       : tmax,
                }

                aux.append(d)
    
    df = pd.DataFrame(aux).set_index(['ensemble','ratio','momentum'])
    print(df)


    with open('scemo.toml','w') as f:
        toml.dump(config,f)






if __name__=='__main__':
    main()