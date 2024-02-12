import tomllib
import gvar as gv
import pickle
import os

def load_toml(file) -> dict:
    with open(file,'rb') as f:
        toml_data: dict = tomllib.load(f)
    return toml_data

def dump_fit_object(fit,base):
    aux = dict(
        x       = fit.x,
        priors  = fit.prior,
        pvalue  = fit.pvalue,
        chi2    = fit.chi2,
        chi2red = fit.chi2red,
    )
    with open(f'{base}_fit.pickle','wb') as handle:
        pickle.dump(aux, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gv.dump(fit.y,f'{base}_y.pickle')
    gv.dump(fit.p,f'{base}_p.pickle')

    return

def read_config_fit(tag,jk=False,path=None): # tag = 'fit2pt_config_Coarse-1_Dsst_000'
    base = tag if path is None else os.path.join(path,tag)

    if jk: 
        file = f'{base}_jk_fit.pickle'
        with open(file,'rb') as f:
            aux = pickle.load(f)
    else:
        file_fit = f'{base}_fit.pickle'
        with open(file_fit,'rb') as f:
            fit = pickle.load(f)
        
        file_p   = f'{base}_p.pickle'
        p = gv.load(file_p)
        
        file_y   = f'{base}_y.pickle'
        y = gv.load(file_y)

        aux = (fit,p,y)

    return aux

def Ndof(Npol,trange,nexc,Nsmr=2):
    Npar = 2*nexc + 2*nexc + 2*nexc + 2*(nexc-1)
    Npoints = Npol*(Nsmr*(Nsmr+1)//2)*(max(trange)-min(trange))
    return Npoints-Npar
