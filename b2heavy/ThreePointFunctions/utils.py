import tomllib
import gvar as gv
import pickle
import os


def read_config_fit(tag,jk=False,path=None): # tag = 'fit2pt_config_Coarse-1_Dsst_000'
    base = tag if path is None else os.path.join(path,tag)

    if jk: 
        file = f'{base}_jk_fit.pickle'
        with open(file,'rb') as f:
            aux = pickle.load(f)
    else:
        file_fit = f'{base}_fit.pickle'
        file_p   = f'{base}_fit_p.pickle'
        with open(file_fit,'rb') as f:
            fit = pickle.load(f)
        p = gv.load(file_p)
        aux = (fit,p)

    return aux


def dump_fit_object(base,f,**res):
    todump = dict(
        x       = f.x,
        y       = gv.mean(f.y),
        cov     = gv.evalcov(f.y),
        prior   = f.prior,
        # chi2    = f.chi2,
        chi2red = res.get('chi2'), 
        chi2aug = res.get('chi2_aug'),
        chi2exp = res.get('chiexp'),
        pexp    = res.get('pexp'),
        pstd    = res.get('pstd')
    )

    if 'pars' in res:
        todump['pars']    = gv.mean(res.get('pars'))
        todump['parscov'] = gv.evalcov(res.get('pars'))


    with open(f'{base}_fit.pickle','wb') as handle:
        pickle.dump(todump, handle, protocol=pickle.HIGHEST_PROTOCOL)
    gv.dump(f.p,f'{base}_fit_p.pickle')

    return


def exists(path,file):
    file = os.path.join(path,file)
    try:
        assert os.path.isfile(file)
        return True
    except AssertionError:
        raise FileNotFoundError(f'{file} is not a valid file')


def exists_analysis(readfrom,ens,obs,mom,jkfit=False,type='2'):
    tag = f'fit{type}pt_config_{ens}_{obs}_{mom}'
    tag = f'{tag}_jk' if jkfit else tag

    exists(readfrom,f'{tag}_fit.pickle')
    exists(readfrom,f'{tag}_fit_p.pickle')

    return True