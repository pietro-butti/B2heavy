This repo contains the code to analyze MILC data aimed at the study of $B\rightarrow D^{(\star)}l\nu$ data.

# Installation
The package is ensured to work with a Python version $\geq$ 3.11.

The following list of packages are required for the installation:
- `numpy`
- `pandas`
- `gvar`
- `lsqfit`
- `jax`
- ...

### Instruction
Once the listed packages are installed in the working evironment, it should be enough to hit
```
pip install /path/to/B2heavy
```

# Workflow (2pts functions)

## Prerequisites
First of all, the user must set:
- The *default analysis folder* in `routines/DEFAULT_ANALYSIS_ROOT.py`. This is going to be the location where all the outputs of the results will be directed to if `--saveto` and `--readfrom` flags will be set to `default`. 
    
    > `--saveto` and `--readfrom` flags are present in almost every routines. They are set by default to `./`, i.e. the directory where the routine is run from.
    Some routines depends on the output of others, so is always advisable to set 
    > ```
    > --saveto default   --readfrom default 
    > ```
    > such that the code will always search for existing analysis and save the output in `DEFAULT_ANALYSIS_ROOT` and avoid repeating the analysis.

- Set location of the data and some general information in `routines/2pts_fit_config.toml`. This file contains general information on how to perform the analysis (data location, fitting ranges, specifics of the fit). At this preliminary stage, the user should modify it at the field `data`, i.e.
```
[data]
    [data.MediumCoarse]
        name     = 'MediumCoarse'
        data_dir = "/path/to/ensemble/dir/"
        binsize  = 13
        mom_list = ['000','100','200','300','400','110']
        Nt       = 48
```
and set here, in order: 
- the ensemble name 
- the location of the folder `Ensemble` 
- the size of the bin for jackknife, the list of available momentum
- the number of total timeslices.


> We remind that the folder `Ensemble` is expected to have the following structure
> ```
>Ensembles
>└── FnalHISQ
>    ├── a0.057
>    │   └── l96192f211b672m0008m022m260-HISQscript.hdf5
>    ├── a0.088
>    │   ├── l4896f211b630m00363m0363m430-HISQscript.hdf5
>    │   └── l6496f211b630m0012m0363m432-HISQscript.hdf5
>    ├── a0.12
>    │   ├── l2464f211b600m0102m0509m635-HISQscript.hdf5
>    │   ├── l3264f211b600m00507m0507m628-HISQscript.hdf5
>    │   ├── l4864f211b600m001907m05252m6382-HISQscript.hdf5
>    └── a0.15
>        ├── l3248f211b580m002426m06730m8447-HISQscript.hdf5
>```

Names of the ensembles and relative files must correspond to the metaparameters contained in `FnalHISQMetadata.py` file. A default version can be found in `/b2heavy/FnalHISQMetadata.py`. In case this needs to be changed, this file can be replaced, but must have the same location.


## 2-points function fits
The fits of the 2-points functions are automatically taken care of by the routine `/routines/fit_2pts_config.py`. Here we explain how to use it.

This routine is thought to perform the fit in an _automatic_ and _sequential_ fashion. The user is expected to gather all the relevant parameters inside the aforementioned "config" file (like `/routines/2pts_fit_config.toml`) in the following way.

The config file is expected to have the following structure for _each one_ of the fit that wants to be performed.
```
fit
└── Fine-Phys
    └── Dst
        └── mom
            ├── 000
            │   ├── tag        = 'Fine-Phys_Dst_000'
            │   ├── nstates    = 3
            │   ├── trange_eff = [ 15, 37]
            │   ├── trange     = [ 7, 37]
            │   └── svd        = 1e-11
            └── 100
                ├── tag        = 'Fine-Phys_Dst_100'
                ├── nstates    = 3
                ├── trange_eff = [ 15, 33]
                ├── trange     = [ 7, 33]
                └── svd        = 0.0185410214733366
```
The specific values of the parameters are considered user-inputs, and therefore they are not inferred automatically in any way.

Once this file has been prepared, the routine can be run in the following way

```
$ python 2pts_fit_config.py --config   [file location of the toml config file. Default: ./2pts_fit_config.toml]         
                            --ensemble [list of ensembles analyzed]                    
                            --meson    [list of meson analyzed]                        
                            --mom      [list of momenta considered]  

                            --jkfit    [repeat same fit inside each bin. Default: false]               
                            --saveto   [where do you want to save files. Default: ./]             
                            --logto    [Log file name]                                 
                            --override [do you want to override pre-existing analysis?]
       
                            --diag     [Default: False]
                            --block    [Default: False] 
                            --scale    [Default: False]    
                            --shrink   [Default: False]  
                            --svd      [Default: None] 
                          
                            --no_priors_chi [don't consider priors in the calculation of expected chi2. Default: False]
       
                            --plot_eff [Default: False]
                            --plot_fit [Default: False]
                            --show     [Default: False]
```  
> In principle, there are no mandatory flags.

> Note that specifics of for the treatment of the covariance matrix are given as flags and they are not considered user inputs. The only exception is `svd`: if one value is indicated after the flag, it _overrides the input in the `toml` file_


> The flags `ensemble`, `meson` and `mom` can be omitted, in that case, the list of fits is inferred from the `toml` config file. Elsewise, a list of ensemble, mesons and momenta can be indicated after the respective flag, using the standard nomenclature, without comas.

> The flag `saveto` is not mandatory, but if specified must be followed by the path of the folder where the user wants to direct the output of the fit.


## Output format
As an example, let's say that we performed the 2pt function fit for the following specifics:
- ensemble: `Fine-1`
- meson: `Dst`
- momentum: `100`
using the routine `/routines/fit_2pts_config.py`.

In this case, inside the output folder we will find the following data files
```
fit2pt_config_Fine-1_Dst_100_fit.pickle   
fit2pt_config_Fine-1_Dst_100_fit_p.pickle
```
In order to have access to the result stored, one can use the function `b2heavy.ThreePointFunctions.utils.read_config_fit` in the following way
```
from b2heavy.ThreePointFunctions.utils import read_config_fit

tag       = 'fit2pt_config_Fine-1_Dst_100'
readfrom  = '/path/to/output/folder/'
fit,pars  = read_config_fit(tag,path=readfrom)
```
The variable `fit` is a dictionary which contains the following data about the fit:
- `fit['x']` - x-data used by `lsqfit`
- `fit['y']` - y-data used by `lsqfit`
- `fit['cov']` - covariance used by `lsqfit` after the manipulation
- `fit['chi2red']` - $\chi^2$ (not reduced)
- `fit['chi2aug']` - $\chi^2$ augmented ($\chi^2 + \chi^2_\text{prior}$)
- `fit['chiexp']` - expected $\chi^2$
- `fit['pexp']` - expected $p$-value
- `fit['pstd']` - finite size corrected $p$-value

The variable `pars` is a dictionary containing all the parameters computed by the fit.

>As an example, one can extrac the energy of the fundamental state as
> ```
> E0 = par['dE'][0]
> ```
> or the matrix elements $Z_{1S}(\mathbf{p}_\perp)$ as
> ```
> z = par['Z.1S.Bot'][0]
>```
> To obtain the real value of $Z_{1S}(\mathbf{p}_\perp)$, one has to elaborate the fit parameter `z` as $\texttt{z} = \frac{\sqrt{Z_{1S}(\mathbf{p}_\perp)}}{2E_0}$ and therefore 
>```
> Z = np.exp(z)**2 * 2*E0
>```
> .
### Jackknife fits
If the routine `/routines/fit_2pts_config.py` has been called with the flag `--jkfit` in the output folder one will also find
```
fit2pt_config_Fine-1_Dst_100_jk_fit.pickle   
fit2pt_config_Fine-1_Dst_100_jk_fit_p.pickle
```
In this case, one can read the results in the same way, but setting the extra keyword `jk=True`
```
from b2heavy.ThreePointFunctions.utils import read_config_fit

tag       = 'fit2pt_config_Fine-1_Dst_100'
readfrom  = '/path/to/output/folder/'
fitjk     = read_config_fit(tag,path=readfrom,jk=True)
```
The output will be different. `fitjk` is a dictionary containing the values of the parameter obtained for each jackknife fit. 
