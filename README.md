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






<!-- ## Pre-analysis
Before performing the complete jackknife analysis on the different ensembles/meson/momenta, it is advisable to explore the fit in terms of performance, p-values and $\chi^2$. This is dealt with the routine `fit_2pts_preanalysis.py`

It performs the following tasks:
- Perform `tmax` analysis: looks for the timeslice in which the relative error starts to be >30% (percentage tunable through the parameter `maxerr`) 
- Perform the fits for given parameters with different modes (with and without shrinking and rescaling of covariance matrix) and collect the $\chi^2$ and the $p$-values of the fit and prints on screen a table that sums up this information -->

<!-- ### Usage
The possible parameters are
```
$ python fit_2pts_preanalysis.py --help

usage: 
python fit_2pts_preanalysis.py --config   [file location of the toml config file]         
                               --ensemble [list of ensembles analyzed]                    
                               --meson    [list of meson analyzed]                        
                               --mom      [list of momenta considered]
                               --saveto   [where do you want to save? Defaults='./' while 'default' goes to DEFAULT_ANALYSIS_ROOT]                     
                               --maxerr   [Error percentage for Tmax]                     
                               --Nstates  [list of N for (N+N) fit (listed without comas)]
                               --tmins    [list of tmins (listed without commas)]
                               --tmaxs    [list of tmaxs (listed without commas), if not specified, the 30 criterion will be applied]
                               --verbose
```


An example is 
```
$ python fit_2pts_preanalysis.py --ensemble Coarse-1 --meson Dsst --mom 100 --maxerr 25 --Nstates 1 2 3 --tmins 14 --tmaxs 23 --verbose

                                                  Ndof      time  chi2 [red]  chi2 [aug]       p value           E0
tag               tmax tmin Nstates scale shrink                                                                   
Coarse-1_Dsst_100 23   14   1       False False     48  0.030131  169.096940  173.781778  5.255437e-12  1.15189(33)
                                          True      48  0.022429   73.449739   79.323610  4.598948e-02  1.15218(37)
                                    True  False     48  0.029249  100.849462  107.749229  2.519042e-04  1.15237(38)
                                          True      48  0.026455   93.425828  100.269303  1.222464e-03  1.15237(38)
                            2       False False     40  0.464358   69.722886   82.972547  2.210708e-03  1.15192(38)
                                          True      40  0.205435   27.291548   35.422582  9.023864e-01  1.15203(40)
                                    True  False     40  0.114803   36.838366   43.638096  5.409826e-01  1.15210(40)
                                          True      40  0.116863   33.828012   40.554026  6.746804e-01  1.15210(40)
                            3       False False     32  6.493438   54.787533   69.206647  3.905211e-04  1.15185(38)
                                          True      32  0.586021   25.052437   33.885287  3.801422e-01  1.15202(40)
                                    True  False     32  0.735699   34.940914   42.549889  6.453774e-02  1.15210(40)
                                          True      32  0.597975   32.138438   39.600610  1.171063e-01  1.15210(40)
```


## Stability of fit
The routine `fit_2pts_stability_test.py` performs the following tasks:
- Performs different fits for a range of `tmins` and `tmaxs` and excited states
- Plot them together
- Perform the model average, if required to

Some technical observations:
- The fits are performed using *always the same priors* found from effective mass and coefficients of a given time range
  
### Usage

```
$ python 2pts_fit_stability_test.py --help

python 2pts_fit_stability_test.py --config        [file location of the toml config file]
                                  --ensemble     [which ensemble?]                       
                                  --meson        [which meson?]                          
                                  --mom          [which momentum?]                       
                                  --prior_trange  [trange for effective mass priors]
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
python fit_2pts_stability_test.py --ensemble Coarse-1 --meson Dsst --mom 100 --prior_trange 14 23 --Nstates 1 2 3 --tmins 7 8 9 10 11 12 13 14 15 16 --tmaxs 23 --saveto default --plot --showfig --plot_AIC
``` -->