This repo contains the code to analyze MILC data aimed at the study of $B\rightarrow D^{(\star)}l\nu$ data.

# Installation
`to be added`

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

- Set location of the data and some general information in `routines/2pts_fit_config.toml`. This file contains general informations on how to perform the analysis (data location, fitting ranges, specifics of the fit). At this preliminary stage, the user should modify it at the field `data`, i.e.
```
[data]
    [data.MediumCoarse]
        name     = 'MediumCoarse'
        data_dir = "/Users/pietro/code/data_analysis/BtoD/Alex"
        binsize  = 13
        mom_list = ['000','100','200','300','400','110','211','222']
        Nt       = 48
```
and set here, in order: the ensemble name, the location of the folder `Ensemble`, the size of the bin for jackknife, the list of available momentum, the number of total timeslices.


## Pre-analysis
Before performing the complete jackknife analysis on the different ensembles/meson/momenta, it is advisable to explore the fit in terms of performance, p-values and $\chi^2$. This is dealt with the routine `fit_2pts_preanalysis.py`

It performs the following tasks:
- Perform `tmax` analysis: looks for the timeslice in which the relative error starts to be >30% (percentage tunable through the parameter `maxerr`) 
- Perform the fits for given parameters with different modes (with and without shrinking and rescaling of covariance matrix) and collect the $\chi^2$ and the $p$-values of the fit and prints on screen a table that sums up this information

### Usage
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
