This repo contains the code to analyze MILC data aimed at the study of $B\rightarrow D^{(\star)}l\nu$ data.

# Installation
`to be added`

# Workflow (2pts functions)

### Prerequisites
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