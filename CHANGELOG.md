# Changelog

The release log for Ax.

## [1.1.0] -- Aug 11, 2025
#### New Features
* New option for the `method` parameter in `client.configure_generation_strategy`:
    `quality` -- allows uers to indicate they would like Ax to generate the highest
    quality candidates it is able to at the expense of slower runtime (#4042)
* New logic for deciding which analyses to produce by default in
    `client.compute_analyses` (#4013)
* New parameters in `client.summarize` allow users to filter their summary by trial
    index and/or trial status (#4012, #4118)

#### Bug Fixes
* Allow `client.summarize` to be called without a `GenerationStrategy` being set
    (i.e. before `client.configure_generation_strategy` or `client.get_next_trails`
    has been called.) (#3801)
* Fixed incorrect grouping in `TopSurfacesAnalysis` (#4095)
* Fixed `ContourPlot` failing to compute in certain search spaces with parameter
    constraints (#4124)
* Misc. plotting fixes and improvements

#### Other changes
* Bumped pinned [botorch](https://github.com/pytorch/botorch) version to 0.15.1
* Performance improvements in `SensitivityAnalysis` (#3891)
* Improved optimization performance in constrained optimization settings (#3585)
* Augmented logging in `Client`, early stopping module (#4044, #4108)
