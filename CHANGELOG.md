# Changelog

The release log for Ax.
## [1.2.0] -- Oct 24, 2025
#### New features
* `DerivedParameterConfig` allows users to specify parameters which are not tuned,
    instead taking the value of some expression of other tunable parameters (#4454)
* New argument `simplify_parameter_changes` in `client.configure_generation_strategy`
    (defaulted to `False`) which when `True` informs Ax to change as few parameters as
    possible relative to the status quo without degrading optimization performance. Has
    a near-zero effect on generation time (#4409)
* Default to `qLogNParEgo` acquisition function for multi-objective optimization in
    multi-objective optimization when the number of objectives is > 4, leading to
    improved walltime performance (#4347)

#### Bug fixes
* Fix issue during candidate generation involving `MapMetrics` providing progressions
    at different scales i.e. one progression goes up to 10^9 and the other goes up to
    10^6 by normalizing to [0, 1] (#4458)

#### Other changes
* Improve visual clarity in `ArmEffectsPlot` by removing certain elements including
    red "infeasibility" halos and optional cumulative best line (#4397, #4398)
* Instructions on citing Ax included in README.md and [ax.dev](https://ax.dev/) (#4317, #4357)
* New "Using external methods for candidate generation in Ax" tutorial on website (#4298)

## [1.1.2] -- Sept 9, 2025
#### Bug fixes
* Fixed rendering issue in ArmEffectsPlot when the number of arms displayed is greater
    than 20 (#4273)

### Other changes
* Enabled Winsorization transform by default, improving surrogate performance in the
    presence of outliers (#4277)

## [1.1.1] -- Sept 4, 2025
#### Bug fixes
* Correctly filter out observations from Abandoned trials/arms during candidate
    generation (#4155)
* Handle scalarized objectives in ResultsAnalysis (#4193)
* Fix bug in polytope sampler when generating more than one candidate in a batch (#4244)

#### Other changes
* Transition from setup.py to pyproject.toml for builds, modernizing Ax's build
    configuration and bringing it in compliance with PEP 518 and PEP 621 (#4100)
* Add py.typed file, which allows typecheckers like Pyre, mypy, etc. to see Ax's types
    and avoid a TypeStub error when importing Ax (#4139)
* Improve legibility of ArmEffectPlot by modifying legend and x-axis labels (#4220,
    #4243)
* Address logspew in OneHotEncoder transform (#4232)

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
