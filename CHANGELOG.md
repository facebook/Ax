# Changelog

The release log for Ax.
## [1.2.3] -- Feb 19, 2026

#### Breaking Changes

**Requirements**
* Python 3.11+ required (#4810)
* Pandas 3.0 upgrade (#4838)
* BoTorch 0.17.0 (#4911)

**API Removals**
* `transition_to` now required on `TransitionCriterion` (#4848) — Users must explicitly specify transition targets
* Removed callable serialization (#4806) — Encoding callables now raises an exception
* Removed legacy classes: `TData` (#4771), `MinimumTrialsInStatus` (#4786), completion criteria (#4850), `arms_per_node` override (#4822)

#### New Features
* **Experiment Lifecycle Tracking**: New `ExperimentStatus` enum (`DRAFT`, `INITIALIZATION`, `OPTIMIZATION`, `COMPLETED`) with automatic status updates from the Scheduler based on generation strategy phase (#4737, #4738, #4891)
* **BOPE (Preference Learning)**: Utility-based traces via PairwiseGP preference models (#4792); `UtilityProgressionAnalysis` support with "User Preference Score" UI (#4793)
* **BONSAI (Pruning)**: New `pruning_target_parameterization` API parameter (#4775); tutorial and documentation added (#4865, #4871)
* `add_tracking_metrics()` method (#4858)
* `TorchAdapter.botorch_model` convenience property (#4827)
* `DerivedParameter` now supports `bool` and `str` types (#4847)
* LLM integration: `LLMProvider`/`LLMMessage` abstractions and `llm_messages` on Experiment (#4826, #4904)
* Improved slice/contour plots: uses status_quo, best trial, center hierarchy (#4841)
* Sensitivity analysis excludes 'step' by default (#4777)
* `ScalarizedOutcomeConstraint` support in feasibility analysis (#4856)

#### Performance
* `DataRow`-backed `Data` class with itertuples — 3.6x faster tensor creation (#4773, #4774, #4798)
* Generation strategy caching: significant speedup in high trial count regimes (#4830)
* Optimization complete logic: O(nodes x TC) to O(TC on current node) (#4828)

#### Bug Fixes
* Fix `GeneratorRun.clone()` not copying metadata (mutations affected original) (#4892)
* Fix OneHot transform not updating hierarchical parameter dependents (#4825)
* Fix Float parameters loaded as ints from SQA (#4853)
* Fix scikit-learn 1.8.0 compatibility with XGBoost (#4816)
* Trials now marked ABANDONED (not FAILED) on metric fetch failure (#4779)
* Baseline improvement healthcheck now shows WARNING instead of FAIL (#4883)

#### Other changes
* Codebase updated to Python 3.11+ idioms: `typing.Self` (#4867), `StrEnum` (#4868), `ExceptionGroup` (PEP 654) (#4877), `asyncio.TaskGroup` (#4878), PEP 604 type annotations (#4912)
* For the complete list of 90+ PRs, see the [GitHub releases page](https://github.com/facebook/Ax/releases)

## [1.2.2] -- Jan 2026

NOTE: This will be the last Ax release before SQLAlchemy becomes a required dependency.

#### Deprecations
* Add deprecation warning to AxClient (#4749)
* Add deprecation warning to 'optimize' loop API (#4697)
* Deprecate Trial.runner (#4460)
* Deprecate TensorboardMetric's `percentile` in favor of `quantile` (#4676)
* Deprecate default_data_type argument to Experiment (#4698)

#### New Features
* Efficient leave-one-out cross-validation for Gaussian processes (#4631)
* Add patience parameter to PercentileEarlyStoppingStrategy (#4595)
* Log-scale support for ChoiceParameter (#4591)
* Support ChoiceParameter in Log transform (#4592)
* Add robust trial status polling to Orchestrator (#4756)
* Expose validation for TL experiments and fetching of candidate TL sources through AxService (#4615)
* Add PreferenceOptimizationConfig with storage layer support (#4638)
* Add PLBO transform and metric ordering validation (#4633)
* Add expect_relativized_outcomes flag to PreferenceOptimizationConfig (#4632)
* Add kendall tau rank correlation diagnostic (#4617)
* Vectorize SearchSpace membership check for performance (#4762)

#### Analyses
* New UtilityProgression Analysis for tracking optimization progress over time (#4535)
* New Best Trials Analysis for identifying top-performing trials (#4545)
* New Early Stopping Healthcheck analysis (#4569)
* New Predictable Metrics Healthcheck analysis (#4598)
* New Baseline Improvement Healthcheck analysis (#4673)
* New Complexity Rating Healthcheck for assessing optimization difficulty (#4556)
* Analysis to visualize experiment generation strategy (#4759)
* Add Pareto frontier display on MOO objective scatter plots (#4708)
* Add Progression Plots for MapMetric experiments to ResultsAnalysis (#4705)
* Add SEM display option to ContourPlot (#4690)
* Add markers to ProgressionPlot line charts (#4693)
* GraphvizAnalysisCard and HierarchicalSearchSpaceGraph visualization (#4616)
* IndividualConstraintsFeasibilityAnalysis replaces ConstraintsFeasibilityAnalysis (#4527)

#### Bug Fixes
* Fix tied trial bug in PercentileESS: use rank() for n_best_trial protection (#4587)
* Fix StandardizeY not updating weights in ScalarizedObjective (#4619)
* Fix StratifiedStandardizeY behavior with ScalarizedObjective & ScalarizedOutcomeConstraint (#4621)
* Fix floating point precision issue in step_size validation (#4604)
* Fix progression normalization logic in early-stopping strategies (#4525)
* Fix dependent parameter handling in Log transform (#4679)
* Drop NaN values in MAP_KEY column before align_partial_results (#4634)
* Filter failed trials from plots (#4725)
* Allow single progression early stopping checks when patience > 0 (#4635)
* Update SOBOL transition criterion to exclude ABANDONED and FAILED trials (#4776)

#### Other changes
* Speed up MapDataReplayMetric (#4654)
* Fast MapData.df implementation (#4487)
* Validate patience <= min_progression in PercentileEarlyStoppingStrategy (#4639)
* Enforce `smoothing` in `[0, 1)` for TensorBoardMetric (#4661)
* Enforce sort_values=True for numeric ordered ChoiceParameter (#4597)
* Add error if PowerTransformY is used with ScalarizedObjective (#4622)
* Support ScalarizedObjective in get_best_parameters with model predictions (#4594)
* Rename model_kwargs -> generator_kwargs (#4668)
* Rename model_gen_kwargs -> generator_gen_kwargs (#4667)
* Rename model_cv_kwargs -> cv_kwargs (#4669)

## [1.2.1] -- Nov 21, 2025
#### Bug fixes
* Improved error messaging for `client.compute_analyses` when certain analyses are
    not yet available (#4441)
* Fix tooltip mismatch bug in `ArmEffectsPlot` (#4479)

#### Other changes
* Bumped pinned [botorch](https://github.com/pytorch/botorch) version to 0.16.1 (#4570)
* Removed deprecated robust optimization functionality (#4493)
* Allow `HierarchicalSearchSpace` to be constructed with multiple root nodes (#4560)

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
