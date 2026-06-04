# Changelog

The release log for Ax.

## [1.3.0] -- Jun 4, 2026

### Compatibility

* Require BoTorch==0.18.0 (#5216)
  * Requires PyTorch>=2.4.0
  * Utilizes NumPyro for faster fully-Bayesian model fitting, requiring JAX as a dependency

### API Changes (`ax/api`)

* `Client.attach_data` (and therefore `Client.complete_trial`) now auto-registers metrics in `raw_data` that are not part of
  the optimization config as tracking metrics, matching the documented contract. Previously this raised `UserInputError` (#5135)
* `Client.configure_optimization` / objective and outcome-constraint strings now support linear equality constraints (`w^T x == b`) (#5174)

**Docs**
* New example starting an Ax experiment from dataframe data (#5098)

### Core (`ax/core`)

* `Objective` and `OutcomeConstraint` are now expression-based and no longer hold `Metric` instances;
  metrics live only on the `Experiment`. Construct them with an expression string, e.g.
  `Objective(expression="ne1 + 2*ne2")` or `OutcomeConstraint(expression="qps >= 7000")`
  (single, scalarized, and multi-objective are all expressed this way). Old-style constructors still
  work but discard the `Metric` after init. `Objective.metric` is removed -- get a metric's name from
  the expression and look it up via `Experiment.get_metric(name)`. This is a step toward deprecating
  `ScalarizedObjective`, `MultiObjective`, `ScalarizedOutcomeConstraint`, and `ObjectiveThreshold` (#5000, #5122, #5045, #5041)
  Note: We are aware of some edge-case bugs in this setup and we may be iterating on this in future releases.
* `ParameterConstraint` gains a new `equality=` kwarg for linear equality constraints (`w^T x == b`); equality constraints are honored in `SearchSpace` membership checks (#5173, #5175)
* Make all `mark_*` methods on `BaseTrial` no-op when the status is unchanged, and avoid rewriting timestamps when the status is unchanged (#5097, #5074)
* Runner lifecycle for search space editing: new `Runner.update` and `Runner.on_search_space_update` methods, a typed `RunnerConfig` infrastructure, and an `UNSET` sentinel (#5131, #5132, #5133, #5130)
* Support for Language-in-the-Loop labeling trials, with data-freshness hashing/stamping and stale-data handling (#4986, #4992, #5144)
* `AddExecutionViability` transform (lives in adapter; relies on core trial/metric plumbing) (#4547)

### Generation (`ax/generators`, `ax/adapter`, `ax/generation_strategy`)

* Replace Pyro with NumPyro (JAX-backed) for fully Bayesian NUTS inference -- roughly 25x faster fit on CPU with equivalent model quality (#5087)
* Threaded equality constraints from `ParameterConstraint` all the way down to BoTorch (#5176, #5177, #5179, #5178, #5182)
* Open-sourced the Transfer Learning stack: `TransferLearningAdapter` (registered as `Generators.BOTL`) (#5052, #5048, #5096, #5047)
* Heterogeneous TL now defaults to `MultiTaskGP` + `LearnedFeatureImputation` (`ImputedMultiTaskGP`) (#5106, #5183, #5193, #5192)
* Dispatch `qEUBO` for preferential BO (#5093) and support `PairwiseGP` in `ModelList` pipelines (#5092)
* `GenerationStrategy.fit(experiment, data)` public method encapsulating the transition-then-fit pattern (#4922)
* New `InSampleUniformGenerator` for model-free in-sample candidate selection (#4987)
* `FreshLILOLabelCheck` transition criterion and hash-based filtering of stale LILO data in the adapter (#4994, #4993)
* Expand the model space for numeric ordered `ChoiceParameter`s (#5210)

**Breaking**
* Removed the deprecated `steps=` kwarg from `GenerationStrategy.__init__` (use `nodes=`) (#5142)
* Removed the deprecated `use_update` argument to `GenerationStep` (#5187)
* Removed the deprecated `generated_points` argument to `RandomGenerator` (#5186)
* Re-index `TorchOptConfig.objective_thresholds` from `(n_outcomes,)` to `(n_objectives,)` (#5018)

### Analysis (`ax/analysis`, `ax/plot`)

* New `TransferLearningAnalysis` with paste/diff comparison links (#4918, #4980)
* New `UtilityRankingPlot` for preference metrics; preference metrics handled across the analysis layer (#5166, #5165)
* New `NotApplicableStateAnalysisCard` (#5035, #5036, #5051, #5163)
* `SensitivityAnalysisPlot` aesthetic improvements; use total-order sensitivity for high-dimensional experiments (#5015, #5115)

### Storage (`ax/storage`)**

* Add `step_size` to `SQAParameter` class, and as a column to the `parameter_v2` table in the DB (#5212). This will be leveraged in the next version.
* SQLAlchemy 2.0 migration: migrate SQA declarative classes to SA 2.0 `Mapped[T]`, make the SQA store SA 2.0-compatible (#5205, #5201, #5169, #5203, #5206)
* Support equality constraints in JSON and SQA storage (#5181)

### Legacy API (`ax/service`)**
Note: These are deprecated and may be removed in a future release.
* `optimize()` is rewritten as a thin wrapper over `ax.api.client.Client`; the `OptimizationLoop` class is removed. The `optimize()` signature and return type are preserved (#5143)
* Removed `AxClient.get_optimization_trace` -- superseded by `UtilityProgressionAnalysis` (#5168)
* Search space editing on `AxClient`: new `add_parameters`, `update_parameters`, and `disable_parameters` methods to mutate the search space after trials have started; parameter constraints can be added alongside new parameters (#4945, #5023)

### Orchestration (`ax/orchestration`)**

* Removed `Scheduler`, `SchedulerOptions`, and `SchedulerInternalError` -- deprecated since Ax 1.0.0; use the `Orchestrator` instead (#5188)



## [1.2.4] -- Mar 4, 2026

#### Bug Fixes
* Fix incorrect feasibility computation when using `qLogProbabilityOfFeasibility` for MOO — objective weights were applied twice via both the posterior transform and the constraint matrix, leading to incorrect results when only objective thresholds (no outcome constraints) were present (#4935)
* Add defensive `issubclass` guard for acquisition function dispatch to prevent silent fallthrough for future subclasses of `qLogProbabilityOfFeasibility` (#4938)
* Require only opt_config metrics for `prepare_arm_data` to fix `ArmEffectsPlot` failures when tracking metrics are missing (#4957)

#### Other changes
* Bumped pinned [botorch](https://github.com/pytorch/botorch) version to 0.17.2 (#4959). This picks up the following changes from botorch 0.17.2:
  - Support `post_processing_func` in `optimize_with_nsgaii` for post-processing optimization results, e.g., to round discrete dimensions to valid values
* Remove unused `objective_thresholds` parameter from `Acquisition.get_botorch_objective_and_transform` — the parameter was silently discarded (#4939)
* Add `Self` type annotations to clone methods for better type inference in subclasses (#4907)
* Heterogeneous search space utilities for transfer learning benchmarks (#4767)
* Migrate benchmarking state dict files for GPyTorch compatibility (#4916)
* Move `merge_multiple_curves` to Advanced tier in complexity classification (#4949)
* Move `infer_reference_point_from_experiment` and `get_tensor_converter_adapter` to `ax/service/utils/best_point.py` (#4940)
* Replace disclosure triangle with info icon in Bento notebooks for analysis cards (#4956)

## [1.2.3] -- Feb 19, 2026

#### Breaking Changes

**Packack Requirements**
* Python 3.11+ required (#4810)
* Pandas 3.0 upgrade (#4838)
* BoTorch 0.17.0 (#4911)

**Method Removals (non-API)**
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
