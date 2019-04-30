---
id: glossary
title: Glossary
sidebar_label: Glossary
---
### Arm
[Parameter](glossary.md#parameter) configuration (assignment of parameters to values). (ref: [Arm](/api/core.html#module-ax.core.arm))
### Bandit optimization
Machine learning-driven version of A/B testing that dynamically allocates traffic to [arms](glossary.md#arm) that are performing well, in search of the best [arm](glossary.md#arm) among a given set.
### Bayesian optimization
Sequential optimization strategy for finding an optimal [arm](glossary.md#arm) in a continuous search space.
### Experiment
Object that keeps track of the whole optimization process. Contains a [search space](glossary.md#search-space), [optimization config](glossary.md#optimization-config), and other metadata. (ref: [Experiment](/api/core.html#module-ax.core.experiment))
### Generator run
Outcome of a single run of the `gen` method of a [model bridge](glossary.md#model-bridge), contains the generated [arms](glossary.md#arm), as well as possibly best [arm](glossary.md#arm) predictions, other model predictions, fit times, etc. (ref: [Generator run](/api/core.html#module-ax.core.generator_run))
### Metric
Interface for fetching data for a specific measurement on an [experiment](glossary.md#experiment) or [trial](glossary.md#trial). (ref: [Metric](/api/core.html#module-ax.core.metric))
### Model
Algorithm that can be used to generate new points in arbitrary formats. (ref: [Model](/api/models.html))
### Model bridge
Adapter for interactions with a [model](glossary.md#model) within the Ax ecosystem. (ref: [Model bridge](/api/modelbridge.html))
### Objective
The [metric](glossary.md#metric) to be optimized, along with an optimization direction (maximize/minimize). (ref: [Objective](/api/core.html#module-ax.core.objective))
### Optimization config
Contains information necessary to run an optimization, i.e. [objective](glossary.md#objective) and [outcome constraints](glossary#outcome-constraints). (ref: [Optimization config](/api/core.html#module-ax.core.optimization_config))
### Outcome constraint
Constraint on [metric](glossary.md#metric) values, can be an order constraint or a sum constraint; violating [arms](glossary.md#arm) will be considered infeasible. (ref: [Outcome constraint](/api/core.html#module-ax.core.outcome_constraint))
### Parameter
Configurable quantity that can be assigned one of multiple possible values, can be continuous (`RangeParameter`), discrete (`ChoiceParameter`) or fixed (`FixedParameter`). (ref: [Parameter](/api/core.html#module-ax.core.parameter))
### Parameter constraint
Places restrictions on the relationships between [parameters](glossary.md#parameter).  For example `buffer_size1 < buffer_size2` or `buffer_size_1 + buffer_size_2 < 1024`. (ref: [Parameter constraint](/api/core.html#module-ax.core.parameter_constraint))
### Relative outcome constraint
[Outcome constraint](glossary.md#outcome-constraint) evaluated relative to the [status quo](glossary.md#status-quo) instead of in absolute value terms. (ref: [Outcome constraint](/api/core.html#module-ax.core.outcome_constraint))
### Runner
Dispatch abstraction that defines how a given [trial](glossary.md#trial) is to be ran locally or externally. (ref: [Runner](/api/core.html#module-ax.core.runner))
### Search space
Continuous, discrete or mixed design space that defines the set of parameterizations that can be evaluated during the optimization. (ref: [Search space](/api/core.html#module-ax.core.search_space))
### SEM
[Standard error](https://en.wikipedia.org/wiki/Standard_error) of the mean, 0.0 for noiseless measurements.
### Simple experiment
Subclass of [experiment](glossary.md#experiment) that assumes synchronous evaluation (uses evaluation function to get data for trials right after they are suggested), abstracts away certain details, and allows for faster instantiation. (ref: [Simple experiment](/api/core.html#module-ax.core.simple_experiment))
### Status quo
An [arm](glossary.md#arm), usually the currently deployed configuration, which provides a baseline for comparing all other [arms](glossary.md#arm). A.k.a. a control [arm](glossary.md#arm). (ref: [Status quo](/api/core.html#ax.core.experiment.Experiment.status_quo))
### Trial
Single step in the experiment, contains a single [arm](glossary.md#arm). In cases where the trial contains multiple [arms](glossary.md#arm) that are deployed simultaneously, we refer to it as a **batch trial**. (ref: [Trial](/api/core.html#module-ax.core.trial), [Batch trial](/api/core.html#module-ax.core.batch_trial))
