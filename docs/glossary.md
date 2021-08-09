---
id: glossary
title: Glossary
sidebar_label: Glossary
---
### Arm
Mapping from [parameters](glossary.md#parameter) (i.e. a parameterization or parameter configuration) to parameter values. An arm provides the configuration to be tested in an Ax [trial](glossary.md#trial). Also known as "treatment group" or "parameterization", the name 'arm' comes from the [Multi-Armed Bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) optimization problem, in which a player facing a row of “one-armed bandit” slot machines has to choose which machines to play when and in what order. [```[Arm]```](/api/core.html#module-ax.core.arm)
### Bandit optimization
Machine learning-driven version of A/B testing that dynamically allocates traffic to [arms](glossary.md#arm) which are performing well, to determine the best [arm](glossary.md#arm) among a given set.
### Batch trial
Single step in the [experiment](glossary.md#experiment), contains multiple [arms](glossary.md#arm) that are **deployed and evaluated together**. A batch trial is not just a trial with many arms; it is a trial for which it is important that the arms are evaluated simultaneously, e.g. in an A/B test where the evaluation results are subject to nonstationarity. For cases where multiple arms are evaluated separately and independently of each other, use multiple regular [trials](glossary.md#trial) with a single arm each. [```[BatchTrial]```](/api/core.html#module-ax.core.batch_trial)
### Bayesian optimization
Sequential optimization strategy for finding an optimal [arm](glossary.md#arm) in a continuous [search space](glossary.md#search-space).
### Evaluation function
Function that takes a parameterization and an optional weight as input and outputs a set of metric evaluations ([more details](trial-evaluation.md#evaluation-function)). Used in the [Loop API](api.md).
### Experiment
Object that keeps track of the whole optimization process. Contains a [search space](glossary.md#search-space), [optimization config](glossary.md#optimization-config), and other metadata. [```[Experiment]```](/api/core.html#module-ax.core.experiment)
### Generation strategy
Abstraction that allows to declaratively specify one or multiple models to use in the course of the optimization and automate transition between them (relevant [tutorial](/tutorials/scheduler.html)). [```[GenerationStrategy]```](/api/modelbridge.html#module-ax.modelbridge.generation_strategy)
### Generator run
Outcome of a single run of the `gen` method of a [model bridge](glossary.md#model-bridge), contains the generated [arms](glossary.md#arm), as well as possibly best [arm](glossary.md#arm) predictions, other [model](glossary.md#model) predictions, fit times etc. [```[GeneratorRun]```](/api/core.html#module-ax.core.generator_run)
### Metric
Interface for fetching data for a specific measurement on an [experiment](glossary.md#experiment) or [trial](glossary.md#trial). [```[Metric]```](/api/core.html#module-ax.core.metric)
### Model
Algorithm that can be used to generate new points in a [search space](glossary.md#search-space). [```[Model]```](/api/models.html)
### Model bridge
Adapter for interactions with a [model](glossary.md#model) within the Ax ecosystem. [```[ModelBridge]```](/api/modelbridge.html)
### Objective
The [metric](glossary.md#metric) to be optimized, with an optimization direction (maximize/minimize). [```[Objective]```](/api/core.html#module-ax.core.objective)
### Optimization config
Contains information necessary to run an optimization, i.e. [objective](glossary.md#objective) and [outcome constraints](glossary#outcome-constraints). [```[OptimizationConfig]```](/api/core.html#module-ax.core.optimization_config)
### Outcome constraint
Constraint on [metric](glossary.md#metric) values, can be an order constraint or a sum constraint; violating [arms](glossary.md#arm) will be considered infeasible. [```[OutcomeConstraint]```](/api/core.html#module-ax.core.outcome_constraint)
### Parameter
Configurable quantity that can be assigned one of multiple possible values, can be continuous ([`RangeParameter`](../api/core.html#ax.core.parameter.RangeParameter)), discrete ([`ChoiceParameter`](../api/core.html#ax.core.parameter.ChoiceParameter)) or fixed ([`FixedParameter`](../api/core.html#ax.core.parameter.FixedParameter)). [```[Parameter]```](/api/core.html#module-ax.core.parameter)
### Parameter constraint
Places restrictions on the relationships between [parameters](glossary.md#parameter).  For example `buffer_size1 < buffer_size2` or `buffer_size_1 + buffer_size_2 < 1024`. [```[ParameterConstraint]```](/api/core.html#module-ax.core.parameter_constraint)
### Relative outcome constraint
[Outcome constraint](glossary.md#outcome-constraint) evaluated relative to the [status quo](glossary.md#status-quo) instead of directly on the metric value. [```[OutcomeConstraint]```](/api/core.html#module-ax.core.outcome_constraint)
### Runner
Dispatch abstraction that defines how a given [trial](glossary.md#trial) is to be run (either locally or by dispatching to an external system). [````[Runner]````](/api/core.html#module-ax.core.runner)
### Scheduler
Configurable closed-loop optimization manager class, capable of conducting a full experiment by deploying trials, polling their results, and leveraging those results to generate and deploy more
trials (relevant [tutorial](/tutorials/scheduler.html)). [````[Scheduler]````](https://ax.dev/versions/latest/api/service.html#module-ax.service.scheduler)
### Search space
Continuous, discrete or mixed design space that defines the set of [parameters](glossary.md#parameter) to be tuned in the optimization, and optionally [parameter constraints](glossary.md#parameter-constraint) on these parameters. The parameters of the [arms](glossary.md#arm) to be evaluated in the optimization are drawn from a search space. [```[SearchSpace]```](/api/core.html#module-ax.core.search_space)
### SEM
[Standard error](https://en.wikipedia.org/wiki/Standard_error) of the [metric](glossary.md#metric)'s mean, 0.0 for noiseless measurements. If no value is provided, defaults to `np.nan`, in which case Ax infers its value using the measurements collected during experimentation.
### Status quo
An [arm](glossary.md#arm), usually the currently deployed configuration, which provides a baseline for comparing all other [arms](glossary.md#arm). Also known as a control [arm](glossary.md#arm). [```[StatusQuo]```](/api/core.html#ax.core.experiment.Experiment.status_quo)
### Trial
Single step in the [experiment](glossary.md#experiment), contains a single [arm](glossary.md#arm). In cases where the trial contains multiple [arms](glossary.md#arm) that are deployed simultaneously, we refer to it as a [batch trial](glossary.md#batch-trial). [```[Trial]```](/api/core.html#module-ax.core.trial), [```[BatchTrial]```](/api/core.html#module-ax.core.batch_trial)
