---
id: glossary
title: Glossary
sidebar_label: Glossary
---
### Arm
[Parameter](glossary.md#parameter) configuration (assignment of parameters to values).
### Bandit optimization
Machine learning-driven version of A/B testing that dynamically allocates traffic to [arms](glossary.md#arm) that are performing well, in search of the best [arm](glossary.md#arm) among a given set.
### Bayesian optimization
Sequential optimization strategy for finding an optimal [arm](glossary.md#arm) in a continuous search space.
### Experiment
Object that keeps track of the whole optimization process. Contains a [search space](glossary.md#search-space), [optimization config](glossary.md#optimization-config), and other metadata.
### Generator run
Outcome of a single run of the `gen` method of a [model bridge](glossary.md#model-bridge), contains the generated [arms](glossary.md#arm), as well as possibly best [arm](glossary.md#arm) predictions, other model predictions, fit times etc.
### Metric
Interface for fetching data for a specific measurement on an [experiment](glossary.md#experiment) or [trial](glossary.md#trial).
### Model
Algorithm that can be used to generate new points in arbitrary formats.
### Model bridge
Adapter for interactions with a [model](glossary.md#model) within the Ax ecosystem.
### Objective
The [metric](glossary.md#metric) to be optimized, along with an optimization direction (maximize/minimize).
### Optimization config
Contains information necessary to run an optimization, i.e. [objective](glossary.md#objective) and [outcome constraints](glossary#outcome-constraints).
### Outcome constraint
Constraint on [metric](glossary.md#metric) values, can be an order constraint or a sum constraint; violating [arms](glossary.md#arm) will be considered infeasible.
### Parameter
Configurable quantity that can be assigned one of multiple possible values, can be continuous (`RangeParameter`), discrete (`ChoiceParameter`) or fixed (`FixedParameter`).
### Parameter constraint
Places restrictions on the relationships between [parameters](glossary.md#parameter).  For example `buffer_size1 < buffer_size2` or `buffer_size_1 + buffer_size_2 < 1024`.
### Relative outcome constraint
[Outcome constraint](glossary.md#outcome-constraint) evaluated relative to the [status quo](glossary.md#status-quo) instead of in absolute value terms.
### Runner
Dispatch abstraction that defines how a given [trial](glossary.md#trial) is to be ran locally or externally.
### Search space
Continuous, discrete or mixed design space that defines the set of parameterizations that can be evaluated during the optimization.
### SEM
[Standard error](https://en.wikipedia.org/wiki/Standard_error) of the mean, 0.0 for noiseless measurements.
### Simple experiment
Subclass of [experiment](glossary.md#experiment) that assumes synchronous evaluation (uses evaluation function to get data for trials right after they are suggested), abstracts away certain details, and allows for faster instantiation.
### Status quo
An [arm](glossary.md#arm), usually the currently deployed configuration, which provides a baseline for comparing all other [arms](glossary.md#arm). A.k.a. a control [arm](glossary.md#arm).
### Trial
Single step in the experiment, contains a single [arm](glossary.md#arm). In cases where the trial contains multiple [arms](glossary.md#arm) that are deployed simultaneously, we refer to it as a **batch trial**.
