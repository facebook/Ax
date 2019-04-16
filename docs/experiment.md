---
id: experiment
title: Experiment
---

## Building Blocks of an Experiment

### Experiment

In Ax, an [experiment](glossary.md) keeps track of the whole optimization process. It contains a search space, optimization config, metadata, information on what metrics to track and how to run iterations, etc. An [experiment](glossary.md) is composed of a sequence of [trials](glossary.md) — evaluations of a point in the search space, called an [arm](glossary.md) in Ax.  A [trial](glossary.md) is added to the experiment when a new arm is proposed by the optimization algorithm for evaluation, and it is completed with [objective](glossary.md) and [metric](glossary.md) values when observation data for that point is attached back to the experiment. For cases where multiple [arms](glossary.md) should be evaluated at the same time, Ax supports [batched trials](glossary.md).

### Search Space and Parameters

A [search space](glossary.md) is composed of a set of [parameters](glossary.md) to be tuned in the experiment, and optionally a set of [parameter constraints](glossary.md) that define restrictions across these parameters (e.g. p_a < p_b). Each parameter has a name, a type (```int```, ```float```, ```bool```, or ```string```), and a domain, which is a representation of the possible values the parameter can take.

Ax supports three types of parameters:

* **Range parameters**: must be of type int or float, and whose domain is represented by a lower and upper bound

```python
from ae.parameter import RangeParameter, ParameterType
range_param = RangeParameter(name="x", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)
```

* **Choice parameters**: domain is a set of values

```python
from ae.parameter import ChoiceParameter, ParameterType
choice_param = ChoiceParameter(name="y", parameter_type=ParameterType.STRING, value=["foo", "bar"](glossary.md))
```

* **Fixed parameters**: domain is a single value

```python
from ae.parameter import FixedParameter, ParameterType
fixed_param = ChoiceParameter(name="z", parameter_type=ParameterType.BOOL, value=True)
```

Ax supports three types of parameter constraints, each of which can only be used on in or float parameters:

* **Linear constraints**: e.g. w * v <= b where w is the vector of parameter weights, v is a vector of parameter values, and b is the specified bound

```python
from ae.parameter_constraint import ParameterConstraint

# 1.0*x * 0.5*y <= 1.0
ParameterConstraint(constraint_dict={"x": 1.0, "y": 0.5}, bound=1.0)
```

* **Order constraints**: a type of linear constraint, which specifies that one parameter must be smaller than the other

```python
from ae.parameter_constraint import OrderConstraint

# x <= y
OrderConstraint(lower_name="x", upper_name="y")
```

* **Sum constraints**: a type of linear constraint, which specifies that the sum of the parameters must be greater or less than a bound

```python
from ae.sum_constraint import SumConstraint

# x + y <= 3
SumConstraint(parameter_names=["x", "y"], is_upper_bound=False, bound=-3.0)
```

Given parameters and parameter constraints, you can construct a search space:

```python
from ae.search_space import SearchSpace

SearchSpace(parameters=[...], parameter_constraints=[...])
```

### Optimization Config

An [optimization config](glossary.md) is composed of an [objective metric](glossary.md) to be minimized or maximized in the experiment, and optionally a set of [outcome constraints](glossary.md) that place restrictions on how other metrics can be moved by the experiment. Note that you cannot constrain the objective metric.

```python
from ae.metric import Metric
from ae.objective import Objective

Objective(metric=Metric(name="m1"), minimize=True)
```

There is no minimum or maximum number of outcome constraints, but an individual metric can have at most two constraints — which is how we represent metrics with both upper and lower bounds.

Outcome constraints may of the form metric >= bound or metric <= bound. The bound can be expressed as an absolute measurement, or relative to the status quo (if applicable), in which case the bound is the acceptable percent change from the status quo's value.

```python
from ae.metric import Metric
from ae.outcome_constraint import OutcomeConstraint

# m2 cannot regress the status quo by more than 5%
OutcomeConstraint(metric=Metric(name="m2"), op = ComparisonOp.GEQ, bound=-5.0, relative=True)
```

### Arm

An [arm](glossary.md) in Ax is a set of parameters and their values with a name attached to it. The name 'arm' comes from the [multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit) optimization problem, in which a player facing a row of “one-armed bandits” slot machines has to choose which machines to play when and in what order. In the case of **hyperparameter optimization**, an [arm](glossary.md) corresponds to a hyperparameter configuration explored in the course of a given optimization.

Create an arm as follows:

```python
from ae.arm import Arm

Arm(parameters={"x": 0, "y": "Foo", z: True}, name="arm1")
```

Arms are typically attached to trials, as discussed in the [Experiment Lifecycle](#experiment-lifecycle) section below.

### Status Quo

An experiment can optionally contain a [status quo](glossary.md) arm, which represents the “control” parameterization. This allows viewing results and doing optimization using [relativized](glossary.md) outcomes, meaning all metrics will be presented as percentage deltas against the status quo.

If the status quo is specified on the experiment, it will be automatically added to every trial that is created.

## Experiment Lifecycle

An experiment consists of a sequence of trials, each of which evaluates one or more arms. Based on this evaluation, our optimization algorithms suggest a new set of one or more arms to evaluate. You then create a second trial containing these suggested arms, evaluate this trial, and repeat.

You can directly add arm(s) to a new trial, or you can add the output of the optimization algorithm, which is a [GeneratorRun](glossary.md):

```python
# If only one arm should be evaluated:
experiment.new_trial().add_arm(Arm(...))

# If multiple arms should be evaluated:
experiment.new_batch_trial().add_arms_and_weights(arms=[Arm(...), Arm(...)])

# To evaluate the arms suggested by a GeneratorRun:
experiment.new_batch_trial().add_generator_run(generator_run=GeneratorRun(...))
```

A trial goes through many phases during the experimentation cycle. It is tracked using a TrialStatus field. The stages are:

* `CANDIDATE` - Trial has just been created and can still be modified before deployment.
* `STAGED` - Relevant for external systems, where the trial configuration has been deployed but not begun the evaluation stage.
* `RUNNING` - Trial is in the process of being evaluated.
* `COMPLETED` - Trial completed evaluation successfully.
* `FAILED` - Trial incurred a failure while being evaluated.
* `ABANDONED` - User manually stopped the trial for some specified reason.

When a trial is first created, its status is "candidate". If applicable, we can call `trial.mark_staged` to move the trial into "staged" mode. We then call `trial.run`
to run the trial, which moves it into the "running" stage. We can then call
`trial.mark_completed`, `trial.mark_failed`, or `trial.mark_abandoned` to end the trial.

Note that if the trial's [runner](runner.md) has "staging_required" = True,
then `trial.run` will first mark the trial as "staged", and we can later call
`trial.mark_running` explicitly to move the trial to "running".
