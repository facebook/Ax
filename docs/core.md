---
id: core
title: Core
---

### Overview

In Ax, an [experiment](glossary.md#experiment) keeps track of the whole optimization process. It contains a search space, optimization config, metadata, information on what metrics to track and how to run iterations, etc. An [experiment](glossary.md#experiment) is composed of a sequence of [trials](glossary.md#trial) each of which has a set of parameterizations (or [arms](glossary.md#arm)) to be evaluated. A [trial](glossary.md#trial) is added to the experiment when a new set of arms is proposed by the optimization algorithm. The trial is then evaluated to compute the values of each [metric](glossary.md#metric) for each arm, which are fed into the algorithms to create a new trial. Most applications have one arm per trial, which is the default implementation.

The core constructs that define the experiment are detailed below.

### Trial VS. Batch Trial

An [experiment](glossary.md#experiment) consists of [trials](glossary.md#trial), which can be one of two types: regular [trial](glossary.md#trial) or [batch trial](glossary.md#batch-trial). A regular [trial](glossary.md#trial) contains a single [arm](glossary.md#arm) and relevant metadata. A [batch trial](glossary.md#batch-trial) contains multiple [arms](glossary.md#arm), relevant metadata, and optionally a set of arm weights, which are a measure of how much of the total resources allocated to evaluating a batch should go towards evaluating the specific arm.

**A [batch trial](glossary.md#batch-trial) is not just a [trial](glossary.md#trial) with many arms!** It is a trial for which it is important that the arms are evaluated **simultaneously and together**. For instance, a batch trial would be appropriate in an A/B test where the evaluation results are subject to nonstationarity and require multiple arms to be deployed (and gathered data for) at the same time. For cases where multiple arms are evaluated separately and independently of each other, use multiple [trials](glossary.md#trial) with a single arm each, which will allow Ax to keep track of their deployment and results appropriately.

### Search Space and Parameters

A [search space](glossary.md#search-space) is composed of a set of [parameters](glossary.md#parameter) to be tuned in the experiment, and optionally a set of [parameter constraints](glossary.md#parameter-constraint) that define restrictions across these parameters (e.g. `p_a <= p_b`). Each parameter has a name, a type (```int```, ```float```, ```bool```, or ```string```), and a domain, which is a representation of the possible values the parameter can take. The search space is used by the optimization algorithms to know which arms are valid to suggest.

Ax supports three types of parameters:

* **Range parameters**: must be of type `int` or `float`, and the domain is represented by a lower and upper bound. If the parameter is specified as an `int`, newly generated points are rounded to the nearest integer by default.

```python
from ax import RangeParameter, ParameterType
float_range_param = RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)
int_range_param = RangeParameter(name="x2", parameter_type=ParameterType.INT, lower=0, upper=10)
```

* **Choice parameters**: domain is a set of values

```python
from ax import ChoiceParameter, ParameterType
choice_param = ChoiceParameter(name="y", parameter_type=ParameterType.STRING, values=["foo", "bar"])
```

* **Fixed parameters**: domain is a single value

```python
from ax import FixedParameter, ParameterType
fixed_param = FixedParameter(name="z", parameter_type=ParameterType.BOOL, value=True)
```

Ax supports three types of parameter constraints, each of which can only be used on `int` or `float` parameters:

* **Linear constraints**: `w * v` <= b where w is the vector of parameter weights, v is a vector of parameter values, * is the dot product, and b is the specified bound. Linear constraints are specified with the bound and a dictionary that maps parameter name to the weight

```python
from ax import ParameterConstraint

param_a = RangeParameter(name="a", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)
param_b = RangeParameter(name="b", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0)

# 1.0 * a + 0.5 * b <= 1.0
con_1 = ParameterConstraint(constraint_dict={"a": 1.0, "b": 0.5}, bound=1.0)
```

* **Order constraints**: specifies that one parameter must be smaller than the other

```python
from ax import OrderConstraint

# a <= b
con_2 = OrderConstraint(lower_parameter=param_a, upper_parameter=param_b)
```

* **Sum constraints**: specifies that the sum of the parameters must be greater or less than a bound

```python
from ax import SumConstraint

# a + b >= 0.5
con_3 = SumConstraint(parameters=[param_a, param_b], is_upper_bound=False, bound=0.5)
```

Given parameters and (optionally) parameter constraints, you can construct a search space:

```python
from ax import SearchSpace

SearchSpace(parameters=[param_a, param_b], parameter_constraints=[con_1, con_2, con_3])
```

### Optimization Config

An [optimization config](glossary.md#optimization-config) is composed of an [objective metric](glossary.md#objective) to be minimized or maximized, and optionally a set of [outcome constraints](glossary.md#outcome-constraint) that place restrictions on how other metrics can be moved by the experiment. Note that you cannot constrain the objective metric.

```python
from ax import Metric
from ax import Objective

objective = Objective(metric=Metric(name="m1"), minimize=True)
```

There is no minimum or maximum number of outcome constraints, but an individual metric can have at most two constraints — which is how we represent metrics with both upper and lower bounds.

Outcome constraints may be of the form `metric >= bound` or `metric <= bound`. The bound can be expressed as an absolute measurement, or relative to the status quo (if applicable), in which case the bound is the acceptable percent change from the status quo's value.

```python
from ax import Metric
from ax import OutcomeConstraint
from ax import ComparisonOp

# m2 cannot regress the status quo by more than 5%
oc = OutcomeConstraint(metric=Metric(name="m2"), op = ComparisonOp.GEQ, bound=-5.0, relative=True)
```

Finally, create the optimization config to attach to the experiment.

```python
from ax import OptimizationConfig

opt_config = OptimizationConfig(objective=objective, outcome_constraints=[oc])
```

### Arm

An [arm](glossary.md#arm) in Ax is a set of [parameters](glossary.md#parameter) and their values with a name attached to it. In the case of **hyperparameter optimization**, an [arm](glossary.md#arm) corresponds to a hyperparameter configuration explored in the course of a given optimization.

An arm is defined by specifying the value for each parameter, and optionally giving it a name:

```python
from ax import Arm

Arm(parameters={"x": 0, "y": "Foo", z: True})

# Names are automatically assigned by the experiment
# but can also be specified by the user
Arm(parameters={"x": 0, "y": "Foo", z: True}, name="arm1")
```

Arms are typically attached to trials, as discussed in the [Experiment Lifecycle](#experiment-lifecycle) section below.

### Status Quo

An experiment can optionally contain a [status quo](glossary.md#status-quo) arm, which represents the “control” parameterization. This allows viewing results and doing optimization using [relativized](glossary.md#relative-outcome-constraint) outcomes, meaning all metrics will be presented as percentage deltas against the status quo.

If the status quo is specified on the experiment, it will be automatically added to every trial that is created.

## Experiment Lifecycle

An experiment consists of a sequence of trials, each of which evaluates one or more arms. For more details on the implementing the evaluation, see the [trial evaluation](trial-evaluation.md) and [metric](data.md) references.

Based on the evaluation results, the optimization algorithm suggest one or more arms to evaluate. You then create a new trial containing these suggested arms, evaluate this trial, and repeat.

You can directly add arm(s) to a new trial, or you can add a [generator run](glossary.md#generator-run) –– output of the optimization algorithm:

```python
# If only one arm should be evaluated
experiment.new_trial().add_arm(Arm(...))

# If multiple arms should be evaluated
experiment.new_batch_trial().add_arms_and_weights(arms=[Arm(...), Arm(...)])

# To evaluate the arms suggested by a GeneratorRun
experiment.new_batch_trial().add_generator_run(generator_run=GeneratorRun(...))
```

A trial goes through multiple phases during the experimentation cycle, tracked by its [`TrialStatus`](../api/core.html#ax.core.base_trial.TrialStatus) field. These stages are:

* `CANDIDATE` - Trial has just been created and can still be modified before deployment.
* `STAGED` - Relevant for external systems, where the trial configuration has been deployed but not begun the evaluation stage.
* `RUNNING` - Trial is in the process of being evaluated.
* `COMPLETED` - Trial completed evaluation successfully.
* `FAILED` - Trial incurred a failure while being evaluated.
* `ABANDONED` - User manually stopped the trial for some specified reason.

When a trial is first created, its status is "candidate". If applicable, we can call `trial.mark_staged` to move the trial into "staged" mode. We then call `trial.run`
to run the trial, which moves it into the "running" stage. We can then call
`trial.mark_completed`, `trial.mark_failed`, or `trial.mark_abandoned` to end the trial.

If the trial's [runner](trial-evaluation.md#adding-your-own-runner) has "staging_required" = True,
then `trial.run` will first mark the trial as "staged", and we can later call
`trial.mark_running` explicitly to move the trial to "running".
