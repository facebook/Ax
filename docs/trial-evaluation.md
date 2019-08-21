---
id: trial-evaluation
title: Trial Evaluation
---

There are 3 paradigms for evaluating [trials](glossary.md#trial):

### Synchronous

In the synchronous paradigm, the user specifies an evaluation function which takes in parameters and outputs metric outcomes. This use case is supported by the [```SimpleExperiment```](/api/core.html#module-ax.core.simple_experiment) class:

```python
from ax import *

def dummy_evaluation_function(
    parameterization, # dict of parameter names to values of those parameters
    weight=None, # optional weight argument
):
    # given parameterization, compute a value for each metric
    x = parameterization["x"]
    y = parameterization["y"]
    objective_val = f(x, y)
    return {"objective": objective_val}

exp = SimpleExperiment(
    name="simple_experiment",
    search_space=SearchSpace(
      parameters=[
        RangeParameter(name="x", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT),
        RangeParameter(name="y", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT),
      ]
    ),
    evaluation_function=dummy_evaluation_function,
    objective_name="objective",
)
```

### Asynchronous

In the asynchronous paradigm, the trial is first deployed and the data is fetched at a later time. This is useful when evaluation happens on an external system and takes a long time to complete, such as for A/B tests. This is supported by the [```Experiment```](/api/core.html#module-ax.core.experiment) class. In this paradigm, the user specifies:
  * [`Runner`](../api/core.html#ax.core.runner.Runner): Defines how to deploy the experiment.
  * List of [`Metrics`](../api/core.html#ax.core.metric.Metric): Each defining how to compute/fetch data for a given metric.

A default runner is specified on the experiment, which is attached to each trial right before deployment. Runners can also be manually added to a trial to override the experiment default.


### Service-like

It is also possible to use Ax in a service-like manner, where Ax just suggests
[Arms](glossary.md#arm), which the client application evaluates and logs the results
back to Ax. In this case, no runner or evaluation function is needed,
since the evaluation is done on the client side. For more information,
refer to [```Service```](/api/service.html) module
reference and the [API docs](api.md).


## Evaluation Function

In synchronous cases where a parameterization can be evaluated right away (for example, when optimizing ML models locally or using a synthetic function), an evaluation function is a convenient way to automate evaluation. The arguments to an evaluation function must be:
- `parameterization`, a mapping of parameter names to their values,
- optionally a `weight` of the parameterization –– nullable `float` representing the fraction of available data on which the parameterization should be evaluated. For example, this could be a downsampling rate in case of hyperparameter optimization (what portion of data the ML model should be trained on for evaluation) or the percentage of users exposed to a given configuration in A/B testing. This `weight` is not used in unweighted experiments and defaults to `None`.

An evaluation function can return:
- A dictionary of metric names to tuples of (mean and [SEM](glossary.md#sem))
- A single (mean, SEM) tuple
- A single mean

In the second case, Ax will assume that the mean and the SEM are for the experiment objective, and in the third case that the mean is for the objective and that SEM is 0.

For example, this evaluation function computes mean and SEM for [Hartmann6](https://www.sfu.ca/~ssurjano/hart6.html) function and for the L2-norm:

```python
from ax.utils.measurement.synthetic_functions import hartmann6
def hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    # Standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}
```

This function computes just the objective mean and SEM, assuming the [Branin](https://www.sfu.ca/~ssurjano/branin.html) function is the objective on the experiment:

```python
from ax.utils.measurement.synthetic_functions import branin
def branin_evaluation_function(parameterization):
    # Standard error is 0, since we are computing a synthetic function.
    return (branin(parameterization.get("x1"), parameterization.get("x2")), 0.0)
```

This form would be equivalent to the above, since SEM is 0:

```python
lambda parameterization: branin(parameterization.get("x1"), parameterization.get("x2"))
```

For an example of an evaluation function that makes use of the `weight` argument, refer to the "Bandit Optimization" tutorial.
## Adding Your Own Runner

In order to control how the experiment is deployed, you can add your own runner. To do so, subclass [`Runner`](../api/core.html#ax.core.runner.Runner) and implement the [`run`](../api/core.html#ax.core.runner.Runner.run) method and [`staging_required`](../api/core.html#ax.core.runner.Runner.staging_required) property.

The [`run`](../api/core.html#ax.core.runner.Runner.run) method accepts a [`Trial`](../api/core.html#ax.core.trial.Trial) and returns a JSON-serializable dictionary of any necessary tracking info to fetch data later from this external system. A unique identifier or name for this trial in the external system should be stored in this dictionary with the key `"name"`, and this can later be accessed via `trial.deployed_name`.

The [`staging_required`](../api/core.html#ax.core.runner.Runner.staging_required) indicates whether the trial requires an intermediate staging period before evaluation begins. This property returns False by default.

An example implementation is given below:

```python
from foo_system import deploy_to_foo
from ax import Runner

class FooRunner(Runner):
    def __init__(self, foo_param):
        self.foo_param = foo_param

    def run(self, trial):
        name_to_params = {
            arm.name: arm.params for arm in trial.arms
        }
        run_metadata = deploy_to_foo(foo_param, name_to_params)
        return run_metadata

    @property
    def staging_required(self):
        return False
```

This is then invoked by calling:

```python
exp = Experiment(...)
exp.runner = FooRunner(foo_param="foo")
trial = exp.new_batch_trial()

# This calls runner's run method and stores metadata output
# in the trial.run_metadata field
trial.run()
```
