---
id: runner
title: Runner
---

## Function Evaluation in Ax

There are 3 paradigms for evaluating trials:

### Synchronous

In the synchronous paradigm, the user specifies an evaluation function which takes in parameters and outputs metric outcomes. This use case is supported by the [```SimpleExperiment```](/api/core.html#module-ax.core.simple_experiment) class:

```python
from ax import *

def dummy_evaluation_function(
    parameterization, # dict of parameter names to values of those parameters
    weight=None, # optional outcome weight argument
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
  * Runner: Defines how to deploy the experiment.
  * List of metrics: Each defining how to compute/fetch data for a given metric.

A default runner is specified on the experiment, which is attached to each trial right before deployment. Runners can also be manually added to a trial to override the experiment default.


### Service-like

Aside from those 2 paradigms, it is also possible to use Ax in a service-like
manner, where Ax just suggests [Arms](glossary.md#arm), which the client application
tries and logs the results back to Ax. In this case, no runner or evaluation
function is needed, since the evaluation is done on the client side. For more
information, refer to [```Service```](/api/core.html#module-ax.service) module
reference and the [API docs](api.md).


## Adding Your Own Runner

To add your own runner, subclass `Runner` and implement the `run` method and `staging_required` property.

The `run` method accepts a Trial and returns a JSON-serializable dictionary of any necessary tracking info to fetch data later from this external system. A unique identifier or name for this trial in the external system should be stored in this dictionary with the key "name", can then be accessed via trial.deployed_name.

The `staging_required` indicates whether the trial requires an intermediate staging period before evaluation begins. This property returns False by default.

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
