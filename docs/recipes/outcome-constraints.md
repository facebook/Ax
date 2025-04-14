# Specifying Outcome Constraints
## Introduction
Outcome constraints can be a crucial component of optimization in Ax. They allow you to specify constraints on the outcomes of your experiment, ensuring that the optimized parameters do not degrade certain metrics.

## Prerequisites
Before specifying outcome constraints, make sure you have an understanding of Ax [experiments](#) and their components.


We will also assume you are already familiar with
[using Ax for ask-tell optimization](#), though this can be used for closed-loop
experiments as well.


## Setup
Before we begin you must instantiate the `AxClient` and configure it with your
experiment.


```python
from ax.service.ax_client import AxClient

client = AxClient()
```

## Steps

1. Define the Metrics You Want to Constrain
2. Create Outcome Constraint Map
3. Add the Constraints to the Experiment


### 1. Define the Metrics You Want to Constrain
Construct a list of `metric_names` to specify the metrics you intend to constrain.

```python
# Define the metrics to constrain

metric_names=["metric_1", "metric_2"],
```

### 2. Create Outcome Constraint Map
Create an `outcome_constraints` dictionary with the constraint information.

```python
outcome_constraints = ["metric_1 <= -1%"]  # the % makes it a relative constraint
```

Alternatively, you can create multiple outcome constraints:


```python
outcome_constraints = [
    "metric_1 <= -1%",
    "metric_2 >= 0.5%"
]
```

### 3. Add the Constraints to the Experiment
Call the `set_optimization_config` method, passing in the list of constraints.


```python
from ax.core.objective import ObjectiveProperties

objectives = {
    'metric_1': ObjectiveProperties(minimize=True)
}
client.set_optimization_config(
    objectives=objectives,
    outcome_constraints=outcome_constraints,
)
```

### Learn More
For further learning, explore these additional resources:

* [Creating tracking metrics in Ax](#)
* [Creating optimization configurations in Ax](#)
