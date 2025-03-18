# Multi-Objective Optimization with Ax

Multi-objective optimization (MOO) allows you to optimize multiple objectives simultaneously, which is particularly useful when you have competing objectives. In this recipe, we will explore how to perform multi-objective optimization using the AxClient.

## Introduction

In multi-objective optimization, objectives are specified through the `ObjectiveProperties` dataclass. This allows you to define whether each objective should be minimized or maximized, and optionally set a threshold to focus the optimization in a specific region of interest.

Note that while MOO can handle multiple objectives, it's generally recommended to keep the number of objectives relatively small. Having too many objectives can lead decreased optimization performance and difficulties in interpreting the results.

## Prerequisites

We will assume you are already familiar with
[using Ax for ask-tell optimization](#), though this can be used for closed-loop
experiments as well.

## Setup

Before we begin, make sure you have an AxClient instance configured for your experiment.


```python
from ax import AxClient

client = AxClient()
```

## Steps

1. Load a sample multi-objective problem
2. Create an experiment with multiple objectives
3. Define an evaluation function
4. Run the optimization

### 1. Load a sample multi-objective problem

We will use the Branin-Currin function as our sample multi-objective problem. This function is a synthetic benchmark problem with two objectives.

```python
import torch
from botorch.test_functions.multi_objective import BraninCurrin

branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

### 2. Create an experiment with multiple objectives

Next, we create an experiment using the AxClient, specifying the objectives and their properties.

```python
from ax.service.utils.instantiation import ObjectiveProperties

client.create_experiment(
    name="moo_experiment",
    parameters=[
        {
            "name": f"x{i+1}",
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for i in range(2)
    ],
    objectives={
        "a": ObjectiveProperties(minimize=False, threshold=branin_currin.ref_point[0]),
        "b": ObjectiveProperties(minimize=False, threshold=branin_currin.ref_point[1]),
    },
    overwrite_existing_experiment=True,
    is_test=True,
)
```

### 3. Define an evaluation function

The evaluation function takes a dictionary of parameter names mapped to values and returns a dictionary of objective names mapped to a tuple of mean and SEM values.

```python
def evaluate(parameters):
    evaluation = branin_currin(
        torch.tensor([parameters.get("x1"), parameters.get("x2")])
    )
    return {"a": (evaluation[0].item(), 0.0), "b": (evaluation[1].item(), 0.0)}
```

### 4. Run the optimization

Finally, we run the optimization by iterating over a number of trials, evaluating each set of parameters, and completing the trial with the results.

```python
for i in range(25):
    parameters, trial_index = client.get_next_trial()
    client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
```

## Learn more

Take a look at these other resources to continue your learning:

- [Set Objective Thresholds to focus candidate generation in a region of interest](#)
