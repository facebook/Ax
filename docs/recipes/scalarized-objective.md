 # Scalarized Objectives with Ax

In some cases, you may want to optimize a linear combination of multiple metrics rather than a single metric. This is where scalarized objectives come into play. You can define an objective function that is a weighted sum of several metrics, allowing you to balance different aspects of performance in your optimization.

## Introduction

Scalarized objectives are useful when you have multiple metrics that you want to consider simultaneously in your optimization process. By assigning weights to each metric, you can control their relative importance in the overall objective function.

## Prerequisites

We assume that you are already familiar with using Ax for experimentation and have an AxClient instance set up.

## Setup

Before we begin, make sure you have an AxClient instance configured for your experiment.

```python
from ax import AxClient

client = AxClient()
client.create_experiment(...)
```

## Steps

1. Define the metrics to be used in the scalarized objective
2. Create the scalarized objective with specified weights
3. Set up and attach the optimization configuration with the scalarized objective

### 1. Define the metrics to be used in the scalarized objective

First, define the metrics that you want to include in your scalarized objective.

```python
from ax.core.metric import Metric

class DummyMetric(Metric):
    def __init__(self, name):
        super().__init__(name=name)

metrics = [
    DummyMetric('metric_1'),
    DummyMetric('metric_2'),
]
```

### 2. Create the scalarized objective with specified weights

Next, create a `ScalarizedObjective` by specifying the metrics and their corresponding weights. The weights determine the relative importance of each metric in the overall objective function.

```python
from ax.core.objective import ScalarizedObjective

weights = [0.65, 0.35]
scalarized_objective = ScalarizedObjective(metrics=metrics, weights=weights)
```

### 3. Set up and attach the optimization configuration with the scalarized objective

Set up the `OptimizationConfig` using the scalarized objective and attach the optimization configuration to your experiment. You can also specify any outcome constraints if needed.

```python
client.set_optimization_config(
    objective=scalarized_objective,
    outcome_constraints=[],
    metric_definitions={},
)
```

## Learn more

Explore these additional resources to deepen your understanding:

- [Adding tracking metrics to your experiment](#)
