# Adding Tracking Metrics to Your Experiment
## Introduction
To gain a deeper understanding of your experiment's performance, you can track additional metrics beyond its primary objective. Ax allows you to add these tracking metrics to your experiment, providing valuable insights into the behavior of your system.

## Prerequisites
Before adding tracking metrics, make sure you have an understanding of Ax [experiments](#) and their components.

We will also assume you are already familiar with
[using Ax for ask-tell optimization](#), though this can be used for closed-loop
experiments as well.

## Setup
Before we begin you must instantiate the `Client` and configure it with your
experiment.

```python
from ax import AxClient

client = AxClient()

client.create_experiment(...)
client.set_optimization_config(...)
```

## Steps

1. Define the Metrics You Want to Track
2. Add the Metrics to the Experiment
3. Save the modifications to your experiment

### 1. Define the Metrics You Want to Track
Construct a list of `metric_names` and, optionally, `metric_definitions` to specify the metrics you intend to track.

Metric Definitions are like a set of instructions or extra details you provide for each metric you want to add to an experiment. Think of it as a way to customize how each metric behaves or is calculated.

When you define a metric, you might want to specify things like:
- What type of metric it is.
- Any specific conditions or filters that should be applied to it.
- Any special settings that are unique to that metric.


```python
# Define the metrics to track

metric_names=["tm1", tm2"],
metric_definitions = {"tm1": {"properties": {"m1_opt": "m1_val"}}}
```

### 2. Add the Metrics to the Experiment
Call the `add_tracking_metrics` method, passing in the list of metrics and metric definitions.

```python
# Add the metrics to the experiment

client.add_tracking_metrics(
    # one with a definition, one without
    metric_names=metric_names,
    metric_definitions=metric_definitions
)
```

### 3. Save the modifications to your experiment
Don't forget to call `save_experiment()` to push the modifications to your experiment. It will print "True" upon success.

```python
client.save_experiment()
```

### Learn More
For further learning, explore these additional resources:

* [Creating metrics in Ax](#)
* [Creating optimization configurations in Ax](#)
