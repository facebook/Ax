# Adding Tracking Metrics to Your Experiment

## Introduction

To gain a deeper understanding of your experiment's performance, you can track
additional metrics beyond its primary objective(s). Ax allows you to add these
tracking metrics to your experiment, providing valuable insights into the
behavior of your system.

## Setup

Before we begin you must instantiate the `Client` and configure it with your
experiment. In this example, we will be setting our objective to a custom
metric.

```python
client = Client()

client.configure_experiment(...)
client.configure_optimization(objective='custom_metric')
```

## Steps

1. Call `configure_tracking_metrics` to add the metrics to your experiment
2. Attaching data with tracking metrics

### 1. Call `configure_tracking_metrics` to add the metrics to your experiment

Call the `configure_tracking_metrics` method, passing in the list of metric
names you would like to track.

If any of the metrics are already defined on the experiment, they will be
skipped with a warning.

```python
# Add the tracking metrics to the experiment by name
client.configure_tracking_metrics(["my_tracking_metric_1", "my_tracking_metric_2"])
```

### 2. Attaching data with tracking metrics

To associate data with your experiment, such as when completing a specific trial
and providing additional information, you can utilize the `complete_trial`
method along with its `raw_data` parameter to attach tracking metrics for that
particular trial.

```python
# Getting just one trial in this example
trial_index, parameters = client.get_next_trials(max_trials=1)().popitem()

client.complete_trial(trial_index=trial_index, raw_data={"my_tracking_metric_1": ..., "my_tracking_metric_2": ...})
```
