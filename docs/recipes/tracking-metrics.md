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

1. Call `configure_metrics` to add the metrics to your experiment
2. Attaching data with tracking metrics

### 1. Call `configure_metrics` to add the metrics to your experiment

Call the `add_tracking_metrics` method, passing in the list of metrics we would
like to track.

When attaching metrics to an experiment, the Client will overwrite existing
metrics on the Experiment with the provided Metric(s) if they share the same
name. If no Metric with the same name exists, the Client will add the Metric as
a tracking metric.

```python
# Add the metrics to the experiment

client.configure_metrics([IMetric(name="my_tracking_metric_1"), IMetric(name="my_tracking_metric_2")])
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
