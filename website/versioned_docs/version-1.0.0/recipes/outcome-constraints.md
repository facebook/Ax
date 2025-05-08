# Specifying Outcome Constraints

## Introduction

Outcome constraints specify constraints on the outcomes of your experiment,
ensuring that the optimized parameters do not degrade certain metrics.

## Prerequisites

Instantiate the `Client` and configure your experiment and metrics.

We will also assume you are already familiar with
[basic Ax usage](../tutorials/getting_started/index.mdx).

```python
client = Client()

client.configure_experiment(...)
client.configure_metrics(...)
```

## Steps

1. Configure an optimization with outcome constraints
2. Continue with iterating over trials and evaluating them

### 1. Configure an optimization with outcome constraints

We can leverage the Client's `configure_optimization` method to configure an
optimization with outcome constraints. This method takes an objective string and
a sequence of outcome constraint strings.

Outcome constraints indicate the preference for a metric to meet a specified
threshold but not be further optimized. Some real world examples of where
outcome constraints can be helpful include:

- Optimizing a model's architecture to improve accuracy but keeping its size
  small enough to fit on a chip
- Optimizing a mechanical component's strength while keeping it under a weight
  limit.

These constraints are expressed as inequalities.

```python
client.configure_optimization(objective="test_objective", outcome_constraints=["qps >= 100"])
```

Sometimes a constraint might not be against an absolute bound, but rather a
desire to not regress more than a certain percent past a baseline
configuration's value

```python
client.attach_baseline(parameters={"x1": 0.5})
client.configure_optimization(
    objective="test_objective",
    outcome_constraints=["qps >= 0.95 * baseline"]
)
```

This example will constrain the outcomes such that the QPS is at least 95% of
the baseline arm's QPS.

Note that scalarized outcome constraints cannot be relative.

### 2. Continue with iterating over trials and evaluating them

Now that your experiment has been configured for a multi-objective optimization,
you can simply continue with iterating over trials and evaluating them as you
typically would.

```python
# Getting just one trial in this example
trial_idx, parameters = client.get_next_trials(max_trials=1)().popitem()
client.complete_trial(...)
```

## Learn more

Take a look at these other resources to continue your learning:

- [Multi-objective Optimizations in Ax](../recipes/multi-objective-optimization.md)
- [Scalarized Objective Optimizations with Ax](../recipes/scalarized-objective.md)
