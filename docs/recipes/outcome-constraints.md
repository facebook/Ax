# Specifying Outcome Constraints
## Introduction
Outcome constraints can be a crucial component of optimization in Ax. They allow you to specify constraints on the outcomes of your experiment, ensuring that the optimized parameters do not degrade certain metrics.

## Prerequisites
Before we begin you must instantiate the `Client` and configure it with your experiment and metrics.

We will also assume you are already familiar with [using Ax for ask-tell optimization](#).

```python
client = Client()

client.configure_experiment(...)
client.configure_metrics(...)
```

## Steps

1. Configure an optimization with outcome constraints
2. Continue with iterating over trials and evaluating them

### 1. Configure an optimization with outcome constraints
We can leverage the Client's `configure_optimization` method to configure an optimization with outcome constraints. This method takes an objective string and an outcome constraints string.

Outcome constraints allow us to express a desire to have a metric clear a threshold but not be further optimized. Some real world examples of where outcome constraints can be helpful include:

- Optimizing a model's architecture to improve accuracy but keeping its size small enough to fit on a chip
- Optimizing a mechanical component's strength while keeping it under a weight limit.

These constraints are expressed as inequalities.

```python
client.configure_optimization(objective="test_objective", outcome_constraints=["qps >= 100"])
```

Sometimes a constraint might not be against an absolute bound, but rather a desire to not regress more than a certain percent past a baseline configuration's value

```python
client.attach_baseline(parameters={"x1": 0.5})
client.configure_optimization(
    objective="test_objective",
    outcome_constraints=["qps >= 0.95 * baseline"]
)
```

This example will constrain the outcomes such that the QPS is at least 95% of the baseline arm's QPS.

Note that scalarized outcome constraints cannot be relative.

### 2. Continue with iterating over trials and evaluating them
Now that your experiment has been configured for a multi-objective optimization, you can simply continue with iterating over trials and evaluating them as you typically would.

```python
trial_idx, parameters = client.get_next_trials().popitem()
client.complete_trial(...)
```

## Learn more

Take a look at these other resources to continue your learning:

- [Multi-objective Optimizations in Ax](#)
- [Scalarized Objective Optimizations with Ax](#)
