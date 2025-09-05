# Multi-Objective Optimization with Ax

Multi-objective optimization (MOO) allows you to optimize multiple objectives
simultaneously, which is particularly useful when you have competing objectives.
In this recipe, we will demonstrate how to perform multi-objective optimization
using the Ax Client.

Note that while MOO can handle multiple objectives, it's generally recommended
to keep the number of objectives relatively small. Having too many objectives
can lead decreased optimization performance and difficulties in interpreting the
results.

## Prerequisites

We will assume you are already familiar with
[basic Ax usage](../tutorials/getting_started/index.mdx).

## Setup

Instantiate the `Client` and configure it with your experiment and metrics.

```python
client = Client()

client.configure_experiment(...)
client.configure_metrics(...)
```

## Steps

1. Configure an optimization with multiple objectives
2. Continue with iterating over trials and evaluating them
3. Observe optimal parametrizations

### 1. Configure an optimization with multiple objectives

We can leverage the Client's `configure_optimization` method. This method takes
in an objective goal as a string, and can be used to specify single-objective,
scalarized-objective, and multi-objective goals. For this recipe, we will define
a multi-objective goal:

```
client.configure_optimization(objectives="-cost, utility")
```

By default, objectives are assumed to be maximized. If you want to minimize an
objective, you can prepend the objective with a `-` sign.

### 2. Continue with iterating over trials and evaluating them

Now that your experiment has been configured for a multi-objective optimization,
you can simply continue with iterating over trials and evaluating them as you
typically would.

```python
# Getting just one trial in this example
trial_idx, parameters = client.get_next_trials(max_trials=1)().popitem()
client.complete_trial(...)
```

### 3. Observe optimal parametrizations

You can now observe the optimal parametrizations by calling
`get_optimal_pareto_frontier`. The function returns a list of tuples containing
the best parameters, their corresponding metric values, the most recent trial
that ran them, and the name of the best arm.

```python
frontier = client.get_pareto_frontier()
for parameters, metrics, trial_index, arm_name in frontier:
    ...
```

## Learn more

Take a look at these other resources to continue your learning:

- [Set scalarized-objective optimizations](../recipes/scalarized-objective.md)
- [Set outcome constraints](../recipes/outcome-constraints.md)
