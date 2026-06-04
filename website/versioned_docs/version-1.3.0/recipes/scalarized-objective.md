# Scalarized Objective Optimizations with Ax

In some cases, you may want to optimize a linear combination of multiple metrics
rather than a single metric. This is where scalarized objectives come into the
picture. You can define an objective function that is a weighted sum of several
metrics, allowing you to balance different aspects of performance in your
optimization.

Scalarized objectives are useful when you have multiple metrics that you want to
consider simultaneously in your optimization process. By assigning weights to
each metric, you can control their relative importance in the overall objective
function.

Using a scalarized objective is a good choice if you have a good idea of what
the weights in the scalarization should be. If you're instead unclear about what
the tradeoffs between the individual objectives are and want to run the
optimization to understand those better, you should instead run a proper
[multi-objective optimization](../recipes/multi-objective-optimization.md).

## Setup

Before we begin you must instantiate the `Client` and configure it with your
experiment and metrics.

We will also assume you are already familiar with
[basic Ax usage](../tutorials/getting_started/index.mdx).

```python
client = Client()

client.configure_experiment(...)
client.configure_metrics(...)
```

## Steps

1. Configure an optimization with a scalarized objective
2. Continue with iterating over trials and evaluating them
3. Observe optimal parametrizations

### 1. Configure an optimization with a scalarized objective

We can leverage the Client's `configure_optimization` method to configure a
scalarized objective optimization. This method takes in an objective goal as a
string, and can be used to specify single-objective, scalarized-objective, and
multi-objective goals. For this recipe, we will use a scalarized-bjective goal:

```python
client.configure_optimization(objectives="2 * objective1 + objective")
```

In this example, we are optimizing a linear combination of two objectives,
`objective1` and `objective2`, and we value improvements to `objective1` twice
as much as improvements in `objective2`.

By default, objectives are assumed to be maximized. If you want to minimize an
objective, you can prepend the objective with a `-`.

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
`get_best_parameterization()`. The function returns a list of tuples containing
the best parameters, their corresponding metric values, the most recent trial
that ran them, and the name of the best arm.

```python
best_parameterization = client.get_best_parameterization()
for parameters, metrics, trial_index, arm_name in best_parameterization:
    ...
```

## Learn more

Take a look at these other resources to continue your learning:

- [Multi-objective Optimizations in Ax](../recipes/multi-objective-optimization.md)
- [Set outcome constraints](../recipes/outcome-constraints.md)
