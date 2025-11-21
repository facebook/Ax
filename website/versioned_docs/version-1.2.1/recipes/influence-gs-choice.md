# Influencing The Choice of a Generation Strategy

## Introduction

The GenerationStrategy used by Ax to generate candidate configurations / trials is determined by various factors, such as experiment characteristics and configuration settings. In this tutorial, we'll explore how to influence the selection of a GenerationStrategy via the Client's customizable configurations.

## Prerequisites

Instantiate the `Client` and configure your experiment and metrics.

We will also assume you are already familiar with
[basic Ax usage](../tutorials/getting_started).

```python
client = Client()

client.configure_experiment(...)
client.configure_metrics(...)
```

## Steps

1. Configure a generation strategy via `configure_generation_strategy`
2. Continue with iterating over trials and evaluating them
3. View the state of the experiment via `summarize()`

### 1. Configure a generation strategy via `configure_generation_strategy`
We can leverage the Client's `configure_generation_strategy` method to configure a generation strategy. This is an optional method used to configure the way candidate parameterizations are generated during the optimization. If not called, a default `GenerationStrategy` will be used.

In this example, we will configure a GenerationStrategy that reduces or skips the initialization step, does not use the center of the search space as the first point, and allows for exceeding the provided initialization budget. To learn more about what other configurations you can leverage, [check out the API docs for configure_generation_strategy](https://ax.readthedocs.io/en/latest/api.html#ax.api.client.Client.configure_generation_strategy)

```python
client.configure_generation_strategy(
    # Reduce or skip the initialization step
    # NOTE: Don't set this to 0 (which would skip initialization) unless you
    # attached manual trials. It's not possible to fully skip the initialization
    # step and start Bayesian optimization without any completed trials with data!
    initialization_budget=1,
    # Do not use center of searchspace as the first point
    initialize_with_center=False,
    # Allow for exceeding the provided initialization budget
    allow_exceeding_initialization_budget=True,
    # Don't count manually added trials against the initialization
    # budget (i.e. ensure that the 3 quasi-random Sobol trials will
    # be run, e.g. to sample the search space more evenly).
    use_existing_trials_for_initialization=False,
)
```

It’s possible to specify a fully custom `GenerationStrategy` via `Client.set_generation_strategy`, but this usage is not part of the Ax backward-compatible API and backward compatibility is not guaranteed: the method or its input could thus change between minor versions in the future.

### 2. Continue with iterating over trials and evaluating them

Now that your experiment has been configured with a desired GenerationStrategy,
you can simply continue with iterating over trials and evaluating them as you
typically would.

```python
# Getting just one trial in this example
trial_idx, parameters = client.get_next_trials(max_trials=1)().popitem()
client.complete_trial(...)
```

### 3. View the state of the experiment via `summarize()`
We can now view the state of the experiment by calling `summarize()`, and
validate that our Generation Strategy configurations were used when generating candidates:

```python
client.summarize()
```

## Learn more

Take a look at these other recipes to continue your learning:

- [Multi-objective Optimizations in Ax](../recipes/multi-objective-optimization)
- [Scalarized Objective Optimizations with Ax](../recipes/scalarized-objective)
