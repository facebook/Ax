---
id: usage
title: Usage
---
Modular design of Ax enables multiple modes of usage, which provide different balances of lightweight structure to flexibility and reproducibility. Below is the high-level overview of these usage modes, ranked from the most lightweight to fullest functionality.

## Service-like API

In this mode, Ax handles the experimentation algorithm(s), but not the execution of experiment iterations. This mode requires little to no knowledge of Ax data structures and easily integrates with various schedulers.

```python
while not completed_experiment:
    trial = ax.get_next_trial()
    suggestion = trial.arm.parameters
    # Locally evaluate suggestion.
    data = evaluate(suggestion)
    ax.log_data(trial.index, data)

best_arm = ax.get_optimized_arm()
```

## Library

In library usage mode, you get access to full flexibility of Ax. This requires some knowledge of _Ax architecture_ and allows for research and experimentation with different optimization algorithms and settings, locally or in a notebook. For simpler use cases, this API includes more lightweight classes, such as  _SimpleExperiment_...



## Managed Loop and CLI

Managed loop provides an ability to run an entire experiment or optimization inside Ax. This functionality requires you to _define metric computations and a way of running experiment iterations_...
