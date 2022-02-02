#!/usr/bin/env python
# coding: utf-8

# #  Service API Example on Hartmann6
# 
# The Ax Service API is designed to allow the user to control scheduling of trials and data computation while having an easy to use interface with Ax.
# 
# The user iteratively:
# - Queries Ax for candidates
# - Schedules / deploys them however they choose
# - Computes data and logs to Ax
# - Repeat

# In[1]:


import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Initialize client
# 
# Create a client object to interface with Ax APIs. By default this runs locally without storage.

# In[2]:


ax = AxClient()


# ## 2. Set up experiment
# An experiment consists of a **search space** (parameters and parameter constraints) and **optimization configuration** (objective name, minimization setting, and outcome constraints). Note that:
# - Only `name`, `parameters`, and `objective_name` arguments are required.
# - Dictionaries in `parameters` have the following required keys: "name" - parameter name, "type" - parameter type ("range", "choice" or "fixed"), "bounds" for range parameters, "values" for choice parameters, and "value" for fixed parameters.
# - Dictionaries in `parameters` can optionally include "value_type" ("int", "float", "bool" or "str"), "log_scale" flag for range parameters, and "is_ordered" flag for choice parameters.
# - `parameter_constraints` should be a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
# - `outcome_constraints` should be a list of strings of form "constrained_metric <= some_bound".

# In[3]:


ax.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
    ],
    objective_name="hartmann6",
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
)


# ## 3. Define how to evaluate trials
# When using Ax a service, evaluation of parameterizations suggested by Ax is done either locally or, more commonly, using an external scheduler. Below is a dummy evaluation function that outputs data for two metrics "hartmann6" and "l2norm". Note that all returned metrics correspond to either the `objective_name` set on experiment creation or the metric names mentioned in `outcome_constraints`.

# In[4]:


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


# Result of the evaluation should generally be a mapping of the format: `{metric_name -> (mean, SEM)}`. If there is only one metric in the experiment – the objective – then evaluation function can return a single tuple of mean and SEM, in which case Ax will assume that evaluation corresponds to the objective. It can also return only the mean as a float, in which case Ax will treat SEM as unknown and use a model that can infer it. For more details on evaluation function, refer to the "Trial Evaluation" section in the docs.

# ## 4. Run optimization loop
# With the experiment set up, we can start the optimization loop.
# 
# At each step, the user queries the client for a new trial then submits the evaluation of that trial back to the client.
# 
# Note that Ax auto-selects an appropriate optimization algorithm based on the search space. For more advance use cases that require a specific optimization algorithm, pass a `generation_strategy` argument into the `AxClient` constructor. Note that when Bayesian Optimization is used, generating new trials may take a few minutes.

# In[5]:


for i in range(30):
    print(f"Running trial {i+1}/30...")
    parameters, trial_index = ax.get_next_trial()
     # Local evaluation here can be replaced with deployment to external system.
    ax.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# ## 5. Retrieve best parameters
# 
# Once it's complete, we can access the best parameters found, as well as the corresponding metric values.

# In[6]:


best_parameters, values = ax.get_best_parameters()
best_parameters


# In[7]:


means, covariances = values
means


# For comparison, Hartmann6 minimum:

# In[8]:


hartmann6.fmin


# ## 6. Plot the response surface and optimization trace
# Here we arbitrarily select "x1" and "x2" as the two parameters to plot for both metrics, "hartmann6" and "l2norm".

# In[9]:


render(
    plot_contour(
        model=ax.generation_strategy.model, param_x='x1', param_y='x2', metric_name='hartmann6'
    )
)


# We can also plot the optimization trace, showing the progression of finding the point with the optimal objective:

# In[10]:


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array([[trial.objective_mean for trial in ax.experiment.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.minimum.accumulate(best_objectives, axis=1),
    optimum=hartmann6.fmin,
    title="Model performance vs. # of iterations",
    ylabel="Hartmann6",
)
render(best_objective_plot)


# # Special Cases

# **Evaluation failure**: should any optimization iterations fail during evaluation, `log_trial_failure` will ensure that the same trial is not proposed again.

# In[11]:


_, trial_index = ax.get_next_trial()
ax.log_trial_failure(trial_index=trial_index)


# **Adding custom trials**: should there be need to evaluate a specific parameterization, `attach_trial` will add it to the experiment.

# In[12]:


ax.attach_trial(parameters={"x1": 9.0, "x2": 9.0, "x3": 9.0, "x4": 9.0, "x5": 9.0, "x6": 9.0})


# **Need to run many trials in parallel**: for optimal results and optimization efficiency, we strongly recommend sequential optimization (generating a few trials, then waiting for them to be completed with evaluation data). However, if your use case needs to dispatch many trials in parallel before they are updated with data and you are running into the *"All trials for current model have been generated, but not enough data has been observed to fit next model"* error, instantiate `AxClient` as `AxClient(enforce_sequential_optimization=False)`.
