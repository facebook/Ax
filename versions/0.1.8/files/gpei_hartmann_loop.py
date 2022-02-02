#!/usr/bin/env python
# coding: utf-8

# # Loop API Example on Hartmann6
# 
# The loop API is the most lightweight way to do optimization in Ax. The user makes one call to `optimize`, which performs all of the optimization under the hood and returns the optimized parameters.
# 
# For more customizability of the optimization procedure, consider the Service or Developer API.

# In[1]:


import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Define evaluation function
# 
# First, we define an evaluation function that is able to compute all the metrics needed for this experiment. This function needs to accept a set of parameter values and can also accept a weight. It should produce a dictionary of metric names to tuples of mean and standard error for those metrics.

# In[2]:


def hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


# If there is only one metric in the experiment – the objective – then evaluation function can return a single tuple of mean and SEM, in which case Ax will assume that evaluation corresponds to the objective. It can also return only the mean as a float, in which case Ax will treat SEM as unknown and use a model that can infer it. For more details on evaluation function, refer to the "Trial Evaluation" section in the docs.

# ## 2. Run optimization
# The setup for the loop is fully compatible with JSON. The optimization algorithm is selected based on the properties of the problem search space.

# In[3]:


best_parameters, values, experiment, model = optimize(
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
    experiment_name="test",
    objective_name="hartmann6",
    evaluation_function=hartmann_evaluation_function,
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 <= 20"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
    total_trials=30, # Optional.
)


# And we can introspect optimization results:

# In[4]:


best_parameters


# In[5]:


means, covariances = values
means


# For comparison, minimum of Hartmann6 is:

# In[6]:


hartmann6.fmin


# ## 3. Plot results
# Here we arbitrarily select "x1" and "x2" as the two parameters to plot for both metrics, "hartmann6" and "l2norm".

# In[7]:


render(plot_contour(model=model, param_x='x1', param_y='x2', metric_name='hartmann6'))


# In[8]:


render(plot_contour(model=model, param_x='x1', param_y='x2', metric_name='l2norm'))


# We also plot optimization trace, which shows best hartmann6 objective value seen by each iteration of the optimization:

# In[9]:


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.minimum.accumulate(best_objectives, axis=1),
    optimum=hartmann6.fmin,
    title="Model performance vs. # of iterations",
    ylabel="Hartmann6",
)
render(best_objective_plot)

