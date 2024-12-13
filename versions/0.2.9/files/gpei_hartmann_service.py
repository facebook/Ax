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


from ax.service.ax_client import AxClient
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Initialize client
# 
# Create a client object to interface with Ax APIs. By default this runs locally without storage.

# In[2]:


ax_client = AxClient()


# ## 2. Set up experiment
# An experiment consists of a **search space** (parameters and parameter constraints) and **optimization configuration** (objective name, minimization setting, and outcome constraints). Note that:
# - Only `name`, `parameters`, and `objective_name` arguments are required.
# - Dictionaries in `parameters` have the following required keys: "name" - parameter name, "type" - parameter type ("range", "choice" or "fixed"), "bounds" for range parameters, "values" for choice parameters, and "value" for fixed parameters.
# - Dictionaries in `parameters` can optionally include "value_type" ("int", "float", "bool" or "str"), "log_scale" flag for range parameters, and "is_ordered" flag for choice parameters.
# - `parameter_constraints` should be a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
# - `outcome_constraints` should be a list of strings of form "constrained_metric <= some_bound".

# In[3]:


ax_client.create_experiment(
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


import numpy as np
def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


# Result of the evaluation should generally be a mapping of the format: `{metric_name -> (mean, SEM)}`. If there is only one metric in the experiment – the objective – then evaluation function can return a single tuple of mean and SEM, in which case Ax will assume that evaluation corresponds to the objective. _It can also return only the mean as a float, in which case Ax will treat SEM as unknown and use a model that can infer it._ 
# 
# For more details on evaluation function, refer to the "Trial Evaluation" section in the Ax docs at [ax.dev](https://ax.dev/)

# ## 4. Run optimization loop
# With the experiment set up, we can start the optimization loop.
# 
# At each step, the user queries the client for a new trial then submits the evaluation of that trial back to the client.
# 
# Note that Ax auto-selects an appropriate optimization algorithm based on the search space. For more advance use cases that require a specific optimization algorithm, pass a `generation_strategy` argument into the `AxClient` constructor. Note that when Bayesian Optimization is used, generating new trials may take a few minutes.

# In[5]:


for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# ### How many trials can run in parallel?
# By default, Ax restricts number of trials that can run in parallel for some optimization stages, in order to improve the optimization performance and reduce the number of trials that the optimization will require. To check the maximum parallelism for each optimization stage:

# In[6]:


ax_client.get_max_parallelism()


# The output of this function is a list of tuples of form (number of trials, max parallelism), so the example above means "the max parallelism is 12 for the first 12 trials and 3 for all subsequent trials." This is because the first 12 trials are produced quasi-randomly and can all be evaluated at once, and subsequent trials are produced via Bayesian optimization, which converges on optimal point in fewer trials when parallelism is limited. `MaxParallelismReachedException` indicates that the parallelism limit has been reached –– refer to the 'Service API Exceptions Meaning and Handling' section at the end of the tutorial for handling.

# ### How to view all existing trials during optimization?

# In[7]:


ax_client.generation_strategy.trials_as_df


# ## 5. Retrieve best parameters
# 
# Once it's complete, we can access the best parameters found, as well as the corresponding metric values.

# In[8]:


best_parameters, values = ax_client.get_best_parameters()
best_parameters


# In[9]:


means, covariances = values
means 


# For comparison, Hartmann6 minimum:

# In[10]:


hartmann6.fmin


# ## 6. Plot the response surface and optimization trace
# Here we arbitrarily select "x1" and "x2" as the two parameters to plot for both metrics, "hartmann6" and "l2norm".

# In[11]:


render(ax_client.get_contour_plot())


# We can also retrieve a contour plot for the other metric, "l2norm" –– say, we are interested in seeing the response surface for parameters "x3" and "x4" for this one.

# In[12]:


render(ax_client.get_contour_plot(param_x="x3", param_y="x4", metric_name="l2norm"))


# Here we plot the optimization trace, showing the progression of finding the point with the optimal objective:

# In[13]:


render(ax_client.get_optimization_trace(objective_optimum=hartmann6.fmin))  # Objective_optimum is optional.


# ## 7. Save / reload optimization to JSON / SQL
# We can serialize the state of optimization to JSON and save it to a `.json` file or save it to the SQL backend. For the former:

# In[14]:


ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.


# In[15]:


restored_ax_client = AxClient.load_from_json_file()  # For custom filepath, pass `filepath` argument.


# To store state of optimization to an SQL backend, first follow [setup instructions](https://ax.dev/docs/storage.html#sql) on Ax website.

# Having set up the SQL backend, pass `DBSettings` to `AxClient` on instantiation (note that `SQLAlchemy` dependency will have to be installed – for installation, refer to [optional dependencies](https://ax.dev/docs/installation.html#optional-dependencies) on Ax website):

# In[16]:


from ax.storage.sqa_store.structs import DBSettings

# URL is of the form "dialect+driver://username:password@host:port/database".
db_settings = DBSettings(url="sqlite:///foo.db")
# Instead of URL, can provide a `creator function`; can specify custom encoders/decoders if necessary.
new_ax = AxClient(db_settings=db_settings)


# When valid `DBSettings` are passed into `AxClient`, a unique experiment name is a required argument (`name`) to `ax_client.create_experiment`. The **state of the optimization is auto-saved** any time it changes (i.e. a new trial is added or completed, etc). 
# 
# To reload an optimization state later, instantiate `AxClient` with the same `DBSettings` and use `ax_client.load_experiment_from_database(experiment_name="my_experiment")`.

# # Special Cases

# **Evaluation failure**: should any optimization iterations fail during evaluation, `log_trial_failure` will ensure that the same trial is not proposed again.

# In[17]:


_, trial_index = ax_client.get_next_trial()
ax_client.log_trial_failure(trial_index=trial_index)


# **Adding custom trials**: should there be need to evaluate a specific parameterization, `attach_trial` will add it to the experiment.

# In[18]:


ax_client.attach_trial(parameters={"x1": 0.9, "x2": 0.9, "x3": 0.9, "x4": 0.9, "x5": 0.9, "x6": 0.9})


# **Need to run many trials in parallel**: for optimal results and optimization efficiency, we strongly recommend sequential optimization (generating a few trials, then waiting for them to be completed with evaluation data). However, if your use case needs to dispatch many trials in parallel before they are updated with data and you are running into the *"All trials for current model have been generated, but not enough data has been observed to fit next model"* error, instantiate `AxClient` as `AxClient(enforce_sequential_optimization=False)`.

# # Service API Exceptions Meaning and Handling
# [**`DataRequiredError`**](https://ax.dev/api/exceptions.html#ax.exceptions.core.DataRequiredError): Ax generation strategy needs to be updated with more data to proceed to the next optimization model. When the optimization moves from initialization stage to the Bayesian optimization stage, the underlying BayesOpt model needs sufficient data to train. For optimal results and optimization efficiency (finding the optimal point in the least number of trials), we recommend sequential optimization (generating a few trials, then waiting for them to be completed with evaluation data). Therefore, the correct way to handle this exception is to wait until more trial evaluations complete and log their data via `ax_client.complete_trial(...)`. 
# 
# However, if there is strong need to generate more trials before more data is available, instantiate `AxClient` as `AxClient(enforce_sequential_optimization=False)`. With this setting, as many trials will be generated from the initialization stage as requested, and the optimization will move to the BayesOpt stage whenever enough trials are completed.

# [**`MaxParallelismReachedException`**](https://ax.dev/api/modelbridge.html#ax.modelbridge.generation_strategy.MaxParallelismReachedException): generation strategy restricts the number of trials that can be ran simultaneously (to encourage sequential optimization), and the parallelism limit has been reached. The correct way to handle this exception is the same as `DataRequiredError` – to wait until more trial evluations complete and log their data via `ax_client.complete_trial(...)`.
#  
# In some cases higher parallelism is important, so `enforce_sequential_optimization=False` kwarg to AxClient allows to suppress limiting of parallelism. It's also possible to override the default parallelism setting for all stages of the optimization by passing `choose_generation_strategy_kwargs` to `ax_client.create_experiment`:

# In[19]:


ax_client = AxClient()
ax_client.create_experiment(
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
    ],
    # Sets max parallelism to 10 for all steps of the generation strategy.
    choose_generation_strategy_kwargs={"max_parallelism_override": 10},
)


# In[20]:


ax_client.get_max_parallelism()  # Max parallelism is now 10 for all stages of the optimization.

