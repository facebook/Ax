#!/usr/bin/env python
# coding: utf-8

# # Developer API Example on Hartmann6
# 
# The Developer API is suitable when the user wants maximal customization of the optimization loop. This tutorial demonstrates optimization of a Hartmann6 function using the `SimpleExperiment` construct, which we use for synchronous experiments, where trials can be evaluated right away.
# 
# For more details on the different Ax constructs, see the "Building Blocks of Ax" tutorial.

# In[1]:


import numpy as np
from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    SearchSpace, 
    SimpleExperiment, 
    OutcomeConstraint, 
)
from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.registry import Models
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Define evaluation function
# 
# First, we define an evaluation function that is able to compute all the metrics needed for this experiment. This function needs to accept a set of parameter values and can also accept a weight. It should produce a dictionary of metric names to tuples of mean and standard error for those metrics. Note that when using `Experiment` (instead of `SimpleExperiment`), it's possible to deploy trials and fetch their evaluation results asynchronously; more on that in the "Building Blocks of Ax" tutorial.

# In[2]:


def hartmann_evaluation_function(
    parameterization, # Mapping of parameter names to values of those parameters.
    weight=None, # Optional weight argument.
):
    x = np.array([parameterization.get(f"x{i}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


# If there is only one metric in the experiment – the objective – then evaluation function can return a single tuple of mean and SEM, in which case Ax will assume that evaluation corresponds to the objective. It can also return only the mean as a float, in which case Ax will treat SEM as unknown and use a model that can infer it. For more details on evaluation function, refer to the "Trial Evaluation" section in the docs.

# ## 2. Create Search Space
# 
# Second, we define a search space, which defines the type and allowed range for the parameters.

# In[3]:


hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(6)
    ]
)


# ## 3. Create Experiment
# 
# Third, we make a `SimpleExperiment`. In addition to the search space and evaluation function, here we define the `objective_name` and `outcome_constraints`.
# 
# When doing the optimization, we will find points that minimize the objective while obeying the constraints (which in this case means `l2norm < 1.25`).

# In[4]:


exp = SimpleExperiment(
    name="test_hartmann",
    search_space=hartmann_search_space,
    evaluation_function=hartmann_evaluation_function,
    objective_name="hartmann6",
    minimize=True,
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False,
        )
    ],
)


# ## 4. Perform Optimization
# 
# Run the optimization using the settings defined on the experiment. We will create 5 random sobol points for exploration followed by 15 points generated using the GPEI optimizer.

# In[5]:


print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))
    
data = exp.eval()
for i in range(15):
    print(f"Running GP+EI optimization trial {i+1}/15...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=data)
    batch = exp.new_trial(generator_run=gpei.gen(1))
    data = exp.eval()
    
print("Done!")


# ## 5. Inspect trials' data
# 
# Now we can inspect the `SimpleExperiment`'s data by calling `eval()`, which retrieves evaluation data for all batches of the experiment.
# 
# We can also use the `eval_trial` function to get evaluation data for a specific trial in the experiment, like so:

# In[6]:


trial_data = exp.eval_trial(exp.trials[1])
trial_data.df


# ## 6. Plot results
# Now we can plot the results of our optimization:

# In[7]:


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
)
render(best_objective_plot)

