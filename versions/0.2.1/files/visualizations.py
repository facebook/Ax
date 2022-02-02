#!/usr/bin/env python
# coding: utf-8

# # Visualizations
# 
# This tutorial illustrates the core visualization utilities available in Ax.

# In[1]:


import numpy as np
from ax.service.ax_client import AxClient

from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import(
    interact_fitted,
    plot_objective_vs_constraints,
    tile_fitted,
)
from ax.plot.slice import plot_slice
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Create experiment and run optimization
# 
# The vizualizations require an experiment object and a model fit on the evaluated data. The routine below is a copy of the Service API tutorial, so the explanation here is omitted. Retrieving the experiment and model objects for each API paradigm is shown in the respective tutorials

# #### 1a. Define search space and evaluation function

# In[2]:


noise_sd = 0.1
param_names = [f"x{i+1}" for i in range(6)]  # x1, x2, ..., x6

def noisy_hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(p_name) for p_name in param_names])
    noise1, noise2 = np.random.normal(0, noise_sd, 2)

    return {
        "hartmann6": (hartmann6(x) + noise1, noise_sd),
        "l2norm": (np.sqrt((x ** 2).sum()) + noise2, noise_sd)
    }


# #### 1b. Create Experiment

# In[3]:


ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters=[
        {
            "name": p_name,
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for p_name in param_names
    ],
    objective_name="hartmann6",
    minimize=True,
    outcome_constraints=["l2norm <= 1.25"]
)


# #### 1c. Run the optimization and fit a GP on all data

# In[4]:


for i in range(20):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=noisy_hartmann_evaluation_function(parameters))


# ## 2. Contour plots
# 
# The plot below shows the response surface for `hartmann6` metric as a function of the `x1`, `x2` parameters.
# 
# The other parameters are fixed in the middle of their respective ranges, which in this example is 0.5 for all of them.

# In[5]:


# this could alternately be done with `ax.plot.contour.plot_contour`
render(ax_client.get_contour_plot(param_x="x1", param_y="x2", metric_name='hartmann6'))


# #### 2a. Interactive contour plot
# 
# The plot below allows toggling between different pairs of parameters to view the contours.

# In[6]:


model = ax_client.generation_strategy.model
render(interact_contour(model=model, metric_name='hartmann6'))


# ## 3. Tradeoff plots
# This plot illustrates the tradeoffs achievable for 2 different metrics. The plot takes the x-axis metric as input (usually the objective) and allows toggling among all other metrics for the y-axis.
# 
# This is useful to get a sense of the pareto frontier (i.e. what is the best objective value achievable for different bounds on the constraint)

# In[7]:


render(plot_objective_vs_constraints(model, 'hartmann6', rel=False))


# ## 4. Cross-validation plots
# 
# CV plots are useful to check how well the model predictions calibrate against the actual measurements. If all points are close to the dashed line, then the model is a good predictor of the real data.

# In[8]:


cv_results = cross_validate(model)
render(interact_cross_validation(cv_results))


# ## 5. Slice plots
# 
# Slice plots show the metric outcome as a function of one parameter while fixing the others. They serve a similar function as contour plots.

# In[9]:


render(plot_slice(model, "x2", "hartmann6"))


# ## 6. Tile plots
# 
# Tile plots are useful for viewing the effect of each arm.

# In[10]:


render(interact_fitted(model, rel=False))

