#!/usr/bin/env python
# coding: utf-8

# # Ax Service API with RayTune on PyTorch CNN
# 
# Ax integrates easily with different scheduling frameworks and distributed training frameworks. In this example, Ax-driven optimization is executed in a distributed fashion using [RayTune](https://ray.readthedocs.io/en/latest/tune.html). 
# 
# RayTune is a scalable framework for hyperparameter tuning that provides many state-of-the-art hyperparameter tuning algorithms and seamlessly scales from laptop to distributed cluster with fault tolerance. RayTune leverages [Ray](https://ray.readthedocs.io/)'s Actor API to provide asynchronous parallel and distributed execution.
# 
# Ray 'Actors' are a simple and clean abstraction for replicating your Python classes across multiple workers and nodes. Each hyperparameter evaluation is asynchronously executed on a separate Ray actor and reports intermediate training progress back to RayTune. Upon reporting, RayTune then uses this information to performs actions such as early termination, re-prioritization, or checkpointing.

# In[1]:


import logging

from ray import tune
from ray.tune import report
from ray.tune.suggest.ax import AxSearch

logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.


# In[2]:


import numpy as np
import torch
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import CNN, evaluate, load_mnist, train

init_notebook_plotting()


# ## 1. Initialize client
# We specify `enforce_sequential_optimization` as False, because Ray runs many trials in parallel. With the sequential optimization enforcement, `AxClient` would expect the first few trials to be completed with data before generating more trials.
# 
# When high parallelism is not required, it is best to enforce sequential optimization, as it allows for achieving optimal results in fewer (but sequential) trials. In cases where parallelism is important, such as with distributed training using Ray, we choose to forego minimizing resource utilization and run more trials in parallel.

# In[3]:


ax = AxClient(enforce_sequential_optimization=False)


# ## 2. Set up experiment
# Here we set up the search space and specify the objective; refer to the Ax API tutorials for more detail.

# In[4]:


MINIMIZE = False  # Whether we should be minimizing or maximizing the objective


# In[5]:


ax.create_experiment(
    name="mnist_experiment",
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    objective_name="mean_accuracy",
    minimize=MINIMIZE,
)


# In[6]:


ax.experiment.optimization_config.objective.minimize


# In[7]:


load_mnist(data_path="~/.data")  # Pre-load the dataset before the initial evaluations are executed.


# ## 3. Define how to evaluate trials
# Since we use the Ax Service API here, we evaluate the parameterizations that Ax suggests, using RayTune. The evaluation function follows its usual pattern, taking in a parameterization and outputting an objective value. For detail on evaluation functions, see [Trial Evaluation](https://ax.dev/docs/runner.html). 

# In[8]:


def train_evaluate(parameterization):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = load_mnist(data_path="~/.data")
    net = train(
        net=CNN(),
        train_loader=train_loader,
        parameters=parameterization,
        dtype=torch.float,
        device=device,
    )
    report(
        mean_accuracy=evaluate(
            net=net,
            data_loader=valid_loader,
            dtype=torch.float,
            device=device,
        )
    )


# ## 4. Run optimization
# Execute the Ax optimization and trial evaluation in RayTune using [AxSearch algorithm](https://ray.readthedocs.io/en/latest/tune-searchalg.html#ax-search):

# In[9]:


tune.run(
    train_evaluate,
    num_samples=30,
    search_alg=AxSearch(
        ax_client=ax, 
        max_concurrent=3, 
        mode="min" if MINIMIZE else "max",  # Ensure that `mode` aligns with the `minimize` setting in `AxClient`
    ),
    verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
    # To use GPU, specify: resources_per_trial={"gpu": 1}.
)


# ## 5. Retrieve the optimization results

# In[10]:


best_parameters, values = ax.get_best_parameters()
best_parameters


# In[11]:


means, covariances = values
means


# ## 6. Plot the response surface and optimization trace

# In[12]:


render(
    plot_contour(
        model=ax.generation_strategy.model,
        param_x="lr",
        param_y="momentum",
        metric_name="mean_accuracy",
    )
)


# In[13]:


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array(
    [[trial.objective_mean * 100 for trial in ax.experiment.trials.values()]]
)
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Accuracy",
)
render(best_objective_plot)

