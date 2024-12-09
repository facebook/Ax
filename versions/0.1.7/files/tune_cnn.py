#!/usr/bin/env python
# coding: utf-8

# # Tune a CNN on MNIST
# 
# This tutorial walks through using Ax to tune two hyperparameters (learning rate and momentum) for a PyTorch CNN on the MNIST dataset trained using SGD with momentum.
# 

# In[1]:


import torch
import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN

init_notebook_plotting()


# In[2]:


torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## 1. Load MNIST data
# First, we need to load the MNIST data and partition it into training, validation, and test sets.
# 
# Note: this will download the dataset if necessary.

# In[3]:


BATCH_SIZE = 512
train_loader, valid_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)


# ## 2. Define function to optimize
# In this tutorial, we want to optimize classification accuracy on the validation set as a function of the learning rate and momentum. The function takes in a parameterization (set of parameter values), computes the classification accuracy, and returns a dictionary of metric name ('accuracy') to a tuple with the mean and standard error.

# In[4]:


def train_evaluate(parameterization):
    net = CNN()
    net = train(net=net, train_loader=train_loader, parameters=parameterization, dtype=dtype, device=device)
    return evaluate(
        net=net,
        data_loader=valid_loader,
        dtype=dtype,
        device=device,
    )


# ## 3. Run the optimization loop
# Here, we set the bounds on the learning rate and momentum and set the parameter space for the learning rate to be on a log scale. 

# In[5]:


best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    ],
    evaluation_function=train_evaluate,
    objective_name='accuracy',
)


# We can introspect the optimal parameters and their outcomes:

# In[6]:


best_parameters


# In[7]:


means, covariances = values
means, covariances


# ## 4. Plot response surface
# 
# Contour plot showing classification accuracy as a function of the two hyperparameters.
# 
# The black squares show points that we have actually run, notice how they are clustered in the optimal region.

# In[8]:


render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy'))


# ## 5. Plot best objective as function of the iteration
# 
# Show the model accuracy improving as we identify better hyperparameters.

# In[9]:


# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)
render(best_objective_plot)


# ## 6. Train CNN with best hyperparameters and evaluate on test set
# Note that the resulting accuracy on the test set might not be exactly the same as the maximum accuracy achieved on the evaluation set throughout optimization. 

# In[10]:


data = experiment.fetch_data()
df = data.df
best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
best_arm = experiment.arms_by_name[best_arm_name]
best_arm


# In[11]:


combined_train_valid_set = torch.utils.data.ConcatDataset([
    train_loader.dataset.dataset, 
    valid_loader.dataset.dataset,
])
combined_train_valid_loader = torch.utils.data.DataLoader(
    combined_train_valid_set, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
)


# In[12]:


net = train(
    net=CNN(),
    train_loader=combined_train_valid_loader, 
    parameters=best_arm.parameters,
    dtype=dtype,
    device=device,
)
test_accuracy = evaluate(
    net=net,
    data_loader=test_loader,
    dtype=dtype,
    device=device,
)


# In[13]:


print(f"Classification Accuracy (test set): {round(test_accuracy*100, 2)}%")

