#!/usr/bin/env python
# coding: utf-8

# # Tune a CNN on MNIST
# 
# This tutorial walks through using Ax to tune two hyperparameters (learning rate and momentum) for a PyTorch CNN on the MNIST dataset trained using SGD with momentum.

# In[1]:


import torch

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train

import torch.nn as nn
import torch.nn.functional as F
from torch._tensor import Tensor
from torch.utils.data import DataLoader

init_notebook_plotting()


# In[2]:


torch.manual_seed(42)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## 1. Load MNIST data
# First, we need to load the MNIST data and partition it into training, validation, and test sets.
# 
# Note: this will download the dataset if necessary.

# In[3]:


BATCH_SIZE = 512
train_loader, valid_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)


# ## 2. Initialize Client
# Create a client object to interface with Ax APIs. By default this runs locally without storage.
# 
#  

# In[4]:


ax_client = AxClient()


# ## 3. Set up experiment
# An experiment consists of a **search space** (parameters and parameter constraints) and **optimization configuration** (objective name, minimization setting, and outcome constraints).

# In[5]:


# Create an experiment with required arguments: name, parameters, and objective_name.
ax_client.create_experiment(
    name="tune_cnn_on_mnist",  # The name of the experiment.
    parameters=[
        {
            "name": "lr",  # The name of the parameter.
            "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
            "bounds": [1e-6, 0.4],  # The bounds for range parameters. 
            # "values" The possible values for choice parameters .
            # "value" The fixed value for fixed parameters.
            "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
            "log_scale": True,  # Optional, whether to use a log scale for range parameters. Defaults to False.
            # "is_ordered" Optional, a flag for choice parameters.
        },
        {
            "name": "momentum",  
            "type": "range",  
            "bounds": [0.0, 1.0],  
        },
    ],
    objectives={"accuracy": ObjectiveProperties(minimize=False)},  # The objective name and minimization setting.
    # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
    # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
)


# ## 4. Define how to evaluate trials
# 

# First we define a simple CNN class to classify the MNIST images

# In[6]:


class CNN(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8 * 8 * 20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 8 * 8 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# In this tutorial, we want to optimize classification accuracy on the validation set as a function of the learning rate and momentum. The `train_evaluate` function takes in a parameterization (set of parameter values), computes the classification accuracy, and returns that metric. 

# In[7]:


def train_evaluate(parameterization):
    """
    Train the model and then compute an evaluation metric.

    In this tutorial, the CNN utils package is doing a lot of work
    under the hood:
        - `train` initializes the network, defines the loss function
        and optimizer, performs the training loop, and returns the
        trained model.
        - `evaluate` computes the accuracy of the model on the
        evaluation dataset and returns the metric.

    For your use case, you can define training and evaluation functions
    of your choosing.

    """
    net = CNN()
    net = train(
        net=net,
        train_loader=train_loader,
        parameters=parameterization,
        dtype=dtype,
        device=device,
    )

    return evaluate(
        net=net, 
        data_loader=valid_loader, 
        dtype=dtype, 
        device=device,
    )


# ## 5. Run optimization loop
# 

# First we use `attach_trial` to attach a custom trial with manually-chosen parameters. This step is optional, but we include it here to demonstrate adding manual trials and to serve as a baseline model with decent performance. 

# In[8]:


# Attach the trial
ax_client.attach_trial(
    parameters={"lr": 0.000026, "momentum": 0.58}
)

# Get the parameters and run the trial 
baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
ax_client.complete_trial(trial_index=0, raw_data=train_evaluate(baseline_parameters))


# Now we start the optimization loop.
# 
# At each step, the user queries the client for a new trial then submits the evaluation of that trial back to the client.
# 
# Note that Ax auto-selects an appropriate optimization algorithm based on the search space. For more advanced use cases that require a specific optimization algorithm, pass a `generation_strategy` argument into the `AxClient` constructor. Note that when Bayesian Optimization is used, generating new trials may take a few minutes.

# In[9]:


for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))


# ### How many trials can run in parallel?
# By default, Ax restricts number of trials that can run in parallel for some optimization stages, in order to improve the optimization performance and reduce the number of trials that the optimization will require. To check the maximum parallelism for each optimization stage:

# In[10]:


ax_client.get_max_parallelism()


# The output of this function is a list of tuples of form (number of trials, max parallelism), so the example above means "the max parallelism is 5 for the first 5 trials and 3 for all subsequent trials." This is because the first 5 trials are produced quasi-randomly and can all be evaluated at once, and subsequent trials are produced via Bayesian optimization, which converges on optimal point in fewer trials when parallelism is limited. MaxParallelismReachedException indicates that the parallelism limit has been reached –– refer to the 'Service API Exceptions Meaning and Handling' section at the end of the tutorial for handling.
# 
# 

# ### How to view all existing trials during optimization?

# In[11]:


ax_client.get_trials_data_frame()


# ## 6. Retrieve best parameters
# 
# Once it's complete, we can access the best parameters found, as well as the corresponding metric values. Note that these parameters may not necessarily be the set that yielded the highest _observed_ accuracy because Ax uses the highest model _predicted_ accuracy to choose the best parameters (see [here](https://ax.dev/api/service.html#module-ax.service.utils.best_point_mixin) for more details). Due to randomness in the data or the algorithm itself, using observed accuracy may result in choosing an outlier for the best set of parameters. Using the model predicted best will use the model to regularize the observations and reduce the likelihood of picking some outlier in the data.

# In[12]:


best_parameters, values = ax_client.get_best_parameters()
best_parameters


# In[13]:


mean, covariance = values
mean


# ## 7. Plot the response surface and optimization trace
# 
# Contour plot showing classification accuracy as a function of the two hyperparameters.
# 
# The black squares show points that we have actually run; notice how they are clustered in the optimal region.

# In[14]:


render(ax_client.get_contour_plot(param_x="lr", param_y="momentum", metric_name="accuracy"))


# Here we plot the optimization trace, showing the progression of finding the point with the optimal objective:
# 
# 

# In[15]:


render(
    ax_client.get_optimization_trace()
)  


# ## 8. Train CNN with best hyperparameters and evaluate on test set
# Note that the resulting accuracy on the test set generally won't be the same as the maximum accuracy achieved on the evaluation set throughout optimization. 

# In[16]:


df = ax_client.get_trials_data_frame()
best_arm_idx = df.trial_index[df["accuracy"] == df["accuracy"].max()].values[0]
best_arm = ax_client.get_trial_parameters(best_arm_idx)
best_arm


# In[17]:


combined_train_valid_set = torch.utils.data.ConcatDataset(
    [
        train_loader.dataset.dataset,
        valid_loader.dataset.dataset,
    ]
)
combined_train_valid_loader = torch.utils.data.DataLoader(
    combined_train_valid_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# In[18]:


net = train(
    net=CNN(),
    train_loader=combined_train_valid_loader,
    parameters=best_arm,
    dtype=dtype,
    device=device,
)
test_accuracy = evaluate(
    net=net,
    data_loader=test_loader,
    dtype=dtype,
    device=device,
)


# In[19]:


print(f"Classification Accuracy (test set): {round(test_accuracy*100, 2)}%")


# ## 9. Save / reload optimization to JSON / SQL
# We can serialize the state of optimization to JSON and save it to a `.json` file or save it to the SQL backend. For the former:

# In[20]:


ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.


# In[21]:


restored_ax_client = (
    AxClient.load_from_json_file()
)  # For custom filepath, pass `filepath` argument.


# To store state of optimization to an SQL backend, first follow [setup instructions](https://ax.dev/docs/storage.html#sql) on Ax website.

# Having set up the SQL backend, pass `DBSettings` to `AxClient` on instantiation (note that `SQLAlchemy` dependency will have to be installed – for installation, refer to [optional dependencies](https://ax.dev/docs/installation.html#optional-dependencies) on Ax website):

# In[22]:


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

# In[23]:


_, trial_index = ax_client.get_next_trial()
ax_client.log_trial_failure(trial_index=trial_index)


# **Need to run many trials in parallel**: for optimal results and optimization efficiency, we strongly recommend sequential optimization (generating a few trials, then waiting for them to be completed with evaluation data). However, if your use case needs to dispatch many trials in parallel before they are updated with data and you are running into the *"All trials for current model have been generated, but not enough data has been observed to fit next model"* error, instantiate `AxClient` as `AxClient(enforce_sequential_optimization=False)`.

# # Service API Exceptions Meaning and Handling
# [**`DataRequiredError`**](https://ax.dev/api/exceptions.html#ax.exceptions.core.DataRequiredError): Ax generation strategy needs to be updated with more data to proceed to the next optimization model. When the optimization moves from initialization stage to the Bayesian optimization stage, the underlying BayesOpt model needs sufficient data to train. For optimal results and optimization efficiency (finding the optimal point in the least number of trials), we recommend sequential optimization (generating a few trials, then waiting for them to be completed with evaluation data). Therefore, the correct way to handle this exception is to wait until more trial evaluations complete and log their data via `ax_client.complete_trial(...)`. 
# 
# However, if there is strong need to generate more trials before more data is available, instantiate `AxClient` as `AxClient(enforce_sequential_optimization=False)`. With this setting, as many trials will be generated from the initialization stage as requested, and the optimization will move to the BayesOpt stage whenever enough trials are completed.

# [**`MaxParallelismReachedException`**](https://ax.dev/api/modelbridge.html#ax.modelbridge.generation_strategy.MaxParallelismReachedException): generation strategy restricts the number of trials that can be run simultaneously (to encourage sequential optimization), and the parallelism limit has been reached. The correct way to handle this exception is the same as `DataRequiredError` – to wait until more trial evluations complete and log their data via `ax_client.complete_trial(...)`.
#  
# In some cases higher parallelism is important, so `enforce_sequential_optimization=False` kwarg to AxClient allows the user to suppress limiting of parallelism. It's also possible to override the default parallelism setting for all stages of the optimization by passing `choose_generation_strategy_kwargs` to `ax_client.create_experiment`:

# In[24]:


ax_client = AxClient()
ax_client.create_experiment(
    parameters=[
        {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
        {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
    ],
    # Sets max parallelism to 10 for all steps of the generation strategy.
    choose_generation_strategy_kwargs={"max_parallelism_override": 10},
)


# In[25]:


ax_client.get_max_parallelism()  # Max parallelism is now 10 for all stages of the optimization.

