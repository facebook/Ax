#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Optimization on Slurm via SubmitIt
# 
# This notebook serves as a quickstart guide for using the Ax library with the SubmitIt library in an ask-tell loop. [SubmitIt](https://github.com/facebookincubator/submitit/) is a Python toolbox for submitting jobs to [Slurm](https://slurm.schedmd.com/quickstart.html). 
# 
# The notebook demonstrates how to use the Ax client in an ask-tell loop where each trial is scheduled to run on a Slurm cluster asynchronously.
# 
# To use this script, run it on a slurm node either as an interactive notebook or export it as a Python script and run it as a Slurm job.
# 
# ## Importing Necessary Libraries
# Let's start by importing the necessary libraries.

# In[1]:


import time
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render
from ax.service.utils.report_utils import exp_to_df
from submitit import AutoExecutor, LocalJob, DebugJob


# ## Defining the Function to Optimize
# We'll define a simple function to optimize. This function takes two parameters, and returns a single metric.

# In[2]:


def evaluate(parameters):
    x = parameters["x"]
    y = parameters["y"]
    return {"result": (x - 3)**2 + (y - 4)**2}


# Note: SubmitIt's [CommandFunction](https://github.com/facebookincubator/submitit/blob/main/docs/examples.md#working-with-commands) allows you to define commands to run on the node and then redirects the standard output.
# 
# ## Setting up Ax
# We'll use Ax's Service API for this example. We start by initializing an AxClient and creating an experiment.

# In[3]:


ax_client = AxClient()
ax_client.create_experiment(
    name="my_experiment",
    parameters=[
        {"name": "x", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "y", "type": "range", "bounds": [-10.0, 10.0]},
    ],
    objectives={"result": ObjectiveProperties(minimize=True)},
    parameter_constraints=["x + y <= 2.0"],  # Optional.
)


# Other commonly used [parameters types](https://ax.dev/docs/glossary.html#parameter) include `choice` parameters and `fixed` parameters. 
# 
# Tip 1: you can specify additional information for parameters such as `log_scale`, if a parameter operates at a log-scale and `is_ordered` for choice parameters that have a meaningful ordering.
# 
# Tip 2: Ax is an excellent choice for multi-objective optimization problems when there are multiple competing objectives and the goal is to find all Pareto-optimal solutions.
# 
# Tip 3: One can define constraints on both the parameters and the outcome.
# 
# ## Setting up SubmitIt
# We'll use SubmitIt's `AutoExecutor` for this example. We start by initializing an `AutoExecutor`, and setting a few commonly used parameters.

# In[4]:


# Log folder and cluster. Specify cluster='local' or cluster='debug' to run the jobs locally during development.
# When we're are ready for deployment, switch to cluster='slurm' 
executor = AutoExecutor(folder="/tmp/submitit_runs", cluster='debug') 
executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.
executor.update_parameters(cpus_per_task=2)


# Other commonly used Slurm parameters include `partition`, `ntasks_per_node`, `cpus_per_task`, `cpus_per_gpu`, `gpus_per_node`, `gpus_per_task`, `qos`, `mem`, `mem_per_gpu`, `mem_per_cpu`, `account`.
# 
# ## Running the Optimization Loop
# Now, we're ready to run the optimization loop. We'll use an ask-tell loop, where we ask Ax for a suggestion, evaluate it using our function, and then tell Ax the result.
# 
# The example loop schedules new jobs whenever there is availability. For tasks that take a similar amount of time regardless of the parameters, it may make more sense to wait for the whole batch to finish before scheduling the next (so ax can make better informed parameter choices).
# 
# Note that `get_next_trials` may not use all available `num_parallel_jobs` if it doesn't have good parameter candidates to run.

# In[5]:


total_budget = 10
num_parallel_jobs = 3

jobs = []
submitted_jobs = 0
# Run until all the jobs have finished and our budget is used up.
while submitted_jobs < total_budget or jobs:
    for job, trial_index in jobs[:]:
        # Poll if any jobs completed
        # Local and debug jobs don't run until .result() is called.
        if job.done() or type(job) in [LocalJob, DebugJob]:
            result = job.result()
            ax_client.complete_trial(trial_index=trial_index, raw_data=result)
            jobs.remove((job, trial_index))
    
    # Schedule new jobs if there is availablity
    trial_index_to_param, _ = ax_client.get_next_trials(
        max_trials=min(num_parallel_jobs - len(jobs), total_budget - submitted_jobs))
    for trial_index, parameters in trial_index_to_param.items():
        job = executor.submit(evaluate, parameters)
        submitted_jobs += 1
        jobs.append((job, trial_index))
        time.sleep(1)
    
    # Display the current trials.
    display(exp_to_df(ax_client.experiment))

    # Sleep for a bit before checking the jobs again to avoid overloading the cluster. 
    # If you have a large number of jobs, consider adding a sleep statement in the job polling loop aswell.
    time.sleep(30)


# 
# ## Finally
# 
# We can retrieve the best parameters and render the response surface.

# In[6]:


best_parameters, (means, covariances) = ax_client.get_best_parameters()
print(f'Best set of parameters: {best_parameters}')
print(f'Mean objective value: {means}')
# The covariance is only meaningful when multiple objectives are present.

render(ax_client.get_contour_plot())

