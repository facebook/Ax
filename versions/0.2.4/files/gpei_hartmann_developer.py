#!/usr/bin/env python
# coding: utf-8

# # Developer API Example on Hartmann6
# 
# The Developer API is suitable when the user wants maximal customization of the optimization loop. This tutorial demonstrates optimization of a Hartmann6 function using the `Experiment` construct. In this example, trials will be evaluated synchronously.

# In[1]:


from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    SearchSpace, 
    Experiment, 
    OutcomeConstraint, 
    OrderConstraint,
    SumConstraint,
    OptimizationConfig,
    Objective,
    Metric,
)
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# ## 1. Create Search Space
# 
# First, we define a search space, which defines the type and allowed range for the parameters.

# In[2]:


hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(6)
    ]
)


# Note that there are two other parameter classes, FixedParameter and ChoiceParameter. Although we won't use these in this example, you can create them as follows.
# 
# 

# In[3]:


choice_param = ChoiceParameter(name="choice", values=["foo", "bar"], parameter_type=ParameterType.STRING)
fixed_param = FixedParameter(name="fixed", value=[True], parameter_type=ParameterType.BOOL)


# Sum constraints enforce that the sum of a set of parameters is greater or less than some bound, and order constraints enforce that one parameter is smaller than the other. We won't use these either, but see two examples below.
# 
# 

# In[4]:


sum_constraint = SumConstraint(
    parameters=[hartmann_search_space.parameters['x0'], hartmann_search_space.parameters['x1']], 
    is_upper_bound=True, 
    bound=5.0,
)

order_constraint = OrderConstraint(
    lower_parameter = hartmann_search_space.parameters['x0'],
    upper_parameter = hartmann_search_space.parameters['x1'],
)


# ## 2. Create Optimization Config
# 
# Second, we define the `optimization_config` with an `objective` and `outcome_constraints`.
# 
# When doing the optimization, we will find points that minimize the objective while obeying the constraints (which in this case means `l2norm < 1.25`).
# 
# Note: we are using `Hartmann6Metric` and `L2NormMetric` here, which have built in evaluation functions for testing.  For creating your own cutom metrics, see [8. Defining custom metrics](#8.-Defining-custom-metrics).

# In[5]:


from ax.metrics.l2norm import L2NormMetric
from ax.metrics.hartmann6 import Hartmann6Metric

param_names = [f"x{i}" for i in range(6)]
optimization_config = OptimizationConfig(
    objective = Objective(
        metric=Hartmann6Metric(name="hartmann6", param_names=param_names), 
        minimize=True,
    ),
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=param_names, noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False,
        )
    ],
)


# ## 3. Define a Runner
# Before an experiment can collect data, it must have a Runner attached. A runner handles the deployment of trials. A trial must be "run" before it can be evaluated.
# 
# Here, we have a dummy runner that does nothing. In practice, a runner might be in charge of pushing an experiment to production.
# 
# The only method that needs to be defined for runner subclasses is run, which performs any necessary deployment logic, and returns a dictionary of resulting metadata.  This metadata can later be accessed through the trial's `run_metadata` property.

# In[6]:


from ax import Runner

class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata


# ## 4. Create Experiment
# Next, we make an `Experiment` with our search space, runner, and optimization config.

# In[7]:


exp = Experiment(
    name="test_hartmann",
    search_space=hartmann_search_space,
    optimization_config=optimization_config,
    runner=MyRunner(),
)


# ## 5. Perform Optimization
# 
# Run the optimization using the settings defined on the experiment. We will create 5 random sobol points for exploration followed by 15 points generated using the GPEI optimizer.
# 
# Instead of a member of the `Models` enum to produce generator runs, users can leverage a `GenerationStrategy`. See the [Generation Strategy Tutorial](https://ax.dev/tutorials/generation_strategy.html) for more info.

# In[8]:


from ax.modelbridge.registry import Models

NUM_SOBOL_TRIALS = 5
NUM_BOTORCH_TRIALS = 15

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=exp.search_space)
    
for i in range(NUM_SOBOL_TRIALS):
    # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
    generator_run = sobol.gen(n=1)
    # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    trial = exp.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    # Mark trial as completed to record when a trial run is completed 
    # and enable fetching of data for metrics on the experiment 
    # (by default, trials must be completed before metrics can fetch their data,
    # unless a metric is explicitly configured otherwise)
    trial.mark_completed()

for i in range(NUM_BOTORCH_TRIALS):
    print(
        f"Running GP+EI optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
    )
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(n=1)
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()
    
print("Done!")


# ## 6. Inspect trials' data
# 
# Now we can inspect the `Experiment`'s data by calling `fetch_data()`, which retrieves evaluation data for all trials of the experiment.
# 
# To fetch trial data, we need to run it and mark it completed. For most metrics in Ax, data is only available once the status of the trial is `COMPLETED`, since in real-worlds scenarios, metrics can typically only be fetched after the trial finished running.
# 
# NOTE: Metrics classes may implement the `is_available_while_running` method. When this method returns `True`, data is available when trials are either `RUNNING` or `COMPLETED`. This can be used to obtain intermediate results from A/B test trials and other online experiments, or when metric values are available immediately, like in the case of synthetic problem metrics.
# 
# We can also use the `fetch_trials_data` function to get evaluation data for a specific trials in the experiment, like so:

# In[9]:


trial_data = exp.fetch_trials_data([NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS - 1])
trial_data.df


# The below call to `exp.fetch_data()` also attaches data to the last trial, which because of the way we looped through Botorch trials in [5. Perform Optimization](5.-Perform-Optimization), would otherwise not have data attached.  This is necessary to get `objective_means` in [7. Plot results](7.-Plot-results).

# In[10]:


exp.fetch_data().df


# ## 7. Plot results
# Now we can plot the results of our optimization:

# In[11]:


import numpy as np
from ax.plot.trace import optimization_trace_single_method

# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
objective_means = np.array([[trial.objective_mean for trial in exp.trials.values()]])
best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(objective_means, axis=1),
        optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
)
render(best_objective_plot)


# ## 8. Defining custom metrics
# In order to perform an optimization, we also need to define an optimization config for the experiment. An optimization config is composed of an objective metric to be minimized or maximized in the experiment, and optionally a set of outcome constraints that place restrictions on how other metrics can be moved by the experiment.
# 
# In order to define an objective or outcome constraint, we first need to subclass Metric. Metrics are used to evaluate trials, which are individual steps of the experiment sequence. Each trial contains one or more arms for which we will collect data at the same time.
# 
# Our custom metric(s) will determine how, given a trial, to compute the mean and SEM of each of the trial's arms.
# 
# The only method that needs to be defined for most metric subclasses is `fetch_trial_data`, which defines how a single trial is evaluated, and returns a pandas dataframe.
#  
# The `is_available_while_running` method is optional and returns a boolean, specifying whether the trial data can be fetched before the trial is complete.  See [6. Inspect trials' data](6.-Inspect-trials'-data) for more details.

# In[12]:


from ax import Data
import pandas as pd


class BoothMetric(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                # in practice, the mean and sem will be looked up based on trial metadata
                # but for this tutorial we will calculate them
                "mean": (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2,
                "sem": 0.0,
            })
        return Data(df=pd.DataFrame.from_records(records))

    def is_available_while_running(self) -> bool:
        return True


# ## 9. Save to JSON or SQL
# At any point, we can also save our experiment to a JSON file. To ensure that our custom metrics and runner are saved properly, we first need to register them.

# In[13]:


from ax.storage.registry_bundle import RegistryBundle

bundle = RegistryBundle(
    metric_clss={BoothMetric: None, L2NormMetric: None, Hartmann6Metric: None},
    runner_clss={MyRunner: None}
)

from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

save_experiment(exp, "experiment.json", encoder_registry=bundle.encoder_registry)


# In[14]:


loaded_experiment = load_experiment("experiment.json", decoder_registry=bundle.decoder_registry)


# To save our experiment to SQL, we must first specify a connection to a database and create all necessary tables.
# 
# 

# In[15]:


from ax.storage.sqa_store.db import init_engine_and_session_factory, get_engine, create_all_tables
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url='sqlite:///foo3.db')

engine = get_engine()
create_all_tables(engine)


# In[16]:


from ax.storage.sqa_store.sqa_config import SQAConfig

exp.name = "new"

sqa_config = SQAConfig(
    json_encoder_registry=bundle.encoder_registry,
    json_decoder_registry=bundle.decoder_registry,
    metric_registry=bundle.metric_registry,
    runner_registry=bundle.runner_registry,
)

save_experiment(exp, config=sqa_config)


# In[17]:


load_experiment(exp.name, config=sqa_config)


# In[ ]:




