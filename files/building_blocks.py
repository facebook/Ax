#!/usr/bin/env python
# coding: utf-8

# # Building Blocks of Ax

# This tutorial illustrates the core Ax classes and their usage by constructing, running, and saving an experiment through the Developer API.

# In[1]:


import pandas as pd
from ax import *


# ## 1. Define the search space

# The core `Experiment` class only has one required parameter, `search_space`. A SearchSpace is composed of a set of parameters to be tuned in the experiment, and optionally a set of parameter constraints that define restrictions across these parameters.
# 
# Here we range over two parameters, each of which can take on values between 0 and 10.

# In[2]:


range_param1 = RangeParameter(name="x1", lower=0.0, upper=10.0, parameter_type=ParameterType.FLOAT)
range_param2 = RangeParameter(name="x2", lower=0.0, upper=10.0, parameter_type=ParameterType.FLOAT)

search_space = SearchSpace(
    parameters=[range_param1, range_param2],
)


# Note that there are two other parameter classes, FixedParameter and ChoiceParameter. Although we won't use these in this example, you can create them as follows.

# In[3]:


choice_param = ChoiceParameter(name="choice", values=["foo", "bar"], parameter_type=ParameterType.STRING)
fixed_param = FixedParameter(name="fixed", value=[True], parameter_type=ParameterType.BOOL)


# Sum constraints enforce that the sum of a set of parameters is greater or less than some bound, and order constraints enforce that one parameter is smaller than the other. We won't use these either, but see two examples below.

# In[4]:


sum_constraint = SumConstraint(
    parameters=[range_param1, range_param2], 
    is_upper_bound=True, 
    bound=5.0,
)

order_constraint = OrderConstraint(
    lower_parameter = range_param1,
    upper_parameter = range_param2,
)


# ## 2. Define the experiment

# Once we have a search space, we can create an experiment.

# In[5]:


experiment = Experiment(
    name="experiment_building_blocks",
    search_space=search_space,
)


# We can also define control values for each parameter by adding a status quo arm to the experiment.

# In[6]:


experiment.status_quo = Arm(
    name="control", 
    parameters={"x1": 0.0, "x2": 0.0},
)


# ## 3. Generate arms

# We can now generate arms, i.e. assignments of parameters to values, that lie within the search space. Below we use a Sobol generator to generate five quasi-random arms. The `Models` registry provides a set of standard models Ax contains.

# In[7]:


sobol = Models.SOBOL(search_space=experiment.search_space)
generator_run = sobol.gen(5)

for arm in generator_run.arms:
    print(arm)


# To inspect available model settings, we can call `view_kwargs` or `view_defaults` on a specific model:

# In[8]:


Models.SOBOL.view_kwargs()  # Shows keyword argument names and typing.


# Any of the default arguments can be overriden by simply passing that keyword argument to the model constructor (e.g. `Models.SOBOL`).

# ## 4. Define an optimization config with custom metrics

# In order to perform an optimization, we also need to define an optimization config for the experiment. An optimization config is composed of an objective metric to be minimized or maximized in the experiment, and optionally a set of outcome constraints that place restrictions on how other metrics can be moved by the experiment. 
# 
# In order to define an objective or outcome constraint, we first need to subclass `Metric`. Metrics are used to evaluate trials, which are individual steps of the experiment sequence. Each trial contains one or more arms for which we will collect data at the same time.
# 
# Our custom metric(s) will determine how, given a trial, to compute the mean and SEM of each of the trial's arms.
# 
# The only method that needs to be defined for most metric subclasses is `fetch_trial_data`, which defines how a single trial is evaluated, and returns a pandas dataframe.

# In[9]:


class BoothMetric(Metric):
    def fetch_trial_data(self, trial):  
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2,
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))


# Once we have our metric subclasses, we can define an optimization config.

# In[10]:


optimization_config = OptimizationConfig(
    objective = Objective(
        metric=BoothMetric(name="booth"), 
        minimize=True,
    ),
)

experiment.optimization_config = optimization_config


# Outcome constraints can also be defined as follows. We won't use outcome constraints in this example, but they can be passed into the optimization config via the `outcome_constraints` argument.

# In[11]:


outcome_constraint = OutcomeConstraint(
    metric=Metric("constraint"), 
    op=ComparisonOp.LEQ, 
    bound=0.5,
)


# ## 5. Define a runner

# Before an experiment can collect data, it must have a `Runner` attached. A runner handles the deployment of trials. A trial must be "run" before it can be evaluated.
# 
# Here, we have a dummy runner that does nothing. In practice, a runner might be in charge of pushing an experiment to production.
# 
# The only method that needs to be defined for runner subclasses is `run`, which performs any necessary deployment logic, and returns a dictionary of resulting metadata.

# In[12]:


class MyRunner(Runner):
    def run(self, trial):
        return {"name": str(trial.index)}
    
experiment.runner = MyRunner()


# ## 6. Create a trial

# Now we can collect data for arms within our search space and begin the optimization. We do this by:
# 1. Generating arms for an initial exploratory batch (already done above, using Sobol)
# 2. Adding these arms to a trial
# 3. Running the trial
# 4. Evaluating the trial
# 5. Generating new arms based on the results, and repeating

# In[13]:


experiment.new_batch_trial(generator_run=generator_run)


# Note that the arms attached to the trial are the same as those in the generator run above, except for the status quo, which is automatically added to each trial.

# In[14]:


for arm in experiment.trials[0].arms:
    print(arm)


# If our trial should contain contain only one arm, we can use `experiment.new_trial` instead.

# In[15]:


experiment.new_trial().add_arm(Arm(name='single_arm', parameters={'x1': 1, 'x2': 1}))


# In[16]:


print(experiment.trials[1].arm)


# ## 7. Fetch data

# To fetch trial data, we need to run it and mark it completed. For most metrics in Ax, data is only available once the [status of the trial](https://ax.dev/api/core.html#ax.core.base_trial.TrialStatus) is `COMPLETED`, since in real-worlds scenarios, metrics can typically only be fetched after the trial finished running.
# 
# NOTE: Metrics classes may implement the [`is_available_while_running`](https://ax.dev/api/core.html#ax.core.metric.Metric.is_available_while_running) method. When this method returns `True`, data is available when trials are either `RUNNING` or `COMPLETED`. This can be used to obtain intermediate results from A/B test trials and other online experiments, or when metric values are available immediately, like in the case of synthetic problem metrics.

# In[17]:


experiment.trials[0].run().mark_completed()


# In[18]:


data = experiment.fetch_data()


# We can inspect the data that was fetched for each (arm, metric) pair.

# In[19]:


data.df


# ## 8. Iterate using GP+EI

# Now we can model the data collected for the initial set of arms via Bayesian optimization (using the Botorch model default of [Gaussian Process with Expected Improvement acquisition function](https://botorch.org/api/acquisition.html#botorch.acquisition.analytic.ExpectedImprovement)) to determine the new arms for which to fetch data next.

# In[20]:


gpei = Models.BOTORCH(experiment=experiment, data=data)
generator_run = gpei.gen(5)
experiment.new_batch_trial(generator_run=generator_run)


# In[21]:


for arm in experiment.trials[2].arms:
    print(arm)


# In[22]:


experiment.trials[2].run()
data = experiment.fetch_data()
data.df


# ## 9. Save to JSON or SQL

# At any point, we can also save our experiment to a JSON file. To ensure that our custom metrics and runner are saved properly, we first need to register them.

# In[23]:


from ax.storage.metric_registry import register_metric
from ax.storage.runner_registry import register_runner

register_metric(BoothMetric)
register_runner(MyRunner)

from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

save_experiment(experiment, "experiment.json")


# In[24]:


loaded_experiment = load_experiment("experiment.json")


# To save our experiment to SQL, we must first specify a connection to a database and create all necessary tables.

# In[25]:


from ax.storage.sqa_store.db import init_engine_and_session_factory,get_engine, create_all_tables
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url='sqlite:///foo2.db')

engine = get_engine()
create_all_tables(engine)


# In[26]:


save_experiment(experiment)


# In[27]:


load_experiment(experiment.name)


# ## 10. SimpleExperiment

# `SimpleExperiment` is a subclass of `Experiment` that assumes synchronous evaluation of trials and is therefore able to abstract away certain details and enable faster instantiation.
# 
# Rather than defining custom metrics and an optimization config, we define an evaluation function that determines the mean and SEM for a given parameterization.

# In[28]:


def evaluation_function(params):
    return (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2


# In[29]:


simple_experiment = SimpleExperiment(
    search_space=search_space,
    evaluation_function=evaluation_function,
)


# We add trials and evaluate them as before.

# In[30]:


simple_experiment.new_trial().add_arm(Arm(name='single_arm', parameters={'x1': 1, 'x2': 1}))


# In[31]:


data = simple_experiment.fetch_data()


# In[32]:


data.df

