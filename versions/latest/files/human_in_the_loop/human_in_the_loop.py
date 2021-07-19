#!/usr/bin/env python
# coding: utf-8

# # Using Ax for Human-in-the-loop Experimentation¶

# While Ax can be used in as a fully automated service, generating and deploying candidates Ax can be also used in a trial-by-trial fashion, allowing for human oversight. 
# 
# Typically, human intervention in Ax is necessary when there are clear tradeoffs between multiple metrics of interest. Condensing multiple outcomes of interest into a single scalar quantity can be really challenging. Instead, it can be useful to specify an objective and constraints, and tweak these based on the information from the experiment. 
# 
# To facilitate this, Ax provides the following key features:
# 
# 1. Constrained optimization
# 2. Interfaces for easily modifying optimization goals
# 3. Utilities for visualizing and deploying new trials composed of multiple optimizations. 
# 
# 
# In this tutorial, we'll demonstrate how Ax enables users to explore these tradeoffs. With an understanding of the tradeoffs present in our data, we'll then make use of the constrained optimization utilities to generate candidates from multiple different optimization objectives, and create a conglomerate batch, with all of these candidates in together in one trial. 

# ## Experiment Setup
# 
# For this tutorial, we will assume our experiment has already been created.

# In[1]:


from ax import Data, Metric, OptimizationConfig, Objective, OutcomeConstraint, ComparisonOp, json_load
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.factory import get_GPEI
from ax.plot.diagnostic import tile_cross_validation
from ax.plot.scatter import plot_multiple_metrics, tile_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting

import pandas as pd

init_notebook_plotting()


# In[2]:


experiment = json_load.load_experiment('hitl_exp.json')


# ### Initial Sobol Trial

# Bayesian Optimization experiments almost always begin with a set of random points. In this experiment, these points were chosen via a Sobol sequence, accessible via the `ModelBridge` factory.
# 
# A collection of points run and analyzed together form a `BatchTrial`. A `Trial` object provides metadata pertaining to the deployment of these points, including details such as when they were deployed, and the current status of their experiment. 
# 
# Here, we see an initial experiment has finished running (COMPLETED status).

# In[3]:


experiment.trials[0]


# In[4]:


experiment.trials[0].time_created


# In[5]:


# Number of arms in first experiment, including status_quo
len(experiment.trials[0].arms)


# In[6]:


# Sample arm configuration
experiment.trials[0].arms[0]


# ## Experiment Analysis
# 
# **Optimization Config**
# 
# An important construct for analyzing an experiment is an OptimizationConfig. An OptimizationConfig contains an objective, and outcome constraints. Experiment's can have a default OptimizationConfig, but models can also take an OptimizationConfig as input independent of the default.
# 
# **Objective:** A metric to optimize, along with a direction to optimize (default: maximize)
# 
# **Outcome Constraint:** A metric to constrain, along with a constraint direction (<= or >=), as well as a bound. 
# 
# Let's start with a simple OptimizationConfig. By default, our objective metric will be maximized, but can be minimized by setting the `minimize` flag. Our outcome constraint will, by default, be evaluated as a relative percentage change. This percentage change is computed relative to the experiment's status quo arm. 

# In[7]:


experiment.status_quo


# In[8]:


objective_metric = Metric(name="metric_1")
constraint_metric = Metric(name="metric_2")

experiment.optimization_config = OptimizationConfig(
    objective=Objective(objective_metric),
    outcome_constraints=[
        OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=5),
    ]
)


# **Data**
# 
# Another critical piece of analysis is data itself! Ax data follows a standard format, shown below. This format is imposed upon the underlying data structure, which is a Pandas DataFrame. 
# 
# A key set of fields are required for all data, for use with Ax models. 
# 
# It's a good idea to double check our data before fitting models -- let's make sure all of our expected metrics and arms are present.

# In[9]:


data = Data(pd.read_json('hitl_data.json'))
data.df.head()


# In[10]:


data.df['arm_name'].unique()


# In[11]:


data.df['metric_name'].unique()


# **Search Space** 
# 
# The final component necessary for human-in-the-loop optimization is a SearchSpace. A SearchSpace defines the feasible region for our parameters, as well as their types.
# 
# Here, we have both parameters and a set of constraints on those parameters. 
# 
# Without a SearchSpace, our models are unable to generate new candidates. By default, the models will read the search space off of the experiment, when they are told to generate candidates. SearchSpaces can also be specified by the user at this time. Sometimes, the first round of an experiment is too restrictive--perhaps the experimenter was too cautious when defining their initial ranges for exploration! In this case, it can be useful to generate candidates from new, expanded search spaces, beyond that specified in the experiment.  

# In[12]:


experiment.search_space.parameters


# In[13]:


experiment.search_space.parameter_constraints


# ### Model Fit
# 
# Fitting BoTorch's GPEI will allow us to predict new candidates based on our first Sobol batch. 
# Here, we make use of the default settings for GP-EI defined in the ModelBridge factory. 

# In[14]:


gp = get_GPEI(
    experiment=experiment,
    data=data,
)


# We can validate the model fits using cross validation, shown below for each metric of interest. Here, our model fits leave something to be desired--the tail ends of each metric are hard to model. In this situation, there are three potential actions to take: 
# 
# 1. Increase the amount of traffic in this experiment, to reduce the measurement noise.
# 2. Increase the number of points run in the random batch, to assist the GP in covering the space.
# 3. Reduce the number of parameters tuned at one time. 
# 
# However, away from the tail effects, the fits do show a strong correlations, so we will proceed with candidate generation. 

# In[15]:


cv_result = cross_validate(gp)
render(tile_cross_validation(cv_result))


# The parameters from the initial batch have a wide range of effects on the metrics of interest, as shown from the outcomes from our fitted GP model. 

# In[16]:


render(tile_fitted(gp, rel=True))


# In[17]:


METRIC_X_AXIS = 'metric_1'
METRIC_Y_AXIS = 'metric_2'

render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
))


# ### Candidate Generation
# 
# With our fitted GPEI model, we can optimize EI (Expected Improvement) based on any optimization config.
# We can start with our initial optimization config, and aim to simply maximize the playback smoothness, without worrying about the constraint on quality. 

# In[18]:


unconstrained = gp.gen(
    n=3,
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
    )
)


# Let's plot the tradeoffs again, but with our new arms. 

# In[19]:


render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'unconstrained': unconstrained,
    }
))


# ### Change Objectives

# With our unconstrained optimization, we generate some candidates which are pretty promising with respect to our objective! However, there is a clear regression in our constraint metric, above our initial 5% desired constraint. Let's add that constraint back in.  

# In[20]:


constraint_5 = OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=5)
constraint_5_results = gp.gen(
    n=3, 
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
        outcome_constraints=[constraint_5]
    )
)


# This yields a *GeneratorRun*, which contains points according to our specified optimization config, along with metadata about how the points were generated. Let's plot the tradeoffs in these new points. 

# In[21]:


from ax.plot.scatter import plot_multiple_metrics
render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'constraint_5': constraint_5_results
    }
))


# It is important to note that the treatment of constraints in GP EI is probabilistic. The acquisition function weights our objective by the probability that each constraint is feasible. Thus, we may allow points with a very small probability of violating the constraint to be generated, as long as the chance of the points increasing our objective is high enough. 
# 
# You can see above that the point estimate for each point is significantly below a 5% increase in the constraint metric, but that there is uncertainty in our prediction, and the tail probabilities do include probabilities of small regressions beyond 5%. 

# In[22]:


constraint_1 = OutcomeConstraint(metric=constraint_metric, op=ComparisonOp.LEQ, bound=1)
constraint_1_results = gp.gen(
    n=3, 
    optimization_config=OptimizationConfig(
        objective=Objective(objective_metric),
        outcome_constraints=[constraint_1],
    )
)


# In[23]:


render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        "constraint_1": constraint_1_results,
    }
))


# Finally, let's view all three sets of candidates together. 

# In[24]:


render(plot_multiple_metrics(
    gp,
    metric_x=METRIC_X_AXIS,
    metric_y=METRIC_Y_AXIS,
    generator_runs_dict={
        'unconstrained': unconstrained,
        'loose_constraint': constraint_5_results,
        'tight_constraint': constraint_1_results,
    }
))


# ## Creating a New Trial
# 
# Having done the analysis and candidate generation for three different optimization configs, we can easily create a new `BatchTrial` which combines the candidates from these three different optimizations. Each set of candidates looks promising -- the point estimates are higher along both metric values than in the previous batch. However, there is still a good bit of uncertainty in our predictions. It is hard to choose between the different constraint settings without reducing this noise, so we choose to run a new trial with all three constraint settings. However, we're generally convinced that the tight constraint is too conservative. We'd still like to reduce our uncertainty in that region, but we'll only take one arm from that set.

# In[25]:


# We can add entire generator runs, when constructing a new trial. 
trial = experiment.new_batch_trial().add_generator_run(unconstrained).add_generator_run(constraint_5_results)

# Or, we can hand-pick arms. 
trial.add_arm(constraint_1_results.arms[0])


# The arms are combined into a single trial, along with the `status_quo` arm. Their generator can be accessed from the trial as well. 

# In[26]:


experiment.trials[1].arms


# The original `GeneratorRuns` can be accessed from within the trial as well. This is useful for later analyses, allowing introspection of the `OptimizationConfig` used for generation (as well as other information, e.g. `SearchSpace` used for generation).

# In[27]:


experiment.trials[1]._generator_run_structs


# Here, we can see the unconstrained set-up used for our first set of candidates.  

# In[28]:


experiment.trials[1]._generator_run_structs[0].generator_run.optimization_config

