#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models, ModelRegistryBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features

from ax.utils.testing.core_stubs import get_branin_search_space, get_branin_experiment


# # Generation Strategy (GS) Tutorial
# 
# `GenerationStrategy` ([API reference](https://ax.dev/api/modelbridge.html#ax.modelbridge.generation_strategy.GenerationStrategy)) is a key abstraction in Ax:
# - It allows for specifying multiple optimization algorithms to chain one after another in the course of the optimization. 
# - Many higher-level APIs in Ax use generation strategies: Service and Loop APIs, `Scheduler` etc. (tutorials for all those higher-level APIs are here: https://ax.dev/tutorials/).
# - Generation strategy allows for storage and resumption of modeling setups, making optimization resumable from SQL or JSON snapshots.
# 
# This tutorial walks through a few examples of generation strategies and discusses its important settings. Before reading it, we recommend familiarizing yourself with how `Model` and `ModelBridge` work in Ax: https://ax.dev/docs/models.html#deeper-dive-organization-of-the-modeling-stack.
# 
# **Contents:**
# 1. Quick-start examples
#    1. Manually configured GS
#    2. Auto-selected GS
#    3. Candidate generation from a GS
# 2. Deep dive: `GenerationStep` a building block of the generation strategy
#    1. Describing a model
#    2. Other `GenerationStep` settings
#    3. Chaining `GenerationStep`-s together
#    4. `max_parallelism` enforcement and handling the `MaxParallelismReachedException`
# 3. `GenerationStrategy` storage
#    1. JSON storage
#    2. SQL storage
# 4. Advanced considerations / "gotchas"
#    1. Generation strategy produces `GeneratorRun`-s, not `Trial`-s
#    2. `model_kwargs` elements that don't have associated serialization logic in Ax
#    3. Why prefer `Models` registry enum entries over a factory function?
#    4. How to request more modeling setups in `Models`?
#    
# ----

# ## 1. Quick-start examples
# 
# ### 1A. Manually configured generation strategy
# 
# Below is a typical generation strategy used for most single-objective optimization cases in Ax:

# In[2]:


gs = GenerationStrategy(
    steps=[
        # 1. Initialization step (does not require pre-existing data and is well-suited for 
        # initial sampling of the search space)
        GenerationStep(
            model=Models.SOBOL,
            num_trials=5,  # How many trials should be produced from this generation step
            min_trials_observed=3, # How many trials need to be completed to move to next model
            max_parallelism=5,  # Max parallelism for this step
            model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
        ),
        # 2. Bayesian optimization step (requires data obtained from previous phase and learns
        # from all data available at the time of each new candidate generation call)
        GenerationStep(
            model=Models.GPEI,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        ),
    ]
)


# ### 1B. Auto-selected generation strategy
# 
# Ax provides a [`choose_generation_strategy`](https://github.com/facebook/Ax/blob/main/ax/modelbridge/dispatch_utils.py#L115) utility, which can auto-select a suitable generation strategy given a search space and an array of other optional settings. The utility is fairly simple at the moment, but additional development (support for multi-objective optimization, multi-fidelity optimization, Bayesian optimization with categorical kernels etc.) is coming soon.

# In[3]:


gs = choose_generation_strategy(
    # Required arguments:
    search_space=get_branin_search_space(),  # Ax `SearchSpace`
    
    # Some optional arguments (shown with their defaults), see API docs for more settings:
    # https://ax.dev/api/modelbridge.html#module-ax.modelbridge.dispatch_utils
    use_batch_trials=False,  # Whether this GS will be used to generate 1-arm `Trial`-s or `BatchTrials`
    no_bayesian_optimization=False,  # Use quasi-random candidate generation without BayesOpt
    max_parallelism_override=None,  # Integer, to which to set the `max_parallelism` setting of all steps in this GS   
)
gs


# ### 1C. Candidate generation from a generation strategy
# 
# While often used through Service or Loop API or other higher-order abstractions like the Ax `Scheduler` (where the generation strategy is used to fit models and produce candidates from them under-the-hood), it's also possible to use the GS directly, in place of a `ModelBridge` instance. The interface of `GenerationStrategy.gen` is the same as `ModelBridge.gen`.
# 

# In[4]:


experiment = get_branin_experiment()


# Note that it's important to **specify pending observations** to the call to `gen` to avoid getting the same points re-suggested. Without `pending_observations` argument, Ax models are not aware of points that should be excluded from generation. Points are considered "pending" when they belong to `STAGED`, `RUNNING`, or `ABANDONED` trials (with the latter included so model does not re-suggest points that are considered "bad" and should not be re-suggested).
# 
# If the call to `get_pending_obervation_features` becomes slow in your setup (since it performs data-fetching etc.), you can opt for `get_pending_observation_features_based_on_trial_status` (also from `ax.modelbridge.modelbridge_utils`), but note the limitations of that utility (detailed in its docstring).

# In[5]:


generator_run = gs.gen(
    experiment=experiment, # Ax `Experiment`, for which to generate new candidates
    data=None, # Ax `Data` to use for model training, optional.
    n=1, # Number of candidate arms to produce
    pending_observations=get_pending_observation_features(experiment),  # Points that should not be re-generated
    # Any other kwargs specified will be passed through to `ModelBridge.gen` along with `GenerationStep.model_gen_kwargs`
)
generator_run


# Then we can add the newly produced [`GeneratorRun`](https://ax.dev/docs/glossary.html#generator-run) to the experiment as a [`Trial` (or `BatchTrial` if `n` > 1)](https://ax.dev/docs/glossary.html#trial):

# In[6]:


trial = experiment.new_trial(generator_run)
trial


# **Important notes on `GenerationStrategy.gen`:**
# - if `data` argument above is not specified, GS will pull experiment data from cache via `experiment.lookup_data`,
# - without specifying `pending_observations`, the GS (and any model in Ax) could produce the same candidate over and over, as without that argument the model is not 'aware' that the candidate is part of a `RUNNING` or `ABANDONED` trial and should not be re-suggested again.
# 
# In cases where `get_pending_observation_features` is too slow and the experiment consists of 1-arm `Trial`-s only, it's possible to use `get_pending_observation_features_based_on_trial_status` instead (found in the same file).

# Note that when using the Ax Service API, one of the arguments to `AxClient` is `choose_generation_strategy_kwargs`; specifying that argument is a convenient way to influence the choice of generation strategy in `AxClient` without manually specifying a full `GenerationStrategy`.

# -----

# ## 2. `GenerationStep` as a building block of generation strategy

# ### 2A. Describing a model to use in a given `GenerationStep`
# 
# There are two ways of specifying a model for a generation step: via an entry in a `Models` enum or via a 'factory function' –– a callable model constructor (e.g. [`get_GPEI`](https://github.com/facebook/Ax/blob/0e454b71d5e07b183c0866855555b6a21ddd5da1/ax/modelbridge/factory.py#L154) and other factory functions in the same file). Note that using the latter path, a factory function, will prohibit `GenerationStrategy` storage and is generally discouraged. 

# ### 2B. Other `GenerationStep` settings
# 
# All of the available settings are described in the documentation:

# In[7]:


print(GenerationStep.__doc__)


# ## 2C. Chaining `GenerationStep`-s together
# 
# A `GenerationStrategy` moves from one step to another when: 
# 1. `N=num_trials` generator runs were produced and attached as trials to the experiment AND 
# 2. `M=min_trials_observed` have been completed and have data.
# 
# **Caveat: `enforce_num_trials` setting**:
# 
# 1. If `enforce_num_trials=True` for a given generation step, if 1) is reached but 2) is not yet reached, the generation strategy will raise a `DataRequiredError`, indicating that more trials need to be completed before the next step.
# 2. If `enforce_num_trials=False`, the GS will continue producing generator runs from the current step until 2) is reached.

# ## 2D. `max_parallelism` enforcement
# 
# Generation strategy can restrict the number of trials that can be ran simultaneously (to encourage sequential optimization, which benefits Bayesian optimization performance). When the parallelism limit is reached, a call to `GenerationStrategy.gen` will result in a `MaxParallelismReachedException`.
# 
# The correct way to handle this exception:
# 1. Make sure that `GenerationStep.max_parallelism` is configured correctly for all steps in your generation strategy (to disable it completely, configure `GenerationStep.max_parallelism=None`),
# 2. When encountering the exception, wait to produce more generator runs until more trial evluations complete and log the trial completion via `trial.mark_completed`.

# ----
# 
# ## 3. SQL and JSON storage of a generation strategy

# When used through Service API or `Scheduler`, generation strategy will be automatically stored to SQL or JSON via specifying `DBSettings` to either `AxClient` or `Scheduler` (details in respective tutorials in the ["Tutorials" page](https://ax.dev/tutorials/)). Generation strategy can also be stored to SQL or JSON individually, as shown below.
# 
# More detail on SQL and JSON storage in Ax generally can be [found in "Building Blocks of Ax" tutorial](https://ax.dev/tutorials/building_blocks.html#9.-Save-to-JSON-or-SQL).

# ### 3A. SQL storage
# For SQL storage setup in Ax, read through the ["Storage" documentation page](https://ax.dev/docs/storage.html).
# 
# Note that unlike an Ax experiment, a generation strategy does not have a name or another unique identifier. Therefore, a generation strategy is stored in association with experiment and can be retrieved by the associated experiment's name.

# In[8]:


from ax.storage.sqa_store.save import save_generation_strategy, save_experiment
from ax.storage.sqa_store.load import load_experiment, load_generation_strategy_by_experiment_name

from ax.storage.sqa_store.db import init_engine_and_session_factory,get_engine, create_all_tables
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment

init_engine_and_session_factory(url='sqlite:///foo2.db')

engine = get_engine()
create_all_tables(engine)

save_experiment(experiment)
save_generation_strategy(gs)

experiment = load_experiment(experiment_name=experiment.name)
gs = load_generation_strategy_by_experiment_name(
    experiment_name=experiment.name, 
    experiment=experiment,  # Can optionally specify experiment object to avoid loading it from database twice
)
gs


# ### 3B. JSON storage

# In[9]:


from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json

gs_json = object_to_json(gs)  # Can be written to a file or string via `json.dump` etc.
gs = object_from_json(gs_json)  # Decoded back from JSON (can be loaded from file, string via `json.load` etc.)
gs


# ------

# ## 3. Advanced considerations
# 
# Below is a list of important "gotchas" of using generation strategy (especially outside of the higher-level APIs like the Service API or the `Scheduler`):
# 
# ### 3A. `GenerationStrategy.gen` produces `GeneratorRun`-s, not trials
# 
# Since `GenerationStrategy.gen` mimics `ModelBridge.gen` and allows for human-in-the-loop usage mode, a call to `gen` produces a `GeneratorRun`, which can then be added (or altered before addition or not added at all) to a `Trial` or `BatchTrial` on a given experiment. So it's important to add the generator run to a trial, since otherwise it will not be attached to the experiment on its own.

# In[10]:


generator_run = gs.gen(
    experiment=experiment, n=1, pending_observations=get_pending_observation_features(experiment)
)
experiment.new_trial(generator_run)


# ### 3B. `model_kwargs` elements that do not define serialization logic in Ax

# Note that passing objects that are not yet serializable in Ax (e.g. a BoTorch `Prior` object) as part of `GenerationStep.model_kwargs` or `GenerationStep.model_gen_kwargs` will prevent correct generation strategy storage. If this becomes a problem, feel free to open an issue on our Github: https://github.com/facebook/Ax/issues to get help with adding storage support for a given object.

# ### 3C. Why prefer `Models` enum entries over a factory function?

# 1. **Storage potential:** a call to, for example, `Models.GPEI` captures all arguments to the model and model bridge and stores them on a generator runs, subsequently produced by the model. Since the capturing logic is part of `Models.__call__` function, it is not present in a factory function. Furthermore, there is no safe and flexible way to serialize callables in Python.
# 2. **Standardization:** While a 'factory function' is by default more flexible (accepts any specified inputs and produces a `ModelBridge` with an underlying `Model` instance based on them), it is not standard in terms of its inputs. `Models` introduces a standardized interface, making it easy to adapt any example to one's specific case.

# ### 3D. How can I request more modeling setups added to `Models` and natively supported in Ax?

# Please open a [Github issue](https://github.com/facebook/Ax/issues) to request a new modeling setup in Ax (or for any other questions or requests).
