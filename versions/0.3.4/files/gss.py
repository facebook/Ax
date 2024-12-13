#!/usr/bin/env python
# coding: utf-8

# This tutorial illustrates use of a Global Stopping Strategy (GSS) in combination with the Service API. For background on the Service API, see the Service API Tutorial: https://ax.dev/tutorials/gpei_hartmann_service.html GSS is also supported in the Scheduler API, where it can be provided as part of `SchedulerOptions`. For more on `Scheduler`, see the Scheduler tutorial: https://ax.dev/tutorials/scheduler.html
# 
# Global Stopping stops an optimization loop when some data-based criteria are met which suggest that future trials will not be very helpful. For example, we might stop when there has been very little improvement in the last five trials. This is as opposed to trial-level early stopping, which monitors the results of expensive evaluations and terminates those that are unlikely to produce promising results, freeing resources to explore more promising configurations. For more on trial-level early stopping, see the tutorial: https://ax.dev/tutorials/early_stopping/early_stopping.html

# In[1]:


import numpy as np

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import Branin, branin
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


# # 1. What happens without global stopping? Optimization can run for too long.
# This example uses the Branin test problem. We run 25 trials, which turns out to be far more than needed, because we get close to the optimum quite quickly.

# In[2]:


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"branin": (branin(x), 0.0)}


# In[3]:


params = [
    {
        "name": f"x{i + 1}",
        "type": "range",
        "bounds": [*Branin._domain[i]],
        "value_type": "float",
        "log_scale": False,
    }

    for i in range(2)
]


# In[4]:


ax_client = AxClient(random_seed=0, verbose_logging=False)

ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=params,
    objectives={"branin": ObjectiveProperties(minimize=True)},
    is_test=True,
)


# In[5]:


get_ipython().run_cell_magic('time', '', 'for i in range(25):\n    parameters, trial_index = ax_client.get_next_trial()\n    # Local evaluation here can be replaced with deployment to external system.\n    ax_client.complete_trial(\n        trial_index=trial_index, raw_data=evaluate(parameters)\n    )\n')


# In[6]:


render(ax_client.get_optimization_trace())


# # 2. Optimization with global stopping, with the Service API

# Rather than running a fixed number of trials, we can use a GlobalStoppingStrategy (GSS), which checks whether some stopping criteria have been met when `get_next_trial` is called. Here, we use an `ImprovementGlobalStoppingStrategy`, which checks whether the the last `window_size` trials have improved by more than some threshold amount.
# 
# For single-objective optimization, which we are doing here, `ImprovementGlobalStoppingStrategy` checks if an improvement is "significant" by comparing it to the inter-quartile range (IQR) of the objective values attained so far. 
# 
# `ImprovementGlobalStoppingStrategy` also supports multi-objective optimization (MOO), in which case it checks whether the percentage improvement in hypervolume over the last `window_size` trials exceeds `improvement_bar`.

# In[7]:


from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.exceptions.core import OptimizationShouldStop


# In[8]:


# Start considering stopping only after the 5 initialization trials + 5 GPEI trials.
# Stop if the improvement in the best point in the past 5 trials is less than
# 1% of the IQR thus far.
stopping_strategy = ImprovementGlobalStoppingStrategy(
    min_trials=5 + 5, window_size=5, improvement_bar=0.01
)


# In[9]:


ax_client_gss = AxClient(
    global_stopping_strategy=stopping_strategy, random_seed=0, verbose_logging=False
)

ax_client_gss.create_experiment(
    name="branin_test_experiment",
    parameters=params,
    objectives={"branin": ObjectiveProperties(minimize=True)},
    is_test=True,
)


# If there has not been much improvement, `ImprovementGlobalStoppingStrategy` will raise an exception. If the exception is raised, we catch it and terminate optimization.

# In[10]:


for i in range(25):
    try:
        parameters, trial_index = ax_client_gss.get_next_trial()
    except OptimizationShouldStop as exc:
        print(exc.message)
        break
    ax_client_gss.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


# In[11]:


render(ax_client_gss.get_optimization_trace())


# # 3. Write your own custom Global Stopping Strategy

# You can write a custom Global Stopping Strategy by subclassing `BaseGlobalStoppingStrategy` and use it where  `ImprovementGlobalStoppingStrategy` was used above.

# In[12]:


from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from typing import Tuple
from ax.core.experiment import Experiment
from ax.core.base_trial import TrialStatus
from ax.global_stopping.strategies.improvement import constraint_satisfaction


# Here, we define `SimpleThresholdGlobalStoppingStrategy`, which stops when we observe a point better than a provided threshold. This can be useful when there is a known optimum. For example, the Branin function has an optimum of zero. When the optimum is not known, this can still be useful from a satisficing perspective: For example, maybe we need a model to take up less than a certain amount of RAM so it doesn't crash our usual hardware, but there is no benefit to further improvements.

# In[13]:


class SimpleThresholdGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """
    A GSS that stops when we observe a point better than `threshold`.
    """
    def __init__(
        self,
        min_trials: int,
        inactive_when_pending_trials: bool = True,
        threshold: float = 0.1
    ):
        self.threshold = threshold
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials
        )
    
    def _should_stop_optimization(
        self, experiment: Experiment
    ) -> Tuple[bool, str]:
        """
        Check if the best seen is better than `self.threshold`.
        """
        feasible_objectives = [
            trial.objective_mean
            for trial in experiment.trials_by_status[TrialStatus.COMPLETED]
            if constraint_satisfaction(trial)
        ]

        # Computing the interquartile for scaling the difference
        if len(feasible_objectives) <= 1:
            message = "There are not enough feasible arms tried yet."
            return False, message
        
        minimize = experiment.optimization_config.objective.minimize
        if minimize:
            best = np.min(feasible_objectives)
            stop = best < self.threshold
        else:
            best = np.max(feasible_objectives)
            stop = best > self.threshold

        comparison = "less" if minimize else "greater"
        if stop:
            message = (
                f"The best objective seen is {best:.3f}, which is {comparison} "
                f"than the threshold of {self.threshold:.3f}."
            )
        else:
            message = ""

        return stop, message


# In[14]:


stopping_strategy = SimpleThresholdGlobalStoppingStrategy(min_trials=5, threshold=1.)


# In[15]:


ax_client_custom_gss = AxClient(
    global_stopping_strategy=stopping_strategy,
    random_seed=0,
    verbose_logging=False,
)

ax_client_custom_gss.create_experiment(
    name="branin_test_experiment",
    parameters=params,
    objectives={"branin": ObjectiveProperties(minimize=True)},
    is_test=True,
)


# In[16]:


for i in range(25):
    try:
        parameters, trial_index = ax_client_custom_gss.get_next_trial()
    except OptimizationShouldStop as exc:
        print(exc.message)
        break
    ax_client_custom_gss.complete_trial(
        trial_index=trial_index, raw_data=evaluate(parameters)
    )


# In[17]:


render(ax_client_custom_gss.get_optimization_trace())


# In[ ]:




