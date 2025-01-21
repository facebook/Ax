#!/usr/bin/env python
# coding: utf-8

# # Using external methods for candidate generation in Ax
# 
# Out of the box, Ax offers many options for candidate generation, most of which utilize Bayesian optimization algorithms built using [BoTorch](https://botorch.org/). For users that want to leverage Ax for experiment orchestration (via `AxClient` or `Scheduler`) and other features (e.g., early stopping), while relying on other methods for candidate generation, we introduced `ExternalGenerationNode`. 
# 
# A `GenerationNode` is a building block of a `GenerationStrategy`. They can be combined together utilize different methods for generating candidates at different stages of an experiment. `ExternalGenerationNode` exposes a lightweight interface to allow the users to easily integrate their methods into Ax, and use them as standalone or with other `GenerationNode`s in a `GenerationStrategy`.
# 
# In this tutorial, we will implement a simple generation node using `RandomForestRegressor` from sklearn, and combine it with Sobol (for initialization) to optimize the Hartmann6 problem.
# 
# NOTE: This is for illustration purposes only. We do not recommend using this strategy as it typically does not perform well compared to Ax's default algorithms due to it's overly greedy behavior.

# In[1]:


import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import RangeParameter
from ax.core.types import TParameterization
from ax.modelbridge.external_generation_node import ExternalGenerationNode
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import MaxTrials
from ax.plot.trace import plot_objective_value_vs_trial_index
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import hartmann6
from sklearn.ensemble import RandomForestRegressor


class RandomForestGenerationNode(ExternalGenerationNode):
    """A generation node that uses the RandomForestRegressor
    from sklearn to predict candidate performance and picks the
    next point as the random sample that has the best prediction.

    To leverage external methods for candidate generation, the user must
    create a subclass that implements ``update_generator_state`` and
    ``get_next_candidate`` methods. This can then be provided
    as a node into a ``GenerationStrategy``, either as standalone or as
    part of a larger generation strategy with other generation nodes,
    e.g., with a Sobol node for initialization.
    """

    def __init__(self, num_samples: int, regressor_options: Dict[str, Any]) -> None:
        """Initialize the generation node.

        Args:
            regressor_options: Options to pass to the random forest regressor.
            num_samples: Number of random samples from the search space
                used during candidate generation. The sample with the best
                prediction is recommended as the next candidate.
        """
        t_init_start = time.monotonic()
        super().__init__(node_name="RandomForest")
        self.num_samples: int = num_samples
        self.regressor: RandomForestRegressor = RandomForestRegressor(
            **regressor_options
        )
        # We will set these later when updating the state.
        # Alternatively, we could have required experiment as an input
        # and extracted them here.
        self.parameters: Optional[List[RangeParameter]] = None
        self.minimize: Optional[bool] = None
        # Recording time spent in initializing the generator. This is
        # used to compute the time spent in candidate generation.
        self.fit_time_since_gen: float = time.monotonic() - t_init_start

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """A method used to update the state of the generator. This includes any
        models, predictors or any other custom state used by the generation node.
        This method will be called with the up-to-date experiment and data before
        ``get_next_candidate`` is called to generate the next trial(s). Note
        that ``get_next_candidate`` may be called multiple times (to generate
        multiple candidates) after a call to  ``update_generator_state``.

        For this example, we will train the regressor using the latest data from
        the experiment.

        Args:
            experiment: The ``Experiment`` object representing the current state of the
                experiment. The key properties includes ``trials``, ``search_space``,
                and ``optimization_config``. The data is provided as a separate arg.
            data: The data / metrics collected on the experiment so far.
        """
        search_space = experiment.search_space
        parameter_names = list(search_space.parameters.keys())
        metric_names = list(experiment.optimization_config.metrics.keys())
        if any(
            not isinstance(p, RangeParameter) for p in search_space.parameters.values()
        ):
            raise NotImplementedError(
                "This example only supports RangeParameters in the search space."
            )
        if search_space.parameter_constraints:
            raise NotImplementedError(
                "This example does not support parameter constraints."
            )
        if len(metric_names) != 1:
            raise NotImplementedError(
                "This example only supports single-objective optimization."
            )
        # Get the data for the completed trials.
        num_completed_trials = len(experiment.trials_by_status[TrialStatus.COMPLETED])
        x = np.zeros([num_completed_trials, len(parameter_names)])
        y = np.zeros([num_completed_trials, 1])
        for t_idx, trial in experiment.trials.items():
            if trial.status == "COMPLETED":
                trial_parameters = trial.arm.parameters
                x[t_idx, :] = np.array([trial_parameters[p] for p in parameter_names])
                trial_df = data.df[data.df["trial_index"] == t_idx]
                y[t_idx, 0] = trial_df[trial_df["metric_name"] == metric_names[0]][
                    "mean"
                ].item()

        # Train the regressor.
        self.regressor.fit(x, y)
        # Update the attributes not set in __init__.
        self.parameters = search_space.parameters
        self.minimize = experiment.optimization_config.objective.minimize

    def get_next_candidate(
        self, pending_parameters: List[TParameterization]
    ) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        We will draw ``self.num_samples`` random samples from the search space
        and predict the objective value for each sample. We will then return
        the sample with the best predicted value.

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation. This is often used to avoid generating duplicate candidates.
                We ignore this here for simplicity.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """
        bounds = np.array([[p.lower, p.upper] for p in self.parameters.values()])
        unit_samples = np.random.random_sample([self.num_samples, len(bounds)])
        samples = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * unit_samples
        # Predict the objective value for each sample.
        y_pred = self.regressor.predict(samples)
        # Find the best sample.
        best_idx = np.argmin(y_pred) if self.minimize else np.argmax(y_pred)
        best_sample = samples[best_idx, :]
        # Convert the sample to a parameterization.
        candidate = {
            p_name: best_sample[i].item()
            for i, p_name in enumerate(self.parameters.keys())
        }
        return candidate


# ## Construct the GenerationStrategy
# 
# We will use Sobol for the first 5 trials and defer to random forest for the rest.

# In[2]:


generation_strategy = GenerationStrategy(
    name="Sobol+RandomForest",
    nodes=[
        GenerationNode(
            node_name="Sobol",
            model_specs=[ModelSpec(Models.SOBOL)],
            transition_criteria=[
                MaxTrials(
                    # This specifies the maximum number of trials to generate from this node, 
                    # and the next node in the strategy.
                    threshold=5,
                    block_transition_if_unmet=True,
                    transition_to="RandomForest"
                )
            ],
        ),
        RandomForestGenerationNode(num_samples=128, regressor_options={}),
    ],
)


# ## Run a simple experiment using AxClient
# 
# More details on how to use AxClient can be found in the [tutorial](https://ax.dev/tutorials/gpei_hartmann_service.html).

# In[3]:


ax_client = AxClient(generation_strategy=generation_strategy)

ax_client.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": f"x{i}",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
        }
        for i in range(1, 7)
    ],
    objectives={"hartmann6": ObjectiveProperties(minimize=True)},
)


def evaluate(parameterization: TParameterization) -> Dict[str, Tuple[float, float]]:
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    return {"hartmann6": (checked_cast(float, hartmann6(x)), 0.0)}


# ### Run the optimization loop

# In[4]:


for i in range(15):
    parameterization, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=evaluate(parameterization)
    )


# ### View the trials generated during optimization

# In[5]:


exp_df = exp_to_df(ax_client.experiment)
exp_df


# In[6]:


plot_objective_value_vs_trial_index(
    exp_df=exp_df,
    metric_colname="hartmann6",
    minimize=True,
    title="Hartmann6 Objective Value vs. Trial Index",
)

