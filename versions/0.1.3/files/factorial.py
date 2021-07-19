
import numpy as np
import pandas as pd
import sklearn as skl
from typing import Dict, Optional, Tuple, Union
from ax import Arm, ChoiceParameter, Models, ParameterType, SearchSpace, SimpleExperiment
from ax.plot.scatter import plot_fitted
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.stats.statstools import agresti_coull_sem

init_notebook_plotting()

search_space = SearchSpace(
    parameters=[
        ChoiceParameter(
            name="factor1",
            parameter_type=ParameterType.STRING,
            values=["level11", "level12", "level13"],
        ),
        ChoiceParameter(
            name="factor2",
            parameter_type=ParameterType.STRING,
            values=["level21", "level22"],
        ),
        ChoiceParameter(
            name="factor3",
            parameter_type=ParameterType.STRING,
            values=["level31", "level32", "level33", "level34"],
        ),
    ]
)

one_hot_encoder = skl.preprocessing.OneHotEncoder(
    categories=[par.values for par in search_space.parameters.values()], 
)

def factorial_evaluation_function(
    # `parameterization` is a dict of parameter names to values of those parameters.
    parameterization: Dict[str, Optional[Union[str, bool, float]]],
    # `weight` is the weight of the parameterization, 
    # which is used to determine the variance of the estimate.
    weight: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:  # Mapping of metric names to tuple of mean and standard error.
    batch_size = 10000
    noise_level = 0.0
    weight = weight if weight is not None else 1.0
    coefficients = np.array([
        0.1, 0.2, 0.3,
        0.1, 0.2,
        0.1, 0.2, 0.3, 0.4
    ])
    features = np.array(list(parameterization.values())).reshape(1, -1)
    encoded_features = one_hot_encoder.fit_transform(features)
    z = coefficients @ encoded_features.T + np.sqrt(noise_level) * np.random.randn()
    p = np.exp(z) / (1 + np.exp(z))
    plays = np.random.binomial(batch_size, weight)
    successes = np.random.binomial(plays, p)
    mean = float(successes) / plays
    sem = agresti_coull_sem(successes, plays)
    return {
        "success_metric": (mean, sem)
    }

exp = SimpleExperiment(
    name="my_factorial_closed_loop_experiment",
    search_space=search_space,
    evaluation_function=factorial_evaluation_function,
    objective_name="success_metric",
)
exp.status_quo = Arm(
    parameters={"factor1": "level11", "factor2": "level21", "factor3": "level31"}
)

factorial = Models.FACTORIAL(search_space=exp.search_space)
factorial_run = factorial.gen(n=-1)  # Number of arms to generate is derived from the search space.
print(len(factorial_run.arms))

trial = (
    exp.new_batch_trial()
    .add_generator_run(factorial_run, multiplier=1)
)

trial.reweight_status_quo(4)
print(trial.arm_weights[trial.status_quo])

models = []
for i in range(4):
    print("Running iteration {}...".format(i+1))
    data = exp.eval_trial(trial)
    thompson = Models.THOMPSON(
        experiment=exp, data=data, min_weight=0.01
    )
    models.append(thompson)
    thompson_run = thompson.gen(n=-1)
    trial = exp.new_batch_trial().add_generator_run(thompson_run)

render(plot_fitted(models[0], metric="success_metric", rel=False))

render(plot_fitted(models[-1], metric="success_metric", rel=False))

results = pd.DataFrame(
    [
        {"values": ",".join(arm.parameters.values()), "weight": weight}
        for arm, weight in trial.normalized_arm_weights().items()
    ]
)
print(results)

from ax.plot.bandit_rollout import plot_bandit_rollout
from ax.utils.notebook.plotting import render

render(plot_bandit_rollout(exp))

from ax.plot.marginal_effects import plot_marginal_effects
render(plot_marginal_effects(models[0], 'success_metric'))
