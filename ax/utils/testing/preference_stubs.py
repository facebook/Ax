# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import numpy as np
import torch
from ax.core import Arm, GeneratorRun, OptimizationConfig
from ax.core.experiment import Experiment
from ax.core.types import TEvaluationOutcome, TParameterization
from ax.service.utils.instantiation import InstantiationBase
from ax.utils.common.constants import Keys
from botorch.utils.sampling import draw_sobol_samples
from pyre_extensions import assert_is_instance, none_throws


def sum_utility(parameters: TParameterization) -> float:
    """Test utility function that sums over parameter values"""
    values = [assert_is_instance(v, float) for v in parameters.values()]
    return sum(values)


def pairwise_pref_metric_eval(
    parameters: dict[str, TParameterization],
) -> dict[str, TEvaluationOutcome]:
    """evaluating pairwise comparisons using utility_func"""
    assert len(parameters.keys()) == 2
    arm1, arm2 = list(parameters.keys())
    arm1_sum, arm2_sum = sum_utility(parameters[arm1]), sum_utility(parameters[arm2])
    is_arm1_preferred = int(arm1_sum - arm2_sum > 0)
    return {
        arm1: {Keys.PAIRWISE_PREFERENCE_QUERY.value: is_arm1_preferred},
        arm2: {Keys.PAIRWISE_PREFERENCE_QUERY.value: 1 - is_arm1_preferred},
    }


def _metric_name_to_value(metric_name: str, metric_names: list[str]) -> float:
    i = sorted(metric_names).index(metric_name)
    return (i + 1) * 1000.0 + np.random.standard_normal()


def experimental_metric_eval(
    parameters: dict[str, Any], metric_names: list[str]
) -> dict[str, TEvaluationOutcome]:
    """evaluating experimental metrics

    Args:
        parameters: Dict of arm name to parameterization
        metric_names: List of metric names

    Returns:
        Dict of arm name to metric name to (mean, sem)
    """
    result_dict = {}
    for arm_name in parameters.keys():
        result_dict[arm_name] = {
            metric_name: (
                _metric_name_to_value(metric_name, metric_names),
                10.0,
            )
            for metric_name in metric_names
        }
    return result_dict


def get_pbo_experiment(
    num_parameters: int = 2,
    num_experimental_metrics: int = 3,
    parameter_names: list[str] | None = None,
    tracking_metric_names: list[str] | None = None,
    num_experimental_trials: int = 3,
    num_preference_trials: int = 3,
    num_preference_trials_w_repeated_arm: int = 5,
    include_sq: bool = True,
    partial_data: bool = False,
    unbounded_search_space: bool = False,
    experiment_name: str = "pref_experiment",
    optimization_config: OptimizationConfig | None = None,
) -> Experiment:
    """Create synthetic preferential BO experiment"""
    if tracking_metric_names is None:
        tracking_metric_names = [
            f"metric{i}" for i in range(1, num_experimental_metrics + 1)
        ]
    else:
        assert len(tracking_metric_names) == num_experimental_metrics

    if parameter_names is None:
        parameter_names = [f"x{i}" for i in range(1, num_parameters + 1)]
    else:
        assert len(parameter_names) == num_parameters

    param_bounds = [10.0, 30.0] if not unbounded_search_space else [-1e9, 1e9]
    if include_sq:
        sq = {param_name: np.mean(param_bounds) for param_name in parameter_names}
        sq_arm = Arm(parameters=sq, name="status_quo")
    else:
        sq = None
        sq_arm = None

    parameters = [
        {
            "name": param_name,
            "type": "range",
            # make the default search space non-unit for better clarity in testing
            "bounds": param_bounds,  # Changed from list to tuple
        }
        for param_name in parameter_names
    ]

    has_preference_query = (
        num_preference_trials > 0 or num_preference_trials_w_repeated_arm > 0
    )

    if optimization_config:
        opt_config = optimization_config
    elif has_preference_query:
        opt_config = InstantiationBase.make_optimization_config(
            objectives={Keys.PAIRWISE_PREFERENCE_QUERY.value: "maximize"},
            objective_thresholds=[],
            outcome_constraints=[],
            status_quo_defined=True,
        )
    elif len(tracking_metric_names) > 0:
        opt_config = InstantiationBase.make_optimization_config(
            objectives={tracking_metric_names[0]: "maximize"},
            objective_thresholds=[],
            outcome_constraints=[],
            status_quo_defined=True,
        )
    else:
        opt_config = None

    tracking_metrics = [
        InstantiationBase._make_metric(name=metric_name)
        for metric_name in tracking_metric_names
    ]

    experiment = Experiment(
        name=experiment_name,
        description="This is a test exp",
        experiment_type="NOTEBOOK",
        search_space=InstantiationBase.make_search_space(
            # pyre-fixme[6]: Incompatible parameter type
            parameters=parameters,
            parameter_constraints=None,
        ),
        optimization_config=opt_config,
        tracking_metrics=tracking_metrics,
        is_test=True,
        status_quo=sq_arm,
    )

    # Adding arms with experimental metrics
    t_bounds = torch.full(
        (2, len(parameter_names)), param_bounds[0], dtype=torch.double
    )
    t_bounds[1] = param_bounds[1]
    X = None
    if num_experimental_trials > 0:
        X = draw_sobol_samples(
            bounds=t_bounds, n=num_experimental_trials, q=1, seed=0
        ).squeeze(1)
    for t in range(num_experimental_trials):
        arm = {}
        for i, param_name in enumerate(experiment.search_space.parameters.keys()):
            arm[param_name] = none_throws(X)[t, i].item()
        gr = (
            # pyre-ignore: Incompatible parameter type [6]
            GeneratorRun(arms=[Arm(parameters=arm), Arm(parameters=sq)])
            if include_sq
            else GeneratorRun(arms=[Arm(parameters=arm)])
        )
        trial = experiment.new_batch_trial(generator_run=gr)
        raw_data = experimental_metric_eval(
            parameters={a.name: a.parameters for a in trial.arms},
            metric_names=tracking_metric_names,
        )
        # create incomplete data by dropping the first metric
        if partial_data:
            for v in raw_data.values():
                del assert_is_instance(v, dict)[tracking_metric_names[-1]]
        trial.attach_batch_trial_data(raw_data=raw_data)
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

    # Adding arms with preferential queries
    if not unbounded_search_space and num_preference_trials > 0:
        X = draw_sobol_samples(
            bounds=t_bounds, n=2 * num_preference_trials, q=1, seed=0
        ).squeeze(1)
    for t in range(num_preference_trials):
        arms = []
        for j in range(2):
            param_dict = {}
            for i, param_name in enumerate(experiment.search_space.parameters.keys()):
                # if this experiment used as PE experiment
                if unbounded_search_space:
                    # matching how metrics are generated in experimental_metric_eval
                    param_dict[param_name] = _metric_name_to_value(
                        metric_name=param_name, metric_names=parameter_names
                    )
                else:
                    param_dict[param_name] = none_throws(X)[t * 2 + j, i].item()
            arms.append(Arm(parameters=param_dict))
        gr = GeneratorRun(arms=arms)

        trial = experiment.new_batch_trial(generator_run=gr)
        trial.attach_batch_trial_data(
            raw_data=pairwise_pref_metric_eval(
                parameters={a.name: a.parameters for a in trial.arms}
            )
        )
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

    # Adding preferential queries using previously evaluated arms
    for _ in range(num_preference_trials_w_repeated_arm):
        arms = np.random.choice(
            list(experiment.arms_by_name.values()), 2, replace=False
        )
        trial = experiment.new_batch_trial()
        trial.add_arms_and_weights(arms=arms)
        trial.attach_batch_trial_data(
            raw_data=pairwise_pref_metric_eval(
                parameters={a.name: a.parameters for a in trial.arms}
            )
        )
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

    return experiment
