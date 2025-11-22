# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.surrogate.hss.base import HierarchicalSearchSpaceSurrogate
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from xgboost import XGBRegressor


CIFAR10_SURROGATE_BASELINE_VALUE = 0.7594003677368164
CIFAR10_SURROGATE_OPTIMAL_VALUE = 0.82043886


def get_cifar10_surrogate_search_space() -> SearchSpace:
    search_space = SearchSpace(
        parameters=[
            FixedParameter(
                name="root",
                value=True,
                parameter_type=ParameterType.BOOL,
                dependents={
                    True: [
                        "lr",
                        "two-more-blocks",
                        "use_softplus_activation",
                        "use_weight_decay",
                    ],
                },
            ),
            RangeParameter(
                name="lr",
                parameter_type=ParameterType.FLOAT,
                lower=1e-5,
                upper=1e-1,
                log_scale=True,
            ),
            ChoiceParameter(
                name="two-more-blocks",
                parameter_type=ParameterType.BOOL,
                values=[False, True],
                is_ordered=True,
                sort_values=True,
            ),
            ChoiceParameter(
                name="use_softplus_activation",
                parameter_type=ParameterType.BOOL,
                values=[False, True],
                is_ordered=True,
                sort_values=True,
                dependents={
                    False: [],
                    True: ["softplus_beta"],
                },
            ),
            RangeParameter(
                name="softplus_beta",
                parameter_type=ParameterType.FLOAT,
                lower=0.5,
                upper=2.0,
            ),
            ChoiceParameter(
                "use_weight_decay",
                parameter_type=ParameterType.BOOL,
                values=[False, True],
                dependents={
                    False: [],
                    True: ["weight_decay"],
                },
                is_ordered=True,
                sort_values=True,
            ),
            RangeParameter(
                name="weight_decay",
                parameter_type=ParameterType.FLOAT,
                lower=1e-8,
                upper=1e-2,
                log_scale=True,
            ),
        ],
    )

    return search_space


def get_cifar10_surrogate_arguments() -> tuple[
    list[list[str]], list[str], list[dict[str, bool]], list[XGBRegressor]
]:
    """
    Construct the arguments to be passed to `HierarchicalSearchSpaceSurrogate` that
    creates a surrogate model on CIFAR10.

    Examples:
    >>> lst_active_param_names, flag_param_names, lst_flag_config, lst_xgb_models = (
    ...     get_fashion_mnist_surrogate_arguments()
    ... )
    >>> surrogate = HierarchicalSearchSpaceSurrogate(
    ...     outcome_names=["test accuracy"],
    ...     lst_active_param_names=lst_active_param_names,
    ...     flag_param_names=flag_param_names,
    ...     lst_flag_config=lst_flag_config,
    ...     lst_xgb_models=lst_xgb_models,
    ... )
    """
    lst_active_param_names: list[list[str]] = [
        ["lr", "two-more-blocks"],
        ["lr", "two-more-blocks", "weight_decay"],
        ["lr", "two-more-blocks", "softplus_beta"],
        ["lr", "two-more-blocks", "softplus_beta", "weight_decay"],
    ]

    flag_param_names: list[str] = [
        "use_softplus_activation",
        "use_weight_decay",
    ]

    lst_flag_config: list[dict[str, bool]] = [
        {"use_softplus_activation": False, "use_weight_decay": False},
        {"use_softplus_activation": False, "use_weight_decay": True},
        {"use_softplus_activation": True, "use_weight_decay": False},
        {"use_softplus_activation": True, "use_weight_decay": True},
    ]

    lst_xgb_models: list[XGBRegressor] = [XGBRegressor() for i in range(4)]

    for i, xgb_model in enumerate(lst_xgb_models):
        xgb_model.load_model(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "checkpoints",
                f"cifar10_xgb_model_{i:d}.json",
            )
        )

    return lst_active_param_names, flag_param_names, lst_flag_config, lst_xgb_models


def get_cifar10_surrogate_benchmark(
    num_trials: int = 50,
) -> BenchmarkProblem:
    """Create a BenchmarkProblem for tuning hyperparameters of a residual network on
    CIFAR10. The tunable hyperparameters are as follows:
    1. The learning rate;
    2. Whether to use two additional residual blocks;
    3. Whether to use the softplus activation (instead of ReLU), and the coefficient
    beta in the softplus activation;
    4. Whether to use weight decay, and the weight decay coefficient.
    More details on the tunable hyperparameters can be found in the return value of
    `get_cifar10_surrogate_search_space`. The returned benchmark problem uses
    gradient-boosted trees to predict the validation accuracy, which makes the oracle
    computationally cheap to evaluate.
    """
    search_space = get_cifar10_surrogate_search_space()

    lst_active_param_names, flag_param_names, lst_flag_config, lst_xgb_models = (
        get_cifar10_surrogate_arguments()
    )

    test_function = HierarchicalSearchSpaceSurrogate(
        outcome_names=["CIFAR10 Test Accuracy"],
        lst_active_param_names=lst_active_param_names,
        flag_param_names=flag_param_names,
        lst_flag_config=lst_flag_config,
        lst_xgb_models=lst_xgb_models,
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(
                name="CIFAR10 Test Accuracy",
                lower_is_better=False,
                observe_noise_sd=False,
            ),
            minimize=False,
        )
    )

    return BenchmarkProblem(
        name="CIFAR10 Test Accuracy",
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=CIFAR10_SURROGATE_OPTIMAL_VALUE,
        test_function=test_function,
        baseline_value=CIFAR10_SURROGATE_BASELINE_VALUE,
    )
