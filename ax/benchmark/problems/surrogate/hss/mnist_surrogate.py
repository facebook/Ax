# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.problems.surrogate.hss.base import (
    HierarchicalSearchSpaceSurrogate,
    load_xgb_regressor,
)
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


# The optimal value is obtained by a dense grid search, whereas the baseline value is
# obtained by `compute_baseline_value_from_sobol`
MNIST_SURROGATE_OPTIMAL_VALUE = 1.0790634
MNIST_SURROGATE_BASELINE_VALUE = 0.9837410950660705


def get_mnist_surrogate_search_space() -> SearchSpace:
    search_space = SearchSpace(
        parameters=[
            FixedParameter(
                name="root",
                value=True,
                parameter_type=ParameterType.BOOL,
                dependents={
                    True: ["lr", "use_dropout", "use_weight_decay"],
                },
            ),
            RangeParameter(
                name="lr",
                parameter_type=ParameterType.FLOAT,
                lower=1e-5,
                upper=1,
                log_scale=True,
            ),
            ChoiceParameter(
                "use_dropout",
                parameter_type=ParameterType.BOOL,
                values=[False, True],
                is_ordered=True,
                sort_values=True,
                dependents={
                    False: [],
                    True: ["dropout"],
                },
            ),
            RangeParameter(
                name="dropout",
                parameter_type=ParameterType.FLOAT,
                lower=0.1,
                upper=0.5,
            ),
            ChoiceParameter(
                "use_weight_decay",
                parameter_type=ParameterType.BOOL,
                values=[False, True],
                is_ordered=True,
                sort_values=True,
                dependents={
                    False: [],
                    True: ["weight_decay"],
                },
            ),
            RangeParameter(
                name="weight_decay",
                parameter_type=ParameterType.FLOAT,
                lower=1e-5,
                upper=1,
                log_scale=True,
            ),
        ],
    )

    return search_space


def get_mnist_surrogate_arguments() -> tuple[
    list[list[str]], list[str], list[dict[str, bool]], list[XGBRegressor]
]:
    lst_active_param_names: list[list[str]] = [
        ["lr"],
        ["lr", "weight_decay"],
        ["lr", "dropout"],
        ["lr", "dropout", "weight_decay"],
    ]

    flag_param_names: list[str] = ["use_dropout", "use_weight_decay"]

    lst_flag_config: list[dict[str, bool]] = [
        {"use_dropout": False, "use_weight_decay": False},
        {"use_dropout": False, "use_weight_decay": True},
        {"use_dropout": True, "use_weight_decay": False},
        {"use_dropout": True, "use_weight_decay": True},
    ]

    lst_xgb_models: list[XGBRegressor] = [
        load_xgb_regressor(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "checkpoints",
                f"mnist_xgb_model_{i:d}.json",
            )
        )
        for i in range(4)
    ]

    return lst_active_param_names, flag_param_names, lst_flag_config, lst_xgb_models


def get_mnist_surrogate_benchmark(
    num_trials: int = 50,
) -> BenchmarkProblem:
    """Creates a BenchmarkProblem for tuning hyperparameters of a MNIST convolutional
    neural network.

    The hyperparameters are the learning rate, dropout rate, and weight decay. The
    dropout rate and the weight decay coefficient are optional hyperparameters, and can
    be turned off. This benchmark problem uses a surrogate model, i.e., gradient
    boosted trees, to predict the validation accuracy, which makes the oracle
    computationally cheap to evaluate.
    """
    search_space = get_mnist_surrogate_search_space()

    lst_active_param_names, flag_param_names, lst_flag_config, lst_xgb_models = (
        get_mnist_surrogate_arguments()
    )

    test_function = HierarchicalSearchSpaceSurrogate(
        outcome_names=["MNIST Test Accuracy"],
        lst_active_param_names=lst_active_param_names,
        flag_param_names=flag_param_names,
        lst_flag_config=lst_flag_config,
        lst_xgb_models=lst_xgb_models,
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(
                name="MNIST Test Accuracy",
                lower_is_better=False,
                observe_noise_sd=False,
            ),
            minimize=False,
        )
    )

    return BenchmarkProblem(
        name="MNIST Test Accuracy",
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=MNIST_SURROGATE_OPTIMAL_VALUE,
        test_function=test_function,
        baseline_value=MNIST_SURROGATE_BASELINE_VALUE,
    )
