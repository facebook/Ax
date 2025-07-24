# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from dataclasses import dataclass, field
from typing import Mapping, Union

import numpy as np

import torch

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_problem import BenchmarkProblem

from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace

from torch import Tensor
from xgboost import XGBRegressor


# The optimal value is obtained by a dense grid search, whereas the baseline value is
# obtained by `compute_baseline_value_from_sobol`
MNIST_SURROGATE_OPTIMAL_VALUE = 1.0790634
MNIST_SURROGATE_BASELINE_VALUE = 0.9837410950660705


def get_mnist_surrogate_search_space() -> HierarchicalSearchSpace:
    search_space = HierarchicalSearchSpace(
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

    test_function = MNISTSurrogate(
        outcome_names=["MNISTSurrogate"],
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=BenchmarkMetric(name="MNISTSurrogate", lower_is_better=False),
            minimize=False,
        )
    )

    return BenchmarkProblem(
        name="MNISTSurrogate",
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=MNIST_SURROGATE_OPTIMAL_VALUE,
        test_function=test_function,
        baseline_value=MNIST_SURROGATE_BASELINE_VALUE,
    )


def load_gradient_boosted_trees() -> list[XGBRegressor]:
    """
    Load XGBoost checkpoints.

    Returns:
        A list of XGBoost regressors that predict the validation loss of the
        convolutional neural network from the its hyperparameters.
    """
    base_path = os.path.dirname(os.path.realpath(__file__))

    lst_xgb_models = [XGBRegressor() for i in range(4)]

    for i, xgb_model in enumerate(lst_xgb_models):
        xgb_model.load_model(
            os.path.join(
                base_path,
                "checkpoints",
                f"mnist_xgb_model_{i:d}.json",
            )
        )

    return lst_xgb_models


@dataclass(kw_only=True)
class MNISTSurrogate(BenchmarkTestFunction):
    """
    A XGBoost surrogate model for MNIST neural network hyperparameter tuning. The
    architecture of the network is shown below with ReLU activations:
    ```
    self.conv1 = nn.Conv2d(1, 8, 3, 1, 1)
    self.conv2 = nn.Conv2d(8, 8, 3, 1, 1)
    self.fc1 = nn.Linear(8 * 28 * 28, 32)
    self.fc2 = nn.Linear(32, 10)
    ```

    The tunable parameters are the learning rate, dropout rate, and weight decay
    coefficient. The dropout rate and the weight decay coefficient are optional
    hyperparameters.

    There are 4 different cases depending on whether the dropout and weight decay are
    used. ``lst_active_param_names`` and ``lst_flag_config``` track the active
    parameters and are used to select the corresponding XGBoost model for prediction.
    """

    # We use ``lst_active_param_names`` to track the active parameters in each case.
    lst_active_param_names: list[list[str]] = field(
        default_factory=lambda: [
            ["lr"],
            ["lr", "weight_decay"],
            ["lr", "dropout"],
            ["lr", "dropout", "weight_decay"],
        ]
    )

    # This is a list of boolean flag parameters that indicate whether the dropout and
    # weight decay are used.
    flag_param_names: list[str] = field(
        default_factory=lambda: ["use_dropout", "use_weight_decay"]
    )

    # Enumerate all possible combinations of the boolean flag parameters
    lst_flag_config: list[dict[str, bool]] = field(
        default_factory=lambda: [
            {"use_dropout": False, "use_weight_decay": False},
            {"use_dropout": False, "use_weight_decay": True},
            {"use_dropout": True, "use_weight_decay": False},
            {"use_dropout": True, "use_weight_decay": True},
        ]
    )

    # Load the gradient-boosted trees that predict the validation accuracy of the MNIST.
    # Each case uses a separate XGBoost model.
    lst_xgb_models: list[XGBRegressor] = field(
        default_factory=load_gradient_boosted_trees
    )

    def evaluate_true(
        self,
        params: Mapping[str, Union[None, bool, float, int, str]],
    ) -> Tensor:
        """
        Estimate the validation accuracy of the MNIST convolutional neural network.

        Args:
            params: A dictionary of hyperparameters and their values.

        Returns:
            The validation accuracy predicted by gradient-boosted trees.
        """
        flag_config = {
            key: value for key, value in params.items() if key in self.flag_param_names
        }

        task_id = self.lst_flag_config.index(flag_config)  # pyre-ignore[6]

        active_param_names = self.lst_active_param_names[task_id]

        y = self.lst_xgb_models[task_id].predict(
            np.array([params[name] for name in active_param_names])[None, :]
        )

        return torch.from_numpy(y).reshape(1, 1)
