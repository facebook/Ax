# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.core.types import TParamValue
from torch import Tensor
from xgboost import XGBRegressor


def load_xgb_regressor(path: str) -> XGBRegressor:
    """Load an XGBRegressor model from a file.

    This function sets `_estimator_type` before calling `load_model()` to
    ensure compatibility with scikit-learn >= 1.8.0, which changed how
    `_estimator_type` is inherited from `RegressorMixin`.

    Args:
        path: Path to the saved XGBoost model file.

    Returns:
        The loaded XGBRegressor model.
    """
    model = XGBRegressor()
    model._estimator_type = "regressor"
    model.load_model(path)
    return model


@dataclass(kw_only=True)
class HierarchicalSearchSpaceSurrogate(BenchmarkTestFunction):
    lst_active_param_names: list[list[str]]
    flag_param_names: list[str]
    lst_flag_config: list[dict[str, bool]]
    lst_xgb_models: list[XGBRegressor]

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        This function does the following two things to compute the test function values:
        1. Parse the flag parameters to decide which model to use;
        2. Collect all active parameters and pass them to the selected model.
        """
        flag_config = {
            key: value for key, value in params.items() if key in self.flag_param_names
        }

        # pyre-ignore[6]
        task_id = self.lst_flag_config.index(flag_config)

        active_param_names = self.lst_active_param_names[task_id]

        y = self.lst_xgb_models[task_id].predict(
            np.array([params[name] for name in active_param_names])[None, :]
        )

        return torch.from_numpy(y).reshape(1, 1)
