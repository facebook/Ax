# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Optional, Union

import torch
from ax.benchmark.runners.botorch_test import ParamBasedTestProblem


class TestParamBasedTestProblem(ParamBasedTestProblem):
    optimal_value: float = 0.0

    def __init__(
        self, num_objectives: int, noise_std: Optional[Union[float, list[float]]]
    ) -> None:
        self.num_objectives = num_objectives
        self.noise_std = noise_std

    # pyre-fixme[14]: Inconsistent override, as dict[str, float] is not a
    # `TParameterization`
    def evaluate_true(self, params: dict[str, float]) -> torch.Tensor:
        value = sum(elt**2 for elt in params.values())
        return value * torch.ones(self.num_objectives, dtype=torch.double)
