#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction


@dataclass(kw_only=True)
class IdentityTestFunction(BenchmarkTestFunction):
    """
    Test function that returns the value of parameter "x0", ignoring any others.
    """

    outcome_names: Sequence[str] = field(default_factory=lambda: ["objective"])
    n_steps: int = 1

    # pyre-fixme[14]: Inconsistent override
    def evaluate_true(self, params: Mapping[str, float]) -> torch.Tensor:
        """
        Return params["x0"] for each outcome for each time step.

        Args:
            params: A dictionary with key "x0".
        """
        value = params["x0"]
        return torch.full(
            (len(self.outcome_names), self.n_steps), value, dtype=torch.float64
        )
