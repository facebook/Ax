# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from pyre_extensions import assert_is_instance
from torch import Tensor


@dataclass(kw_only=True)
class BoTorchTestFunction(BenchmarkTestFunction):
    """
    Class for generating data from a BoTorch ``BaseTestProblem``.

    Args:
        outcome_names: Names of outcomes. Should have the same length as the
            dimension of the test function, including constraints.
        botorch_problem: The BoTorch ``BaseTestProblem``.
        use_shifted_function: Whether to use the shifted version of the test function.
            If True, an offset tensor is randomly drawn from the test problem bounds,
            and the we evaluate `f(X-offset)` rather than `f(X)`. This is useful for
            changing the location of the optima for test functions that favor the
            center of the search space.
        modified_bounds: The bounds that are used by the Ax search space
            while optimizing the problem. If different from the bounds of the
            test problem, we project the parameters into the test problem
            bounds before evaluating the test problem.
            For example, if the test problem is defined on [0, 1] but the Ax
            search space is integers in [0, 10], an Ax parameter value of
            5 will correspond to 0.5 while evaluating the test problem.
            If modified bounds are not provided, the test problem will be
            evaluated using the raw parameter values.
        dummy_param_names: Names of parameters that do not affect the value of
            the test function because they are not passed to
            ``self.botorch_problem``.
            self.botorch_problem.dim + len(dummy_param_names) should equal the
            number of parameters in the ``params`` passed to ``evaluate_true``.
        n_steps: Number of data points produced per metric and per evaluation. 1
            if data is not time-series. If data is time-series, this will
            eventually become the number of values on a `MapMetric` for
            evaluations that run to completion.
    """

    outcome_names: Sequence[str]
    botorch_problem: BaseTestProblem
    use_shifted_function: bool = False
    modified_bounds: Sequence[tuple[float, float]] | None = None
    _offset: torch.Tensor | None = None
    _original_bounds: torch.Tensor | None = None
    dummy_param_names: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if (
            self.botorch_problem.noise_std is not None
            or getattr(self.botorch_problem, "constraint_noise_std", None) is not None
        ):
            raise ValueError(
                "noise should be set on the `BenchmarkRunner`, not the test function."
            )
        self.botorch_problem = self.botorch_problem.to(dtype=torch.double)
        self._original_bounds = self.botorch_problem.bounds.clone()
        if self.use_shifted_function:
            lo, hi = self.botorch_problem.bounds.unbind(dim=0)
            self._offset = lo + (hi - lo) * torch.rand(
                self.botorch_problem.dim, dtype=torch.double
            )
            # Modify the bounds to match the shifted problem.
            # The Ax-specified bounds are [2 * lo, 2 * hi] which implies that
            # X - offset is in [2 * lo - offset, 2 * hi - offset].
            self.botorch_problem._bounds = [
                (2 * bound[0] - offset, 2 * bound[1] - offset)
                for bound, offset in zip(self.botorch_problem._bounds, self._offset)
            ]
            self.botorch_problem.bounds = (
                2.0 * self.botorch_problem.bounds
                - assert_is_instance(self._offset, Tensor)
            )

    def tensorize_params(self, params: Mapping[str, int | float]) -> torch.Tensor:
        """Converts parameters to a 1d tensor.

        If modified bounds are provided, we normalize the parameters from the modified
        bounds to the unit cube, and then unnormalize to the original problem bounds.

        If `use_shifted_function=True`, we subtract the offset from the resulting tensor
        before returning it.
        """
        X = torch.tensor(
            [v for k, v in params.items() if k not in self.dummy_param_names],
            dtype=torch.double,
        )

        if self.modified_bounds is not None:
            # Normalize from modified bounds to unit cube.
            unit_X = normalize(
                X, torch.tensor(self.modified_bounds, dtype=torch.double).T
            )
            # Unnormalize from unit cube to original problem bounds.
            X = unnormalize(unit_X, assert_is_instance(self._original_bounds, Tensor))
        if self.use_shifted_function:
            X = X - assert_is_instance(self._offset, torch.Tensor)
        return X

    # pyre-fixme [14]: inconsistent override
    def evaluate_true(self, params: Mapping[str, float | int]) -> torch.Tensor:
        expected_n_dims = self.botorch_problem.dim + len(self.dummy_param_names)
        if len(params) != expected_n_dims:
            raise ValueError(
                f"Expected {expected_n_dims} parameters, got {len(params)}."
            )
        x = self.tensorize_params(params=params)
        # self.botorch_problem(x) has shape [n_metrics] if n_metrics > 1,
        # otherwise []. So `objectives` has shape [n_metrics]
        objectives = torch.atleast_1d(self.botorch_problem(x))
        if isinstance(self.botorch_problem, ConstrainedBaseTestProblem):
            constraints = torch.atleast_1d(self.botorch_problem.evaluate_slack_true(x))
            metrics = torch.cat([objectives, constraints], dim=-1)
        else:
            metrics = objectives
        # shape (n_metrics, 1)
        metrics = metrics.unsqueeze(1)
        if self.n_steps == 1:
            return metrics
        # shape (n_metrics, n_steps)
        return metrics.repeat(1, self.n_steps)
