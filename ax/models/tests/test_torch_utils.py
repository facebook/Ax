# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Tuple

import torch
from ax.models.torch.utils import _get_X_pending_and_observed
from ax.utils.common.testutils import TestCase


class TorchUtilsTest(TestCase):
    def test_get_X_pending_and_observed(self):
        def _to_obs_set(X: torch.Tensor) -> Set[Tuple[float]]:
            return {tuple(float(x_i) for x_i in x) for x in X}

        # Apply filter normally
        Xs = [torch.tensor([[0.0, 0.0], [0.0, 1.0]])]
        objective_weights = torch.tensor([1.0])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        fixed_features = {1: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0][1:]
        # self.assertTrue(torch.equal(X_observed, expected), f"{X_observed}{expected}")
        self.assertEqual(_to_obs_set(expected), _to_obs_set(X_observed))

        # Filter too strict; return unfiltered X_observed
        fixed_features = {0: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0]
        # self.assertTrue(torch.equal(X_observed, expected), f"{X_observed}{expected}")
        self.assertEqual(_to_obs_set(expected), _to_obs_set(X_observed))
