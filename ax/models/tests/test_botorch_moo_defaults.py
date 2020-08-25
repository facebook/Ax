#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import get_EHVI
from ax.utils.common.testutils import TestCase


class BotorchMOODefaultsTest(TestCase):
    def test_get_NEI_with_chebyshev_and_missing_Ys_error(self):
        model = MultiObjectiveBotorchModel()
        x = torch.zeros(2, 2)
        weights = torch.ones(2)
        with self.assertRaisesRegex(
            ValueError, "Chebyshev Scalarization requires Ys argument"
        ):
            get_NEI(
                model=model,
                X_observed=x,
                objective_weights=weights,
                chebyshev_scalarization=True,
            )

    def test_get_EHVI_input_validation_errors(self):
        model = MultiObjectiveBotorchModel()
        x = torch.zeros(2, 2)
        weights = torch.ones(2)
        ref_point = torch.zeros(2)
        with self.assertRaisesRegex(
            ValueError, "There are no feasible observed points."
        ):
            get_EHVI(model=model, objective_weights=weights, ref_point=ref_point)
        with self.assertRaisesRegex(
            ValueError, "Expected Hypervolume Improvement requires Ys argument"
        ):
            get_EHVI(
                model=model,
                X_observed=x,
                objective_weights=weights,
                ref_point=ref_point,
            )
