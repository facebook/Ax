#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.posterior_mean import get_PosteriorMean
from ax.utils.common.testutils import TestCase


# TODO (jej): Streamline testing for a simple acquisition function.
class PosteriorMeanTest(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.double

        self.Xs = [
            torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=self.dtype, device=self.device
            )
        ]
        self.Ys = [torch.tensor([[3.0], [4.0]], dtype=self.dtype, device=self.device)]
        self.Yvars = [
            torch.tensor([[0.0], [2.0]], dtype=self.dtype, device=self.device)
        ]
        self.bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.objective_weights = torch.tensor(
            [1.0], dtype=self.dtype, device=self.device
        )
        self.outcome_constraints = (
            torch.tensor([[1.0]], dtype=self.dtype, device=self.device),
            torch.tensor([[5.0]], dtype=self.dtype, device=self.device),
        )

    def test_GetPosteriorMean(self):

        model = BotorchModel(acqf_constructor=get_PosteriorMean)
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
            ),
            metric_names=self.metric_names,
        )

        # test model.gen() with no outcome_constraints. Analytic.
        new_X_dummy = torch.rand(1, 1, 3, dtype=self.dtype, device=self.device)
        Xgen, wgen, _, __ = model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            linear_constraints=None,
        )
        self.assertTrue(torch.equal(wgen, torch.ones(1, dtype=self.dtype)))

        # test model.gen() works with outcome_constraints. qSimpleRegret.
        new_X_dummy = torch.rand(1, 1, 3, dtype=self.dtype, device=self.device)
        Xgen, w, _, __ = model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=None,
        )

        # test model.gen() works with chebyshev scalarization.
        model = MultiObjectiveBotorchModel(acqf_constructor=get_PosteriorMean)
        model.fit(
            Xs=self.Xs * 2,
            Ys=self.Ys * 2,
            Yvars=self.Yvars * 2,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
            ),
            metric_names=["m1", "m2"],
        )
        new_X_dummy = torch.rand(1, 1, 3, dtype=self.dtype, device=self.device)
        Xgen, w, _, __ = model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=torch.ones(2, dtype=self.dtype, device=self.device),
            outcome_constraints=(
                torch.tensor([[1.0, 0.0]], dtype=self.dtype, device=self.device),
                torch.tensor([[5.0]], dtype=self.dtype, device=self.device),
            ),
            objective_thresholds=torch.zeros(2, dtype=self.dtype, device=self.device),
            linear_constraints=None,
            model_gen_options={
                "acquisition_function_kwargs": {"chebyshev_scalarization": True}
            },
        )

        # ValueError with empty X_Observed
        with self.assertRaises(ValueError):
            get_PosteriorMean(
                model=model, objective_weights=self.objective_weights, X_observed=None
            )

        # test model.predict()
        new_X_dummy = torch.rand(1, 1, 3, dtype=self.dtype, device=self.device)
        model.predict(new_X_dummy)
