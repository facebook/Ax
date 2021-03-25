#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.cbo_sac import SACBO, SACGP
from ax.utils.common.testutils import TestCase
from botorch.models.model_list_gp_regression import ModelListGP


class SACBOTest(TestCase):
    @mock.patch("ax.models.torch.botorch_defaults.fit_gpytorch_model")
    def test_SACBO(self, mock_model_fit):
        train_X = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        train_Yvar = 0.1 * torch.ones(3, 1, dtype=torch.double)

        # test setting attributes
        decomposition = {"1": ["0", "1"], "2": ["2", "3"]}
        m1 = SACBO(decomposition=decomposition)
        self.assertDictEqual(decomposition, m1.decomposition)

        # test fit
        m1.fit(
            Xs=[train_X],
            Ys=[train_Y],
            Yvars=[train_Yvar],
            search_space_digest=SearchSpaceDigest(
                feature_names=["0", "1", "2", "3"],
                bounds=[(0.0, 1.0) for _ in range(4)],
            ),
            metric_names=["y"],
        )
        self.assertIsInstance(m1.model, SACGP)

        # test get_and_fit_model with single metric
        gp = m1.get_and_fit_model(
            Xs=[train_X],
            Ys=[train_Y],
            Yvars=[train_Yvar],
            task_features=[],
            fidelity_features=[],
            metric_names=["y"],
        )
        self.assertIsInstance(gp, SACGP)

        # test get_and_fit_model with multiple metrics
        gp_list = m1.get_and_fit_model(
            Xs=[train_X, train_X],
            Ys=[train_Y, train_Y],
            Yvars=[train_Yvar, train_Yvar],
            task_features=[],
            fidelity_features=[],
            metric_names=["y"],
        )
        self.assertIsInstance(gp_list, ModelListGP)

        # test decomposition validation in __init__
        with self.assertRaises(AssertionError):
            SACBO(decomposition={"1": ["x1"], "2": ["x2", "x4"]})

        # test input decomposition indicates parameter name
        m2 = SACBO(decomposition={"1": ["x1", "x3"], "2": ["x2", "x4"]})
        m2.fit(
            Xs=[train_X],
            Ys=[train_Y],
            Yvars=[train_Yvar],
            search_space_digest=SearchSpaceDigest(
                feature_names=["x1", "x2", "x3", "x4"],
                bounds=[(0.0, 1.0) for _ in range(4)],
            ),
            metric_names=["y"],
        )
        self.assertDictEqual(m2.model.decomposition, {"1": [0, 2], "2": [1, 3]})

        # test decomposition validation in get_and_fit_model
        # the feature_names is not passed
        with self.assertRaises(ValueError):
            m2.fit(
                Xs=[train_X],
                Ys=[train_Y],
                Yvars=[train_Yvar],
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[(0.0, 1.0) for _ in range(4)],
                ),
                metric_names=[],
            )

        # pass wrong feature_names
        with self.assertRaises(AssertionError):
            m2.fit(
                Xs=[train_X],
                Ys=[train_Y],
                Yvars=[train_Yvar],
                search_space_digest=SearchSpaceDigest(
                    feature_names=["x0", "x1", "x2", "x3"],
                    bounds=[(0.0, 1.0) for _ in range(4)],
                ),
                metric_names=["y"],
            )
