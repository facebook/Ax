#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import cast

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.cbo_lcea import LCEABO
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.models.contextual import LCEAGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.datasets import SupervisedDataset


class LCEABOTest(TestCase):
    @mock_botorch_optimize
    def test_LCEABO(self) -> None:
        train_X = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]])
        train_Yvar = 0.1 * torch.ones(3, 1)
        feature_names = ["x0", "x1", "x2", "x3"]
        metric_names = ["y"]
        training_data = [
            SupervisedDataset(
                X=train_X,
                Y=train_Y,
                Yvar=train_Yvar,
                feature_names=feature_names,
                outcome_names=metric_names,
            )
        ]

        # Test setting attributes
        decomposition = {"1": ["x0", "x1"], "2": ["x2", "x3"]}
        m1 = LCEABO(decomposition=decomposition)
        self.assertDictEqual(m1.decomposition, decomposition)
        self.assertTrue(m1.train_embedding)
        self.assertIsNone(m1.cat_feature_dict)
        self.assertIsNone(m1.embs_feature_dict)
        self.assertIsNone(m1.embs_dim_list)

        # Test fit
        m1.fit(
            datasets=training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 1.0) for _ in range(4)],
            ),
        )
        self.assertIsInstance(m1.model, LCEAGP)

        # Test get_and_fit_model with single metric
        gp = m1.get_and_fit_model(
            Xs=[train_X],
            Ys=[train_Y],
            Yvars=[train_Yvar],
            task_features=[],
            fidelity_features=[],
            metric_names=["y"],
        )
        self.assertIsInstance(gp, LCEAGP)

        # Test get_and_fit_model with multiple metrics
        gp_list = m1.get_and_fit_model(
            Xs=[train_X, train_X],
            Ys=[train_Y, train_Y],
            Yvars=[train_Yvar, train_Yvar],
            task_features=[],
            fidelity_features=[],
            metric_names=metric_names,
        )
        self.assertIsInstance(gp_list, ModelListGP)

        # Test decomposition validation in __init__
        with self.assertRaises(AssertionError):
            LCEABO(decomposition={"1": ["x0"], "2": ["x1", "x3"]})

        # Test input decomposition indicates parameter name
        m2 = LCEABO(decomposition={"1": ["x0", "x2"], "2": ["x1", "x3"]})
        m2.fit(
            datasets=training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 1.0) for _ in range(4)],
            ),
        )
        self.assertDictEqual(
            cast(LCEAGP, m2.model).decomposition, {"1": [0, 2], "2": [1, 3]}
        )

        # Test decomposition validation in get_and_fit_model
        # does not pass feature names when decomposition uses feature names
        with self.assertRaises(ValueError):
            m2.fit(
                datasets=training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[(0.0, 1.0) for _ in range(4)],
                ),
            )

        # pass wrong feature names
        with self.assertRaises(AssertionError):
            m2.fit(
                datasets=training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=["x0", "x6", "x10", "z"],
                    bounds=[(0.0, 1.0) for _ in range(4)],
                ),
            )
