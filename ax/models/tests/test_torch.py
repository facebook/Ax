#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch_base import TorchModel, TorchOptConfig
from ax.utils.common.testutils import TestCase
from botorch.utils.datasets import SupervisedDataset


class TorchModelTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dataset = SupervisedDataset(
            X=torch.zeros(1, 1),
            Y=torch.zeros(1, 1),
            Yvar=torch.ones(1, 1),
            feature_names=["x1"],
            outcome_names=["y"],
        )
        self.search_space_digest = SearchSpaceDigest(
            feature_names=[],
            bounds=[(0, 1)],
        )
        self.torch_opt_config = TorchOptConfig(objective_weights=torch.ones(1))

    def test_TorchModelFit(self) -> None:
        torch_model = TorchModel()
        torch_model.fit(
            datasets=[self.dataset],
            search_space_digest=SearchSpaceDigest(
                feature_names=["x1"],
                bounds=[(0, 1)],
            ),
        )

    def test_TorchModelPredict(self) -> None:
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.predict(torch.zeros(1))

    def test_TorchModelGen(self) -> None:
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.gen(
                n=1,
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )

    def test_NumpyTorchBestPoint(self) -> None:
        torch_model = TorchModel()
        x = torch_model.best_point(
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
        )
        self.assertIsNone(x)

    def test_TorchModelCrossValidate(self) -> None:
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.cross_validate(
                datasets=[self.dataset],
                X_test=torch.ones(1),
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
