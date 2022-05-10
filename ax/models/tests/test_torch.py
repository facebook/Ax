#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch_base import TorchModel
from ax.utils.common.testutils import TestCase
from botorch.utils.datasets import FixedNoiseDataset


class TorchModelTest(TestCase):
    def setUp(self):
        self.dataset = FixedNoiseDataset(
            X=torch.zeros(1), Y=torch.zeros(1), Yvar=torch.ones(1)
        )

    def testTorchModelFit(self):
        torch_model = TorchModel()
        torch_model.fit(
            datasets=[self.dataset],
            metric_names=["y"],
            search_space_digest=SearchSpaceDigest(
                feature_names=["x1"],
                bounds=[(0, 1)],
            ),
        )

    def testTorchModelPredict(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.predict(torch.zeros(1))

    def testTorchModelGen(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.gen(n=1, bounds=[(0, 1)], objective_weights=torch.ones(1))

    def testNumpyTorchBestPoint(self):
        torch_model = TorchModel()
        x = torch_model.best_point(bounds=[(0, 1)], objective_weights=torch.ones(1))
        self.assertIsNone(x)

    def testTorchModelCrossValidate(self):
        torch_model = TorchModel()
        with self.assertRaises(NotImplementedError):
            torch_model.cross_validate(
                datasets=[self.dataset],
                metric_names=["y"],
                X_test=torch.ones(1),
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )

    def testTorchModelUpdate(self):
        model = TorchModel()
        with self.assertRaises(NotImplementedError):
            model.update(
                datasets=[self.dataset],
                metric_names=["y"],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
