#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.randomforest import RandomForest
from ax.utils.common.testutils import TestCase
from botorch.utils.datasets import SupervisedDataset


class RandomForestTest(TestCase):
    def test_RFModel(self) -> None:
        datasets = [
            SupervisedDataset(
                X=torch.rand(10, 2),
                Y=torch.rand(10, 1),
                Yvar=torch.rand(10, 1),
                feature_names=["x1", "x2"],
                outcome_names=[f"y{i}"],
            )
            for i in range(2)
        ]
        search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2"],
            bounds=[(0, 1)] * 2,
        )

        m = RandomForest(num_trees=5)
        m.fit(
            datasets=datasets,
            search_space_digest=search_space_digest,
        )
        self.assertEqual(len(m.models), 2)
        # pyre-fixme[16]: `RandomForestRegressor` has no attribute `estimators_`.
        self.assertEqual(len(m.models[0].estimators_), 5)

        f, cov = m.predict(torch.rand(5, 2))
        self.assertEqual(f.shape, torch.Size((5, 2)))
        self.assertEqual(cov.shape, torch.Size((5, 2, 2)))

        f, cov = m.cross_validate(
            datasets=datasets,
            search_space_digest=search_space_digest,
            X_test=torch.rand(3, 2),
        )
        self.assertEqual(f.shape, torch.Size((3, 2)))
        self.assertEqual(cov.shape, torch.Size((3, 2, 2)))
