#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.testutils import TestCase


class DiscreteModelTest(TestCase):
    def test_discrete_model_get_state(self) -> None:
        discrete_model = DiscreteModel()
        self.assertEqual(discrete_model._get_state(), {})

    def test_discrete_model_feature_importances(self) -> None:
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.feature_importances()

    def test_DiscreteModelFit(self) -> None:
        discrete_model = DiscreteModel()
        discrete_model.fit(
            Xs=[[[0]]],
            Ys=[[0]],
            Yvars=[[1]],
            parameter_values=[[0, 1]],
            outcome_names=[],
        )

    def test_discreteModelPredict(self) -> None:
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.predict([[0]])

    def test_discreteModelGen(self) -> None:
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.gen(
                n=1, parameter_values=[[0, 1]], objective_weights=np.array([1])
            )

    def test_discreteModelCrossValidate(self) -> None:
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.cross_validate(
                Xs_train=[[[0]]], Ys_train=[[1]], Yvars_train=[[1]], X_test=[[1]]
            )
