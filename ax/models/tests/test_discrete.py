#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.testutils import TestCase


class DiscreteModelTest(TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        pass

    # pyre-fixme[3]: Return type must be annotated.
    def test_discrete_model_get_state(self):
        discrete_model = DiscreteModel()
        self.assertEqual(discrete_model._get_state(), {})

    # pyre-fixme[3]: Return type must be annotated.
    def test_discrete_model_feature_importances(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.feature_importances()

    # pyre-fixme[3]: Return type must be annotated.
    def testDiscreteModelFit(self):
        discrete_model = DiscreteModel()
        discrete_model.fit(
            Xs=[[[0]]],
            Ys=[[0]],
            Yvars=[[1]],
            parameter_values=[[0, 1]],
            outcome_names=[],
        )

    # pyre-fixme[3]: Return type must be annotated.
    def testdiscreteModelPredict(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.predict([[0]])

    # pyre-fixme[3]: Return type must be annotated.
    def testdiscreteModelGen(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.gen(
                n=1, parameter_values=[[0, 1]], objective_weights=np.array([1])
            )

    # pyre-fixme[3]: Return type must be annotated.
    def testdiscreteModelCrossValidate(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.cross_validate(
                Xs_train=[[[0]]], Ys_train=[[1]], Yvars_train=[[1]], X_test=[[1]]
            )
