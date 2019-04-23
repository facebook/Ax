#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.testutils import TestCase


class DiscreteModelTest(TestCase):
    def setUp(self):
        pass

    def testDiscreteModelFit(self):
        discrete_model = DiscreteModel()
        discrete_model.fit(
            Xs=[[[0]]],
            Ys=[[0]],
            Yvars=[[1]],
            parameter_values=[[0, 1]],
            outcome_names=[],
        )

    def testdiscreteModelPredict(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.predict([[0]])

    def testdiscreteModelGen(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.gen(
                n=1, parameter_values=[[0, 1]], objective_weights=np.array([1])
            )

    def testdiscreteModelCrossValidate(self):
        discrete_model = DiscreteModel()
        with self.assertRaises(NotImplementedError):
            discrete_model.cross_validate(
                Xs_train=[[[0]]], Ys_train=[[1]], Yvars_train=[[1]], X_test=[[1]]
            )
