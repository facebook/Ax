#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch
from ax.models.torch.rembo import REMBO
from ax.utils.common.testutils import TestCase


class REMBOTest(TestCase):
    def testREMBOModel(self):
        A = torch.cat((torch.eye(2), -(torch.eye(2))))
        initial_X_d = torch.tensor([[0.25, 0.5], [1, 0], [0, -1]])
        bounds_d = [(-2, 2), (-2, 2)]

        # Test setting attributes
        m = REMBO(A=A, initial_X_d=initial_X_d, bounds_d=bounds_d)
        self.assertTrue(torch.allclose(A, m.A))
        self.assertTrue(torch.allclose(torch.pinverse(A), m._pinvA))
        self.assertEqual(m.bounds_d, bounds_d)
        self.assertEqual(len(m.X_d), 3)

        # Test fit
        # Create high-D data
        X_D = torch.t(torch.mm(A, torch.t(initial_X_d)))
        Xs = [X_D, X_D.clone()]
        Ys = [torch.randn(3, 1)] * 2
        Yvars = [0.1 * torch.ones(3, 1)] * 2
        bounds = [(-1, 1)] * 4
        with self.assertRaises(AssertionError):
            m.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=[(0, 1)] * 4,
                task_features=[],
                feature_names=[],
                metric_names=[],
                fidelity_features=[],
            )

        with mock.patch(
            "ax.models.torch.botorch_defaults.fit_gpytorch_model", autospec=True
        ):
            m.fit(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                bounds=bounds,
                task_features=[],
                feature_names=[],
                metric_names=[],
                fidelity_features=[],
            )
        # Check was fit with the low-d data.
        for x in m.Xs:
            self.assertTrue(torch.allclose(x, m.to_01(initial_X_d)))

        self.assertEqual(len(m.X_d), 3)

        # Test project up
        X_d2 = torch.tensor([[0.25, 0.5], [2.0, 0.0], [-4.0, 4.0]])
        X_D2 = torch.tensor(
            [[0.25, 0.5, -0.25, -0.5], [1.0, 0.0, -1.0, 0.0], [-1.0, 1.0, 1.0, -1.0]]
        )
        Z = m.project_up(X_d2)
        self.assertTrue(torch.allclose(Z, X_D2))

        # Test predict
        f1, var = m.predict(X=X_D)
        self.assertEqual(f1.shape, torch.Size([3, 2]))
        with self.assertRaises(NotImplementedError):
            m.predict(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

        f2, var = m.predict(initial_X_d)
        self.assertTrue(torch.allclose(f1, f2))

        # Test best_point
        x_best = m.best_point(
            bounds=[(-1, 1)] * 4, objective_weights=torch.tensor([1.0, 0.0])
        )
        self.assertEqual(len(x_best), 4)

        # Test cross_validate
        f, var = m.cross_validate(
            Xs_train=[X_D[:-1, :], X_D[:-1, :]],
            Ys_train=[Ys[0][:-1, :], Ys[1][:-1, :]],
            Yvars_train=[Yvars[0][:-1, :], Yvars[1][:-1, :]],
            X_test=X_D[-1:, :],
        )
        self.assertEqual(f.shape, torch.Size([1, 2]))

        # Test gen
        Xgen_d = torch.tensor([[0.4, 0.8], [-0.2, 1.0]])
        acqfv_dummy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        with mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            autospec=True,
            return_value=(Xgen_d, acqfv_dummy),
        ):
            Xgen, w, _, __ = m.gen(
                n=2, bounds=[(-1, 1)] * 4, objective_weights=torch.tensor([1.0, 0.0])
            )
        self.assertEqual(Xgen.shape[1], 4)
        self.assertEqual(len(m.X_d), 5)

        # Test update
        with self.assertRaises(ValueError):
            m.update(
                Xs=[torch.tensor([[0.1, 0.2, 0.3, 0.4]])] * 2,
                Ys=[torch.randn(1, 1)] * 2,
                Yvars=[torch.ones(1, 1)] * 2,
            )

        m.update(
            Xs=[Xgen] * 2, Ys=[torch.randn(2, 1)] * 2, Yvars=[torch.ones(2, 1)] * 2
        )
        for x in m.Xs:
            self.assertTrue(torch.allclose(x, Xgen_d))
