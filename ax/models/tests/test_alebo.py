#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import torch
from ax.models.torch.alebo import (
    ALEBO,
    ALEBOGP,
    ALEBOKernel,
    alebo_acqf_optimizer,
    ei_or_nei,
    extract_map_statedict,
    get_batch_model,
    get_fitted_model,
    get_map_model,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.model_list_gp_regression import ModelListGP


class ALEBOTest(TestCase):
    def testALEBOKernel(self):
        B = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.double
        )
        k = ALEBOKernel(B=B, batch_shape=torch.Size([]))

        self.assertEqual(k.d, 2)
        self.assertTrue(torch.equal(B, k.B))
        self.assertTrue(
            torch.equal(k.triu_indx[0], torch.tensor([0, 0, 1], dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(k.triu_indx[1], torch.tensor([0, 1, 1], dtype=torch.long))
        )
        self.assertEqual(k.Uvec.shape, torch.Size([3]))

        k.Uvec.requires_grad_(False)
        k.Uvec.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        k.Uvec.requires_grad_(True)
        x1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        x2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.double)

        K = k.forward(x1, x2)
        Ktrue = torch.tensor(
            [[np.exp(-0.5 * 18), 1.0], [1.0, np.exp(-0.5 * 18)]], dtype=torch.double
        )
        self.assertTrue(torch.equal(K, Ktrue))

    def testALEBOGP(self):
        # First non-batch
        B = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.double
        )
        train_X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.double)
        train_Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
        train_Yvar = 0.1 * torch.ones(3, 1, dtype=torch.double)

        mll = get_map_model(
            B=B,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            restarts=1,
            init_state_dict=None,
        )
        m = mll.model
        m.eval()
        self.assertIsInstance(m, ALEBOGP)
        self.assertIsInstance(m.covar_module.base_kernel, ALEBOKernel)

        X = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.double)
        f = m(X)
        self.assertEqual(f.mean.shape, torch.Size([3]))
        self.assertEqual(f.variance.shape, torch.Size([3]))
        self.assertEqual(f.covariance_matrix.shape, torch.Size([3, 3]))

        # Batch
        Uvec_b = m.covar_module.base_kernel.Uvec.repeat(5, 1)
        mean_b = m.mean_module.constant.repeat(5, 1)
        output_scale_b = m.covar_module.raw_outputscale.repeat(5)
        m_b = get_batch_model(
            B=B,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            Uvec_batch=Uvec_b,
            mean_constant_batch=mean_b,
            output_scale_batch=output_scale_b,
        )

        self.assertEqual(m_b._aug_batch_shape, torch.Size([5]))
        f = m_b(X)
        self.assertEqual(f.mean.shape, torch.Size([3]))
        self.assertEqual(f.variance.shape, torch.Size([3]))
        self.assertEqual(f.covariance_matrix.shape, torch.Size([3, 3]))
        self.assertEqual(
            m_b.posterior(X).mvn.covariance_matrix.shape, torch.Size([3, 3])
        )

        # The whole process in get_fitted_model
        init_state_dict = m.state_dict()
        m_b2 = get_fitted_model(
            B=B,
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            restarts=1,
            nsamp=5,
            init_state_dict=init_state_dict,
        )
        self.assertEqual(m_b2._aug_batch_shape, torch.Size([5]))

        # Test extract_map_statedict
        map_sds = extract_map_statedict(m_b=m_b, num_outputs=1)
        self.assertEqual(len(map_sds), 1)
        self.assertEqual(len(map_sds[0]), 3)
        self.assertEqual(
            set(map_sds[0]),
            {
                "covar_module.base_kernel.Uvec",
                "covar_module.raw_outputscale",
                "mean_module.constant",
            },
        )
        self.assertEqual(
            map_sds[0]["covar_module.base_kernel.Uvec"].shape, torch.Size([3])
        )

        ml = ModelListGP(m_b, m_b2)
        map_sds = extract_map_statedict(m_b=ml, num_outputs=2)
        self.assertEqual(len(map_sds), 2)
        for i in range(2):
            self.assertEqual(len(map_sds[i]), 3)
            self.assertEqual(
                set(map_sds[i]),
                {
                    "covar_module.base_kernel.Uvec",
                    "covar_module.raw_outputscale",
                    "mean_module.constant",
                },
            )
            self.assertEqual(
                map_sds[i]["covar_module.base_kernel.Uvec"].shape, torch.Size([3])
            )

    def testAcq(self):
        B = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.double
        )
        train_X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.double)
        train_Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
        train_Yvar = 0.1 * torch.ones(3, 1, dtype=torch.double)
        m = ALEBOGP(B=B, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        m.eval()

        objective_weights = torch.tensor([1.0], dtype=torch.double)
        acq = ei_or_nei(
            model=m,
            objective_weights=objective_weights,
            outcome_constraints=None,
            X_observed=train_X,
            X_pending=None,
            q=1,
            noiseless=True,
        )
        self.assertIsInstance(acq, ExpectedImprovement)
        self.assertEqual(acq.best_f.item(), 3.0)

        objective_weights = torch.tensor([-1.0], dtype=torch.double)
        acq = ei_or_nei(
            model=m,
            objective_weights=objective_weights,
            outcome_constraints=None,
            X_observed=train_X,
            X_pending=None,
            q=1,
            noiseless=True,
        )
        self.assertEqual(acq.best_f.item(), 1.0)

        acq = ei_or_nei(
            model=m,
            objective_weights=objective_weights,
            outcome_constraints=None,
            X_observed=train_X,
            X_pending=None,
            q=1,
            noiseless=False,
        )
        self.assertIsInstance(acq, qNoisyExpectedImprovement)

        with mock.patch(
            "ax.models.torch.alebo.optimize_acqf", autospec=True
        ) as optim_mock:
            alebo_acqf_optimizer(
                acq_function=acq,
                bounds=None,
                n=1,
                inequality_constraints=5.0,
                fixed_features=None,
                rounding_func=None,
                raw_samples=100,
                num_restarts=5,
                B=B,
            )

        self.assertIsInstance(
            optim_mock.mock_calls[0][2]["acq_function"], qNoisyExpectedImprovement
        )
        self.assertEqual(optim_mock.mock_calls[0][2]["num_restarts"], 5)
        self.assertEqual(optim_mock.mock_calls[0][2]["inequality_constraints"], 5.0)
        X = optim_mock.mock_calls[0][2]["batch_initial_conditions"]
        self.assertEqual(X.shape, torch.Size([5, 1, 2]))
        # Make sure initialization is inside subspace
        Z = (B @ torch.pinverse(B) @ X[:, 0, :].t()).t()
        self.assertTrue(torch.allclose(Z, X[:, 0, :]))

    def testALEBO(self):
        B = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.double
        )
        train_X = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.double,
        )
        train_Y = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double)
        train_Yvar = 0.1 * torch.ones(3, 1, dtype=torch.double)

        m = ALEBO(B=B, laplace_nsamp=5, fit_restarts=1)
        self.assertTrue(torch.equal(B, m.B))
        self.assertEqual(m.laplace_nsamp, 5)
        self.assertEqual(m.fit_restarts, 1)
        self.assertEqual(m.refit_on_update, True)
        self.assertEqual(m.refit_on_cv, False)
        self.assertEqual(m.warm_start_refitting, False)

        # Test fit
        m.fit(
            Xs=[train_X, train_X],
            Ys=[train_Y, train_Y],
            Yvars=[train_Yvar, train_Yvar],
            bounds=[(-1, 1)] * 5,
            task_features=[],
            feature_names=[],
            metric_names=[],
            fidelity_features=[],
        )
        self.assertIsInstance(m.model, ModelListGP)
        self.assertTrue(torch.allclose(m.Xs[0], (B @ train_X.t()).t()))

        # Test predict
        f, cov = m.predict(X=B)
        self.assertEqual(f.shape, torch.Size([2, 2]))
        self.assertEqual(cov.shape, torch.Size([2, 2, 2]))

        # Test best point
        objective_weights = torch.tensor([1.0, 0.0], dtype=torch.double)
        with self.assertRaises(NotImplementedError):
            m.best_point(bounds=[(-1, 1)] * 5, objective_weights=objective_weights)

        # Test gen
        # With clipping
        with mock.patch(
            "ax.models.torch.alebo.optimize_acqf",
            autospec=True,
            return_value=(m.Xs[0], torch.tensor([])),
        ):
            Xopt, _, _, _ = m.gen(
                n=1,
                bounds=[(-1, 1)] * 5,
                objective_weights=torch.tensor([1.0, 0.0], dtype=torch.double),
            )

        self.assertFalse(torch.allclose(Xopt, train_X))
        self.assertTrue(Xopt.min() >= -1)
        self.assertTrue(Xopt.max() <= 1)
        # Without
        with mock.patch(
            "ax.models.torch.alebo.optimize_acqf",
            autospec=True,
            return_value=(torch.ones(1, 2, dtype=torch.double), torch.tensor([])),
        ):
            Xopt, _, _, _ = m.gen(
                n=1,
                bounds=[(-1, 1)] * 5,
                objective_weights=torch.tensor([1.0, 0.0], dtype=torch.double),
            )

        self.assertTrue(
            torch.allclose(
                Xopt, torch.tensor([[-0.2, -0.1, 0.0, 0.1, 0.2]], dtype=torch.double)
            )
        )

        # Test update
        train_X2 = torch.tensor(
            [
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.double,
        )
        m.update(
            Xs=[train_X, train_X2],
            Ys=[train_Y, train_Y],
            Yvars=[train_Yvar, train_Yvar],
        )
        self.assertTrue(torch.allclose(m.Xs[0], (B @ train_X.t()).t()))
        self.assertTrue(torch.allclose(m.Xs[1], (B @ train_X2.t()).t()))
        m.refit_on_update = False
        m.update(
            Xs=[train_X, train_X2],
            Ys=[train_Y, train_Y],
            Yvars=[train_Yvar, train_Yvar],
        )

        # Test get_and_fit with single meric
        gp = m.get_and_fit_model(
            Xs=[(B @ train_X.t()).t()], Ys=[train_Y], Yvars=[train_Yvar]
        )
        self.assertIsInstance(gp, ALEBOGP)

        # Test cross_validate
        f, cov = m.cross_validate(
            Xs_train=[train_X],
            Ys_train=[train_Y],
            Yvars_train=[train_Yvar],
            X_test=train_X2,
        )
        self.assertEqual(f.shape, torch.Size([3, 1]))
        self.assertEqual(cov.shape, torch.Size([3, 1, 1]))
        m.refit_on_cv = True
        f, cov = m.cross_validate(
            Xs_train=[train_X],
            Ys_train=[train_Y],
            Yvars_train=[train_Yvar],
            X_test=train_X2,
        )
        self.assertEqual(f.shape, torch.Size([3, 1]))
        self.assertEqual(cov.shape, torch.Size([3, 1, 1]))
