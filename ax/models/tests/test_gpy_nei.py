#!/usr/bin/env python3

from unittest import mock

import numpy as np
from ax.models.numpy.gpy import GPyGP
from ax.models.numpy.gpy_nei import (
    compute_best_feasible_value,
    get_infeasible_cost,
    get_initial_points,
    nan_cb,
    nei_and_grad,
    nei_vectorized,
    objective_and_grad,
    optimize_from_x0,
)
from ax.utils.common.testutils import TestCase
from scipy.optimize import approx_fprime


class GPyModelTest(TestCase):
    def setUp(self):
        pass

    def testGetInfeasibleCost(self):
        obj_model = mock.MagicMock()
        f = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
        f_var = f ** 2
        obj_model._raw_predict.return_value = (f, f_var)

        M = get_infeasible_cost(obj_model=obj_model, obj_sign=1.0, X=np.array([[1.0]]))
        self.assertEqual(M, 20.0)
        M = get_infeasible_cost(obj_model=obj_model, obj_sign=-1.0, X=np.array([[1.0]]))
        self.assertEqual(M, 28.0)
        f = f + 100
        obj_model._raw_predict.return_value = (f, f_var)
        M = get_infeasible_cost(obj_model=obj_model, obj_sign=1.0, X=np.array([[1.0]]))
        self.assertEqual(M, 0.0)

    @mock.patch("ax.models.numpy.gpy_nei.nei_vectorized", autospec=True)
    @mock.patch("ax.models.numpy.gpy_nei.SobolEngine", autospec=True)
    def test_get_initial_points(self, sobol_mock, nei_mock):
        draw_mock = mock.MagicMock()
        draw_mock.draw.return_value = np.array(
            [
                [0.0, 0.0],  # feas.
                [0.2, 0.0],  # feas.
                [0.4, 0.2],  # feas.
                [0.6, 0.2],  # feas.
                [0.8, 0.4],  # infeas
                [0.8, 0.4],  # infeas
                [0.6, 0.6],  # infeas
                [0.4, 0.6],  # feas.
                [0.2, 0.8],  # feas.
            ]
        )
        sobol_mock.return_value = draw_mock

        def nei_mock_side_effect(
            X, fantasy_models, obj_idx, obj_sign, con_list, f_best, M
        ):
            return np.arange(X.shape[0])

        nei_mock.side_effect = nei_mock_side_effect

        X0 = get_initial_points(
            nopt=5,
            init_samples=1000,
            bounds=[(1.0, 2.0), (10.0, 20.0)],
            linear_constraints=(np.array([[10.0, 1.0]]), np.array([[30.0]])),
            fixed_features={},
            fantasy_models=None,
            obj_idx=0,
            obj_sign=1.0,
            con_list=None,
            f_best=np.array([0.0]),
            M=0.0,
        )
        Xtrue = np.array(
            [[1.2, 10.0], [1.4, 12.0], [1.6, 12.0], [1.4, 16.0], [1.2, 18.0]]
        )
        self.assertTrue(np.array_equal(X0, Xtrue))

        X0 = get_initial_points(
            nopt=4,
            init_samples=1000,
            bounds=[(1.0, 2.0), (10.0, 20.0)],
            linear_constraints=(np.array([[10.0, 1.0]]), np.array([[30.0]])),
            fixed_features={},
            fantasy_models=None,
            obj_idx=0,
            obj_sign=1.0,
            con_list=None,
            f_best=np.array([0.0]),
            M=0.0,
        )
        Xtrue_set = {tuple(x) for x in Xtrue}
        for x in X0:
            self.assertTrue(tuple(x) in Xtrue_set)

        X0 = get_initial_points(
            nopt=6,
            init_samples=1000,
            bounds=[(1.0, 2.0), (10.0, 20.0)],
            linear_constraints=(np.array([[10.0, 1.0]]), np.array([[30.0]])),
            fixed_features={},
            fantasy_models=None,
            obj_idx=0,
            obj_sign=1.0,
            con_list=None,
            f_best=np.array([0.0]),
            M=0.0,
        )
        Xtrue = np.array(
            [
                [1.0, 10.0],
                [1.2, 10.0],
                [1.4, 12.0],
                [1.6, 12.0],
                [1.4, 16.0],
                [1.2, 18.0],
            ]
        )
        self.assertTrue(np.array_equal(np.sort(X0), Xtrue))

        X0 = get_initial_points(
            nopt=3,
            init_samples=1000,
            bounds=[(1.0, 2.0), (10.0, 20.0)],
            linear_constraints=(np.array([[10.0, 1.0]]), np.array([[30.0]])),
            fixed_features={0: 1.5},
            fantasy_models=None,
            obj_idx=0,
            obj_sign=1.0,
            con_list=None,
            f_best=np.array([0.0]),
            M=0.0,
        )
        self.assertTrue(np.array_equal(X0[:, 0], 1.5 * np.ones(3)))

        with self.assertRaises(Exception):
            X0 = get_initial_points(
                nopt=8,
                init_samples=1000,
                bounds=[(1.0, 2.0), (10.0, 20.0)],
                linear_constraints=(np.array([[10.0, 1.0]]), np.array([[30.0]])),
                fixed_features={},
                fantasy_models=None,
                obj_idx=0,
                obj_sign=1.0,
                con_list=None,
                f_best=np.array([0.0]),
                M=0.0,
            )

    def testComputeBestFeasibleValue(self):
        f1 = np.array([1.0, 2.0, 3.0, 4.0])[:, None]
        f2 = 10 * f1

        m1 = mock.MagicMock()
        m1._raw_predict.return_value = (f1,)
        m2 = mock.MagicMock()
        m2._raw_predict.return_value = (f2,)

        fantasy_models = {0: [m1, m2], 1: [m1, m2]}

        f_best = compute_best_feasible_value(
            cand_X_array=np.array([[]]),
            fantasy_models=fantasy_models,
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
        )
        self.assertTrue(np.array_equal(f_best, np.array([-np.Inf, -np.Inf])))

        cand_X_array = np.array([[1], [2], [3], [4]])
        f_best = compute_best_feasible_value(
            cand_X_array=cand_X_array,
            fantasy_models=fantasy_models,
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
        )
        self.assertTrue(np.array_equal(f_best, np.array([4.0, 40.0])))

        f_best = compute_best_feasible_value(
            cand_X_array=cand_X_array,
            fantasy_models=fantasy_models,
            obj_idx=0,
            obj_sign=-1.0,
            con_list=[],
        )
        self.assertTrue(np.array_equal(f_best, np.array([-1.0, -10.0])))

        f_best = compute_best_feasible_value(
            cand_X_array=cand_X_array,
            fantasy_models=fantasy_models,
            obj_idx=0,
            obj_sign=1.0,
            con_list=[(1, 1, 2)],
        )
        self.assertTrue(np.array_equal(f_best, np.array([2.0, -np.Inf])))

    def testNEI(self):
        # A 1-d model for testing gen
        Xs1 = [np.array([0.2, 0.4, 0.6, 0.8])[:, None] for i in range(3)]
        Ys1 = [
            np.sin(Xs1[0] * 10.0),
            1 + np.cos(Xs1[1] * 10.0),
            0.2 * np.sin(Xs1[2] * 10.0),
        ]
        Yvars1 = [0.1 * np.ones_like(Ys1[i]) for i in range(3)]
        m = GPyGP(map_fit_restarts=2)
        m.fit(
            Xs=Xs1,
            Ys=Ys1,
            Yvars=Yvars1,
            bounds=[(0.0, 1.0)],
            task_features=[],
            feature_names=["x"],
        )
        f_best = np.array([-1.0, -np.Inf])
        M = 10.0
        x0 = np.array([0.05])
        obj_idx = 0
        obj_sign = 1.0
        con_list = [(1, 0.5, -1)]

        fantasy_models, cand_X_array = m._get_fantasy_models(
            obj_idx=obj_idx,
            con_list=con_list,
            pending_observations=[np.array([])] * 3,
            fixed_features={},
            nsamp=2,
            qmc=True,
        )

        ei1 = nei_vectorized(
            X=np.array([x0]),
            fantasy_models=fantasy_models,
            obj_idx=obj_idx,
            obj_sign=obj_sign,
            con_list=con_list,
            f_best=f_best,
            M=M,
        )[0]

        ei2 = nei_and_grad(
            x=x0,
            fantasy_models=fantasy_models,
            obj_idx=obj_idx,
            obj_sign=obj_sign,
            con_list=con_list,
            f_best=f_best,
            M=M,
        )[0]
        self.assertAlmostEqual(ei1, ei2)

        def testf(x):
            return nei_and_grad(
                x, fantasy_models, obj_idx, obj_sign, con_list, f_best, M
            )[0]

        def testdf(x):
            return nei_and_grad(
                x, fantasy_models, obj_idx, obj_sign, con_list, f_best, M
            )[1]

        dfx0 = approx_fprime(x0, testf, epsilon=1e-9)
        self.assertTrue(np.abs(dfx0 - testdf(x0)) < 1e-6)

    @mock.patch("ax.models.numpy.gpy_nei.minimize", side_effect=StopIteration)
    @mock.patch(
        "ax.models.numpy.gpy_nei.objective_and_grad", autospec=True, return_value=(1, 2)
    )
    def testOptimizeFromX0(self, obj_grad_mock, minimize_mock):
        x0 = np.array([1.0, 2.0, 3.0])
        x, fun = optimize_from_x0(
            x0=x0,
            bounds=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
            fixed_features={},
            fantasy_models={},
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
            linear_constraints=None,
            f_best=np.array([]),
            M=0.0,
        )
        self.assertTrue(np.array_equal(x, x0))
        self.assertEqual(minimize_mock.mock_calls[0][2]["constraints"], ())
        self.assertTrue(np.array_equal(minimize_mock.mock_calls[0][2]["x0"], x0))

        A = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
        b = np.array([[5.0], [6.0]])
        x, fun = optimize_from_x0(
            x0=x0,
            bounds=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
            fixed_features={1: 4.0},
            fantasy_models={},
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
            linear_constraints=(A, b),
            f_best=np.array([]),
            M=0.0,
        )
        x0_opt = np.array([1.0, 3.0])
        self.assertEqual(minimize_mock.mock_calls[1][2]["constraints"]["type"], "ineq")
        self.assertTrue(np.array_equal(minimize_mock.mock_calls[1][2]["x0"], x0_opt))
        self.assertEqual(
            minimize_mock.mock_calls[1][2]["bounds"], [(0.0, 1.0), (0.0, 3.0)]
        )
        self.assertEqual(minimize_mock.mock_calls[1][2]["args"][1], {1: 4.0})
        self.assertTrue(
            np.array_equal(
                minimize_mock.mock_calls[1][2]["constraints"]["fun"](x0_opt),
                np.array([-4.0, -18.0]),
            )
        )
        self.assertTrue(
            np.array_equal(
                minimize_mock.mock_calls[1][2]["constraints"]["jac"](x0_opt),
                -np.array([[1.0, 0.0], [0.0, 4.0]]),
            )
        )
        self.assertEqual(minimize_mock.mock_calls[1][2]["args"][0], [0, 2])

        x, fun = optimize_from_x0(
            x0=x0,
            bounds=[(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)],
            fixed_features={0: 4.0, 1: 5.0, 2: 6.0},
            fantasy_models={},
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
            linear_constraints=(A, b),
            f_best=np.array([]),
            M=0.0,
        )
        self.assertTrue(np.array_equal(x, np.array([4.0, 5.0, 6.0])))
        self.assertEqual(fun, 1)

    @mock.patch(
        "ax.models.numpy.gpy_nei.nei_and_grad",
        autospec=True,
        return_value=(5, np.array([1.0, 2.0, 3.0])),
    )
    def testObjectiveAndGrad(self, nei_mock):
        f, grad = objective_and_grad(
            x=np.array([1.0, 3.0]),
            tunable_slice=[0, 2],
            fixed_features={1: 2.0},
            fantasy_models={},
            obj_idx=0,
            obj_sign=1.0,
            con_list=[],
            f_best=0.0,
            M=0.0,
        )
        self.assertEqual(f, -5)
        self.assertTrue(np.array_equal(grad, np.array([-1.0, -3.0])))
        np.array_equal(nei_mock.mock_calls[0][2]["x"], np.array([1.0, 2.0, 3.0]))

    def testNanCb(self):
        nan_cb(np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(StopIteration):
            nan_cb(np.array([4.0, np.nan]))
