#!/usr/bin/env python3

import numpy as np
from ax.models.numpy.gpy import (
    GPyGP,
    _get_GP,
    _mvn_sample,
    _parse_gen_inputs,
    _update_pending_observations,
    _validate_tasks,
)
from ax.utils.common.testutils import TestCase


class GPyModelTest(TestCase):
    def setUp(self):
        self.Xs = [
            np.array([[2.0, 3.0, 0, 1], [4.0, 5.0, 1, 0]]),
            np.array([[1.0, 2.0, 0, 0]]),
        ]
        self.Ys = [np.array([[2.0], [4.0]]), np.array([[1.0]])]
        self.Yvars = [np.array([[0.1], [0.2]]), np.array([[0.1]])]
        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0, 1), (0, 1)]

        # A 1-d model for testing gen
        Xs1 = [np.array([0.2, 0.4, 0.6, 0.8])[:, None] for i in range(3)]
        Ys1 = [
            np.sin(Xs1[0] * 10.0),
            1 + np.cos(Xs1[1] * 10.0),
            0.2 * np.sin(Xs1[2] * 10.0),
        ]
        Yvars1 = [0.1 * np.ones_like(Ys1[i]) for i in range(3)]
        self.m = GPyGP(map_fit_restarts=2)
        self.m.fit(
            Xs=Xs1,
            Ys=Ys1,
            Yvars=Yvars1,
            bounds=[(0.0, 1.0)],
            task_features=[],
            feature_names=["x"],
        )

        # Assume the rounding_func is simply to round to integer
        self.rounding_func = np.round

    def testGPyModel(self):
        # Init
        m = GPyGP(map_fit_restarts=2, refit_on_update=False)
        self.assertEqual(m.map_fit_restarts, 2)
        # Fit
        task_features = []
        feature_names = ["x1", "x2", "t1", "t2"]
        m.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=task_features,
            feature_names=feature_names,
        )
        self.assertEqual(len(m.params), 2)
        self.assertEqual(len(m.params[0]), 5)
        self.assertEqual(len(m.models), 2)
        self.assertEqual(m.task_features, [])
        # Predict
        X = np.array([[1.0, 2.0, 0, 0], [2.0, 3.0, 0, 0], [4.0, 5.0, 0, 0]])
        f, cov = m.predict(X)
        self.assertEqual(f.shape, (3, 2))
        self.assertEqual(cov.shape, (3, 2, 2))
        # Cross validate
        f, cov = m.cross_validate(self.Xs, self.Ys, self.Yvars, X)
        self.assertEqual(f.shape, (3, 2))
        self.assertEqual(cov.shape, (3, 2, 2))
        # Update
        m.update(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.assertEqual(self.Xs[0].shape, (4, 4))
        self.assertEqual(self.Ys[0].shape, (4, 1))
        self.assertEqual(self.Yvars[0].shape, (4, 1))
        self.assertEqual(len(m.models), 2)

        # No multiprocessing
        m2 = GPyGP(map_fit_restarts=2, use_multiprocessing=False)
        m2.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=task_features,
            feature_names=feature_names,
        )
        self.assertEqual(len(m2.params[0]), 5)
        self.assertEqual(len(m2.models), 2)

    def testGetGP(self):
        mt1 = _get_GP(
            X=self.Xs[0],
            Y=self.Ys[0],
            Yvar=self.Yvars[0],
            task_features=[2],
            map_fit_restarts=2,
            primary_task_rank=1,
        )
        self.assertEqual(mt1.ICM.B.W.shape, (2, 1))
        mt1 = _get_GP(
            X=self.Xs[0],
            Y=self.Ys[0],
            Yvar=self.Yvars[0],
            task_features=[2],
            map_fit_restarts=2,
        )
        self.assertEqual(mt1.ICM.B.W.shape, (2, 2))
        mt2 = _get_GP(
            X=self.Xs[0],
            Y=self.Ys[0],
            Yvar=self.Yvars[0],
            task_features=[2, 3],
            map_fit_restarts=2,
        )
        self.assertEqual(mt2.add.ICM.B.W.shape, (2, 2))
        self.assertEqual(mt2.add.ICM.B.active_dims[0], 2)
        self.assertEqual(mt2.add.ICM.Q.W.shape, (2, 1))
        self.assertEqual(mt2.add.ICM.Q.active_dims[0], 3)
        self.assertTrue(np.array_equal(mt2.add.ICM.Mat52.active_dims, np.array([0, 1])))
        with self.assertRaises(ValueError):
            _get_GP(
                X=self.Xs[0],
                Y=self.Ys[0],
                Yvar=self.Yvars[0],
                task_features=[1, 2, 3],
                map_fit_restarts=2,
            )

    def testValidateTasks(self):
        feature_names = ["x1", "x2", "x3"]
        _validate_tasks(
            task_features=[],
            feature_names=feature_names,
            primary_task_name=None,
            secondary_task_name=None,
        )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[],
                feature_names=feature_names,
                primary_task_name="x1",
                secondary_task_name=None,
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0],
                feature_names=feature_names,
                primary_task_name=None,
                secondary_task_name="x1",
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0],
                feature_names=feature_names,
                primary_task_name="x2",
                secondary_task_name=None,
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0, 1],
                feature_names=feature_names,
                primary_task_name=None,
                secondary_task_name=None,
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0, 1],
                feature_names=feature_names,
                primary_task_name=None,
                secondary_task_name=None,
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0, 1],
                feature_names=feature_names,
                primary_task_name="x1",
                secondary_task_name="x3",
            )
        with self.assertRaises(ValueError):
            _validate_tasks(
                task_features=[0, 1, 2],
                feature_names=feature_names,
                primary_task_name=None,
                secondary_task_name=None,
            )
        task_features = _validate_tasks(
            task_features=[0, 1],
            feature_names=feature_names,
            primary_task_name="x1",
            secondary_task_name="x2",
        )
        self.assertEqual(task_features, [0, 1])
        task_features = _validate_tasks(
            task_features=[0, 1],
            feature_names=feature_names,
            primary_task_name="x2",
            secondary_task_name="x1",
        )
        self.assertEqual(task_features, [1, 0])

    def testMvnSample(self):
        mu = np.array([[1.0], [2.0]])
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])

        Ys = _mvn_sample(mu=mu, cov=cov, nsamp=10, qmc=True)
        self.assertEqual(Ys.shape, (10, 2))
        Ys = _mvn_sample(mu=mu, cov=cov, nsamp=10, qmc=False)
        self.assertEqual(Ys.shape, (10, 2))

    def testParseGenInputs(self):
        objective_weights = np.array([-0.1, 0.0, 0.0])
        outcome_constraints = (
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 0.5]]),
            np.array([[-1.0], [0.0]]),
        )

        obj_idx, obj_sign, con_list = _parse_gen_inputs(
            objective_weights=objective_weights, outcome_constraints=outcome_constraints
        )
        self.assertEqual(obj_idx, 0)
        self.assertEqual(obj_sign, -1.0)
        self.assertEqual(con_list, [(1, -1.0, -1.0), (2, 0.5, 0.0)])

        obj_idx, obj_sign, con_list = _parse_gen_inputs(
            objective_weights=objective_weights, outcome_constraints=None
        )
        self.assertEqual(con_list, [])

        with self.assertRaises(ValueError):
            _parse_gen_inputs(
                objective_weights=np.array([1.0, 2.0, 3.0]),
                outcome_constraints=outcome_constraints,
            )
        with self.assertRaises(ValueError):
            _parse_gen_inputs(
                objective_weights=None, outcome_constraints=outcome_constraints
            )
        with self.assertRaises(ValueError):
            _parse_gen_inputs(
                objective_weights=objective_weights,
                outcome_constraints=(np.array([[0.0, -1.0, 1.0]]), np.array([[-1.0]])),
            )

    def testGetFantasyModels(self):
        # Test the usual case
        fantasy_models, cand_X_array = self.m._get_fantasy_models(
            obj_idx=0,
            con_list=[(1, 0.5, -1)],
            pending_observations=[np.array([])] * 3,
            fixed_features={},
            nsamp=2,
            qmc=True,
        )
        self.assertEqual(list(fantasy_models.keys()), [0, 1])
        self.assertEqual(len(fantasy_models[0]), 2)
        self.assertEqual(len(fantasy_models[1]), 2)
        self.assertTrue(np.array_equal(np.sort(cand_X_array, axis=0), self.m.Xs[0]))
        for i in range(2):
            for j in range(2):
                self.assertTrue(np.array_equal(fantasy_models[i][j].X, self.m.Xs[i]))
                self.assertFalse(np.array_equal(fantasy_models[i][j].Y, self.m.Ys[j]))
                self.assertTrue(
                    np.array_equal(
                        fantasy_models[i][j].het_Gauss.variance.values, np.zeros((4, 1))
                    )
                )

        # Handle pending observations
        pending_observations = [np.array([[8.0]]), np.array([[]]), np.array([[]])]
        fantasy_models, cand_X_array = self.m._get_fantasy_models(
            obj_idx=0,
            con_list=[(1, 0.5, -1)],
            pending_observations=pending_observations,
            fixed_features={},
            nsamp=1,
            qmc=True,
        )
        Xtrue = np.vstack((self.m.Xs[0], np.array([[8.0]])))
        self.assertTrue(np.array_equal(fantasy_models[0][0].X, Xtrue))
        self.assertTrue(np.array_equal(fantasy_models[1][0].X, self.m.Xs[1]))
        self.assertTrue(np.array_equal(np.sort(cand_X_array, axis=0), self.m.Xs[0]))

        # Handle fixed features
        fantasy_models, cand_X_array = self.m._get_fantasy_models(
            obj_idx=0,
            con_list=[(1, 0.5, -1)],
            pending_observations=pending_observations,
            fixed_features={0: 8.0},
            nsamp=2,
            qmc=True,
        )
        Yvar_true = np.array([0.1, 0.1, 0.1, 0.1, 0.0])[:, None]
        self.assertEqual(len(fantasy_models[0]), 2)
        self.assertTrue(np.array_equal(fantasy_models[0][0].X, Xtrue))
        self.assertTrue(np.array_equal(fantasy_models[1][0].X, self.m.Xs[1]))
        # Only sample the pending observation
        self.assertTrue(
            np.array_equal(fantasy_models[0][0].het_Gauss.variance.values, Yvar_true)
        )
        self.assertTrue(
            np.array_equal(
                fantasy_models[1][0].het_Gauss.variance.values, 0.1 * np.ones((4, 1))
            )
        )
        self.assertEqual(cand_X_array.size, 0)

        # Fixed features, that satisfy observation
        fantasy_models, cand_X_array = self.m._get_fantasy_models(
            obj_idx=0,
            con_list=[(1, 0.5, -1)],
            pending_observations=pending_observations,
            fixed_features={0: 0.2},
            nsamp=2,
            qmc=True,
        )
        self.assertTrue(np.array_equal(fantasy_models[0][0].X, Xtrue))
        self.assertTrue(np.array_equal(fantasy_models[1][0].X, self.m.Xs[1]))
        Yvar_true0 = np.array([0.0, 0.1, 0.1, 0.1, 0.0])
        Yvar_true1 = np.array([0.0, 0.1, 0.1, 0.1])
        self.assertTrue(
            np.array_equal(
                fantasy_models[0][0].het_Gauss.variance.values, Yvar_true0[:, None]
            )
        )
        self.assertTrue(
            np.array_equal(
                fantasy_models[1][0].het_Gauss.variance.values, Yvar_true1[:, None]
            )
        )
        self.assertTrue(np.array_equal(cand_X_array, np.array([[0.2]])))
        self.assertTrue(
            np.array_equal(
                fantasy_models[0][0].optimizer_array, self.m.models[0].optimizer_array
            )
        )

        # Nothing to fantasize
        fantasy_models, cand_X_array = self.m._get_fantasy_models(
            obj_idx=0,
            con_list=[(1, 0.5, -1)],
            pending_observations=[np.array([])] * 3,
            fixed_features={0: 20.0},
            nsamp=2,
            qmc=True,
        )
        self.assertEqual(len(fantasy_models[0]), 1)

    def testUpdatePendingObservations(self):
        po = [np.array([]), np.array([[2.0, 3.0]])]
        po = _update_pending_observations(po, np.array([4.0, 5.0]))
        self.assertTrue(np.array_equal(po[0], np.array([[4.0, 5.0]])))
        self.assertTrue(np.array_equal(po[1], np.array([[2.0, 3.0], [4.0, 5.0]])))

    def testGen(self):
        self.m.use_multiprocessing = True
        X, w = self.m.gen(
            n=3,
            bounds=[(0.1, 1.0)],
            objective_weights=np.array([1.0, 0.0, 0.0]),
            outcome_constraints=None,
            linear_constraints=(np.array([[-1.0]]), np.array([[-0.5]])),
            fixed_features=None,
            pending_observations=None,
            model_gen_options={"nsamp": 2, "qmc": True, "nopt": 5, "init_samples": 100},
            rounding_func=self.rounding_func,
        )
        self.assertEqual(X.shape, (3, 1))
        # the linear constraint is >= 0.5, so the rounding should push it up to 1.0
        self.assertTrue(X.min() == 1.0)
        self.assertTrue(np.array_equal(w, np.ones(3)))

        self.m.use_multiprocessing = False
        X, w = self.m.gen(
            n=3,
            bounds=[(0.1, 1.0)],
            objective_weights=np.array([1.0, 0.0, 0.0]),
            outcome_constraints=(np.array([[-1.0]]), np.array([[-1.0]])),
            linear_constraints=(np.array([[1.0]]), np.array([[0.5]])),
            fixed_features=None,
            pending_observations=[np.array([[0.1]]), np.array([]), np.array([])],
            model_gen_options={"nsamp": 2, "qmc": True, "nopt": 5, "init_samples": 100},
        )
        self.assertTrue(X.max() <= 0.5)
        self.assertTrue(X.min() >= 0.1)

        X, w = self.m.gen(
            n=3,
            bounds=[(0.1, 1.0)],
            objective_weights=np.array([1.0, 0.0, 0.0]),
            outcome_constraints=None,
            linear_constraints=(np.array([[1.0]]), np.array([[0.5]])),
            fixed_features={0: 0.2},
            pending_observations=None,
            model_gen_options={"nsamp": 2, "qmc": True, "nopt": 5, "init_samples": 100},
        )
        self.assertTrue(np.array_equal(X, 0.2 * np.ones((3, 1))))

        X, w = self.m.gen(
            n=0,
            bounds=[(0.1, 1.0)],
            objective_weights=np.array([1.0, 0.0, 0.0]),
            outcome_constraints=None,
            linear_constraints=None,
            fixed_features=None,
            pending_observations=None,
            model_gen_options=None,
        )

    def testBestPoint(self):
        xbest = self.m.best_point(
            bounds=[(0.1, 1.0)], objective_weights=np.array([1.0, 0.0, 0.0])
        )
        self.assertEqual(xbest.shape, (1,))
