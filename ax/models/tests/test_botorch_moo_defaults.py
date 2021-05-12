#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    get_EHVI,
    pareto_frontier_evaluator,
    get_weighted_mc_objective_and_objective_thresholds,
    get_outcome_constraint_transforms,
)
from ax.utils.common.testutils import TestCase
from botorch.utils.sampling import manual_seed
from botorch.utils.testing import MockModel, MockPosterior

GET_ACQF_PATH = "ax.models.torch.botorch_moo_defaults.get_acquisition_function"
GET_CONSTRAINT_PATH = (
    "ax.models.torch.botorch_moo_defaults.get_outcome_constraint_transforms"
)
GET_OBJ_PATH = (
    "ax.models.torch.botorch_moo_defaults."
    "get_weighted_mc_objective_and_objective_thresholds"
)

FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_model"


def dummy_predict(model, X):
    # Add column to X that is a product of previous elements.
    mean = torch.cat([X, torch.prod(X, dim=1).reshape(-1, 1)], dim=1)
    cov = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[1])
    return mean, cov


class FrontierEvaluatorTest(TestCase):
    def setUp(self):
        self.X = torch.tensor(
            [[1.0, 0.0], [1.0, 1.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        )
        self.Y = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 3.0, 3.0],
                [2.0, 2.0, 4.0],
                [3.0, 1.0, 3.0],
            ]
        )
        self.Yvar = torch.zeros(5, 3)
        self.outcome_constraints = (
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[3.5]]),
        )
        self.objective_thresholds = torch.tensor([0.5, 1.5])
        self.objective_weights = torch.tensor([1.0, 1.0])
        bounds = [(0.0, 4.0), (0.0, 4.0)]
        self.model = MultiObjectiveBotorchModel(model_predictor=dummy_predict)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            self.model.fit(
                Xs=[self.X],
                Ys=[self.Y],
                Yvars=[self.Yvar],
                search_space_digest=SearchSpaceDigest(
                    feature_names=["x1", "x2"],
                    bounds=bounds,
                ),
                metric_names=["a", "b", "c"],
            )
            _mock_fit_model.assert_called_once()

    def test_pareto_frontier_raise_error_when_missing_data(self):
        with self.assertRaises(ValueError):
            pareto_frontier_evaluator(
                model=self.model,
                objective_thresholds=self.objective_thresholds,
                objective_weights=self.objective_weights,
                Yvar=self.Yvar,
            )

    def test_pareto_frontier_evaluator_raw(self):
        Yvar = torch.diag_embed(self.Yvar)
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=Yvar,
        )
        pred = self.Y[2:4]
        self.assertTrue(torch.allclose(Y, pred), f"{Y} does not match {pred}")
        expected_cov = Yvar[2:4]
        self.assertTrue(torch.allclose(expected_cov, cov))
        self.assertTrue(torch.equal(torch.arange(2, 4), indx))

        # Omit objective_thresholds
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            Y=self.Y,
            Yvar=Yvar,
        )
        pred = self.Y[2:]
        self.assertTrue(torch.allclose(Y, pred), f"{Y} does not match {pred}")
        expected_cov = Yvar[2:]
        self.assertTrue(torch.allclose(expected_cov, cov))
        self.assertTrue(torch.equal(torch.arange(2, 5), indx))

        # Change objective_weights so goal is to minimize b
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=torch.tensor([1.0, -1.0]),
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=Yvar,
        )
        pred = self.Y[[0, 4]]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )
        expected_cov = Yvar[[0, 4]]
        self.assertTrue(torch.allclose(expected_cov, cov))

        # test no points better than reference point
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=torch.full_like(self.objective_thresholds, 100.0),
            Y=self.Y,
            Yvar=Yvar,
        )
        self.assertTrue(torch.equal(Y, self.Y[:0]))
        self.assertTrue(torch.equal(cov, torch.zeros(0, 3, 3)))
        self.assertTrue(torch.equal(torch.tensor([], dtype=torch.long), indx))

    def test_pareto_frontier_evaluator_predict(self):
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            X=self.X,
        )
        pred = self.Y[2:4]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )
        self.assertTrue(torch.equal(torch.arange(2, 4), indx))

    def test_pareto_frontier_evaluator_with_outcome_constraints(self):
        Y, cov, indx = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=self.Yvar,
            outcome_constraints=self.outcome_constraints,
        )
        pred = self.Y[2, :]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )
        self.assertTrue(torch.equal(torch.tensor([2], dtype=torch.long), indx))


class BotorchMOODefaultsTest(TestCase):
    def test_get_EHVI_input_validation_errors(self):
        weights = torch.ones(2)
        objective_thresholds = torch.zeros(2)
        mm = MockModel(MockPosterior())
        with self.assertRaisesRegex(
            ValueError, "There are no feasible observed points."
        ):
            get_EHVI(
                model=mm,
                objective_weights=weights,
                objective_thresholds=objective_thresholds,
            )

    def test_get_weighted_mc_objective_and_objective_thresholds(self):
        objective_weights = torch.tensor([0.0, 1.0, 0.0, 1.0])
        objective_thresholds = torch.arange(4, dtype=torch.float)
        (
            weighted_obj,
            new_obj_thresholds,
        ) = get_weighted_mc_objective_and_objective_thresholds(
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
        )
        self.assertTrue(torch.equal(weighted_obj.weights, objective_weights[[1, 3]]))
        self.assertEqual(weighted_obj.outcomes.tolist(), [1, 3])
        self.assertTrue(torch.equal(new_obj_thresholds, objective_thresholds[[1, 3]]))

    def test_get_ehvi(self):
        weights = torch.tensor([0.0, 1.0, 1.0])
        X_observed = torch.rand(4, 3)
        X_pending = torch.rand(1, 3)
        constraints = (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([[10.0]]))
        Y = torch.rand(4, 3)
        mm = MockModel(MockPosterior(mean=Y))
        objective_thresholds = torch.arange(3, dtype=torch.float)
        obj_and_obj_t = get_weighted_mc_objective_and_objective_thresholds(
            objective_weights=weights,
            objective_thresholds=objective_thresholds,
        )
        (weighted_obj, new_obj_thresholds) = obj_and_obj_t
        cons_tfs = get_outcome_constraint_transforms(constraints)
        with manual_seed(0):
            seed = torch.randint(1, 10000, (1,)).item()
        with ExitStack() as es:
            mock_get_acqf = es.enter_context(mock.patch(GET_ACQF_PATH))
            es.enter_context(mock.patch(GET_CONSTRAINT_PATH, return_value=cons_tfs))
            es.enter_context(mock.patch(GET_OBJ_PATH, return_value=obj_and_obj_t))
            es.enter_context(manual_seed(0))
            get_EHVI(
                model=mm,
                objective_weights=weights,
                outcome_constraints=constraints,
                objective_thresholds=objective_thresholds,
                X_observed=X_observed,
                X_pending=X_pending,
            )
            mock_get_acqf.assert_called_once_with(
                acquisition_function_name="qEHVI",
                model=mm,
                objective=weighted_obj,
                X_observed=X_observed,
                X_pending=X_pending,
                constraints=cons_tfs,
                mc_samples=128,
                qmc=True,
                alpha=0.0,
                seed=seed,
                ref_point=new_obj_thresholds.tolist(),
                Y=Y,
            )
