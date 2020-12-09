#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import mock

import torch
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    get_EHVI,
    pareto_frontier_evaluator,
    get_default_partitioning_alpha,
)
from ax.utils.common.testutils import TestCase


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
                bounds=bounds,
                task_features=[],
                feature_names=["x1", "x2"],
                metric_names=["a", "b", "c"],
                fidelity_features=[],
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
        Y, cov = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=self.Yvar,
        )
        pred = self.Y[2:4]
        self.assertTrue(torch.allclose(Y, pred), f"{Y} does not match {pred}")

        # Omit objective_thresholds
        Y, cov = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            Y=self.Y,
            Yvar=self.Yvar,
        )
        pred = self.Y[2:]
        self.assertTrue(torch.allclose(Y, pred), f"{Y} does not match {pred}")

        # Change objective_weights so goal is to minimize b
        Y, cov = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=torch.tensor([1.0, -1.0]),
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=self.Yvar,
        )
        pred = self.Y[[0, 4]]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )

    def test_pareto_frontier_evaluator_predict(self):
        Y, cov = pareto_frontier_evaluator(
            model=self.model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            X=self.X,
        )
        pred = self.Y[2:4]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )

    def test_pareto_frontier_evaluator_with_outcome_constraints(self):
        Y, cov = pareto_frontier_evaluator(
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


class BotorchMOODefaultsTest(TestCase):
    def test_get_NEI_with_chebyshev_and_missing_Ys_error(self):
        model = MultiObjectiveBotorchModel()
        x = torch.zeros(2, 2)
        weights = torch.ones(2)
        with self.assertRaisesRegex(
            ValueError, "Chebyshev Scalarization requires Ys argument"
        ):
            get_NEI(
                model=model,
                X_observed=x,
                objective_weights=weights,
                chebyshev_scalarization=True,
            )

    def test_get_EHVI_input_validation_errors(self):
        model = MultiObjectiveBotorchModel()
        x = torch.zeros(2, 2)
        weights = torch.ones(2)
        objective_thresholds = torch.zeros(2)
        with self.assertRaisesRegex(
            ValueError, "There are no feasible observed points."
        ):
            get_EHVI(
                model=model,
                objective_weights=weights,
                objective_thresholds=objective_thresholds,
            )
        with self.assertRaisesRegex(
            ValueError, "Expected Hypervolume Improvement requires Ys argument"
        ):
            get_EHVI(
                model=model,
                X_observed=x,
                objective_weights=weights,
                objective_thresholds=objective_thresholds,
            )

    def test_get_default_partitioning_alpha(self):
        self.assertEqual(0.0, get_default_partitioning_alpha(2))
        self.assertEqual(1e-5, get_default_partitioning_alpha(3))
        self.assertEqual(1e-4, get_default_partitioning_alpha(4))
        with warnings.catch_warnings(record=True) as ws:
            self.assertEqual(0.1, get_default_partitioning_alpha(7))
        self.assertEqual(len(ws), 1)
