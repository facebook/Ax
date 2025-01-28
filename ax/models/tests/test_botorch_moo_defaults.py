#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import ExitStack
from typing import Any, cast
from unittest import mock
from warnings import catch_warnings, simplefilter

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_defaults import NO_OBSERVED_POINTS_MESSAGE
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    get_outcome_constraint_transforms,
    get_qLogEHVI,
    get_qLogNEHVI,
    get_weighted_mc_objective_and_objective_thresholds,
    infer_objective_thresholds,
    pareto_frontier_evaluator,
)
from ax.models.torch.utils import _get_X_pending_and_observed
from ax.models.torch_base import TorchModel
from ax.utils.common.random import with_rng_seed
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize_context_manager
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.testing import MockModel, MockPosterior
from gpytorch.utils.warnings import NumericalWarning
from torch._tensor import Tensor


MOO_DEFAULTS_PATH: str = "ax.models.torch.botorch_moo_defaults"
GET_ACQF_PATH: str = MOO_DEFAULTS_PATH + ".get_acquisition_function"
GET_CONSTRAINT_PATH: str = MOO_DEFAULTS_PATH + ".get_outcome_constraint_transforms"
GET_OBJ_PATH: str = (
    MOO_DEFAULTS_PATH + ".get_weighted_mc_objective_and_objective_thresholds"
)
FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_mll"


def _fit_model(
    model: TorchModel, X: torch.Tensor, Y: torch.Tensor, Yvar: torch.Tensor
) -> None:
    bounds = [(0.0, 4.0), (0.0, 4.0)]
    datasets = [
        SupervisedDataset(
            X=X,
            Y=Y,
            Yvar=Yvar,
            feature_names=["x1", "x2"],
            outcome_names=["a", "b", "c"],
        )
    ]
    search_space_digest = SearchSpaceDigest(feature_names=["x1", "x2"], bounds=bounds)
    with mock_botorch_optimize_context_manager(), catch_warnings():
        simplefilter(action="ignore", category=NumericalWarning)
        model.fit(datasets=datasets, search_space_digest=search_space_digest)


class FrontierEvaluatorTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.objective_thresholds = torch.tensor([0.5, 1.5])
        self.objective_weights = torch.tensor([1.0, 1.0])

    def test_pareto_frontier_raise_error_when_missing_data(self) -> None:
        with self.assertRaises(ValueError):
            pareto_frontier_evaluator(
                model=MultiObjectiveBotorchModel(),
                objective_thresholds=self.objective_thresholds,
                objective_weights=self.objective_weights,
                Yvar=self.Yvar,
            )

    def test_pareto_frontier_evaluator_raw(self) -> None:
        model = BoTorchModel()
        _fit_model(model=model, X=self.X, Y=self.Y, Yvar=self.Yvar)
        Yvar = torch.diag_embed(self.Yvar)
        Y, cov, indx = pareto_frontier_evaluator(
            model=model,
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
            model=model,
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
            model=model,
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
            model=model,
            objective_weights=self.objective_weights,
            objective_thresholds=torch.full_like(self.objective_thresholds, 100.0),
            Y=self.Y,
            Yvar=Yvar,
        )
        self.assertTrue(torch.equal(Y, self.Y[:0]))
        self.assertTrue(torch.equal(cov, torch.zeros(0, 3, 3)))
        self.assertTrue(torch.equal(torch.tensor([], dtype=torch.long), indx))

    def test_pareto_frontier_evaluator_predict(self) -> None:
        def dummy_predict(
            model: MultiObjectiveBotorchModel,
            X: Tensor,
            use_posterior_predictive: bool = False,
        ) -> tuple[Tensor, Tensor]:
            # Add column to X that is a product of previous elements.
            mean = torch.cat([X, torch.prod(X, dim=1).reshape(-1, 1)], dim=1)
            cov = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[1])
            return mean, cov

        # pyre-fixme: Incompatible parameter type [6]: In call
        # `MultiObjectiveBotorchModel.__init__`, for argument `model_predictor`,
        # expected `typing.Callable[[Model, Tensor, bool], Tuple[Tensor,
        # Tensor]]` but got named arguments
        model = MultiObjectiveBotorchModel(model_predictor=dummy_predict)
        _fit_model(model=model, X=self.X, Y=self.Y, Yvar=self.Yvar)

        Y, _, indx = pareto_frontier_evaluator(
            model=model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            X=self.X,
        )
        pred = self.Y[2:4]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )
        self.assertTrue(torch.equal(torch.arange(2, 4), indx))

    def test_pareto_frontier_evaluator_with_outcome_constraints(self) -> None:
        model = MultiObjectiveBotorchModel()
        Y, _, indx = pareto_frontier_evaluator(
            model=model,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            Y=self.Y,
            Yvar=self.Yvar,
            outcome_constraints=(
                torch.tensor([[0.0, 0.0, 1.0]]),
                torch.tensor([[3.5]]),
            ),
        )
        pred = self.Y[2, :]
        self.assertTrue(
            torch.allclose(Y, pred), f"actual {Y} does not match pred {pred}"
        )
        self.assertTrue(torch.equal(torch.tensor([2], dtype=torch.long), indx))

    def test_pareto_frontier_evaluator_with_nan(self) -> None:
        Y = torch.cat([self.Y, torch.zeros(5, 1)], dim=-1)
        Yvar = torch.zeros(5, 4, 4)
        weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
        outcome_constraints = (
            torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            torch.tensor([[3.5]]),
        )
        # Evaluate without NaNs as a baseline.
        _, _, idx = pareto_frontier_evaluator(
            model=None, objective_weights=weights, Y=Y, Yvar=Yvar
        )
        self.assertEqual(idx.tolist(), [2, 3, 4])
        # Set an element of idx 2 to NaN. Should be removed.
        Y[2, 1] = float("nan")
        _, _, idx = pareto_frontier_evaluator(
            model=None, objective_weights=weights, Y=Y, Yvar=Yvar
        )
        self.assertEqual(idx.tolist(), [3, 4])
        # Set the unused constraint element of idx 3 to NaN. No effect.
        Y[3, 2] = float("nan")
        _, _, idx = pareto_frontier_evaluator(
            model=None, objective_weights=weights, Y=Y, Yvar=Yvar
        )
        self.assertEqual(idx.tolist(), [3, 4])
        # Add constraint, 3 should be removed.
        _, _, idx = pareto_frontier_evaluator(
            model=None,
            objective_weights=weights,
            Y=Y,
            Yvar=Yvar,
            outcome_constraints=outcome_constraints,
        )
        self.assertEqual(idx.tolist(), [4])
        # Set unused index of 4 to NaN. No effect.
        Y[4, 3] = float("nan")
        _, _, idx = pareto_frontier_evaluator(
            model=None,
            objective_weights=weights,
            Y=Y,
            Yvar=Yvar,
            outcome_constraints=outcome_constraints,
        )
        self.assertEqual(idx.tolist(), [4])


class BotorchMOODefaultsTest(TestCase):
    def test_get_qLogEHVI_input_validation_errors(self) -> None:
        weights = torch.ones(2)
        objective_thresholds = torch.zeros(2)
        # Note: this is a real BoTorch `Model` with a real `Posterior`, not a
        # `unittest.mock.Mock`
        mm = MockModel(posterior=MockPosterior())
        with self.assertRaisesRegex(ValueError, NO_OBSERVED_POINTS_MESSAGE):
            get_qLogEHVI(
                model=mm,
                objective_weights=weights,
                objective_thresholds=objective_thresholds,
            )

    def test_get_weighted_mc_objective_and_objective_thresholds(self) -> None:
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

    def test_get_qLogNEHVI_input_validation_errors(self) -> None:
        weights = torch.ones(2)
        objective_thresholds = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, NO_OBSERVED_POINTS_MESSAGE):
            get_qLogNEHVI(
                # pyre-fixme[6] In call `get_qLogNEHVI`, for argument `model`,
                # expected `Model` but got `None`.
                model=None,
                objective_weights=weights,
                objective_thresholds=objective_thresholds,
            )

    @mock.patch(  # pyre-ignore
        "ax.models.torch.botorch_moo_defaults._check_posterior_type",
        wraps=lambda y: y,
    )
    def test_get_qLogEHVI(self, _) -> None:
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
        with with_rng_seed(0):
            seed = torch.randint(1, 10000, (1,)).item()
        with ExitStack() as es:
            mock_get_acqf = es.enter_context(mock.patch(GET_ACQF_PATH))
            es.enter_context(
                mock.patch(MOO_DEFAULTS_PATH + ".assert_is_instance", wraps=cast)
            )
            es.enter_context(mock.patch(GET_CONSTRAINT_PATH, return_value=cons_tfs))
            es.enter_context(mock.patch(GET_OBJ_PATH, return_value=obj_and_obj_t))
            es.enter_context(with_rng_seed(0))
            get_qLogEHVI(
                model=mm,
                objective_weights=weights,
                outcome_constraints=constraints,
                objective_thresholds=objective_thresholds,
                X_observed=X_observed,
                X_pending=X_pending,
            )
            mock_get_acqf.assert_called_once_with(
                acquisition_function_name="qLogEHVI",
                model=mm,
                objective=weighted_obj,
                X_observed=X_observed,
                X_pending=X_pending,
                constraints=cons_tfs,
                mc_samples=128,
                alpha=0.0,
                seed=seed,
                ref_point=new_obj_thresholds.tolist(),
                Y=Y,
            )

    # test infer objective thresholds alone
    @mock.patch(  # pyre-ignore
        "ax.models.torch.botorch_moo_defaults._check_posterior_type",
        wraps=lambda y: y,
    )
    def test_infer_objective_thresholds(self, _, cuda: bool = False) -> None:
        # TODO: refactor this test into smaller test cases
        for dtype in (torch.float, torch.double):
            tkwargs: dict[str, Any] = {
                "device": torch.device("cuda") if cuda else torch.device("cpu"),
                "dtype": dtype,
            }
            posterior = MockPosterior(
                mean=torch.tensor(
                    [
                        [11.0, 2.0],
                        [9.0, 3.0],
                    ],
                    **tkwargs,
                )
            )
            Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)]
            bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
            outcome_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]], **tkwargs),
                torch.tensor([[10.0]], **tkwargs),
            )
            linear_constraints = (
                torch.tensor([1.0, 0.0, 0.0], **tkwargs),
                torch.tensor([2.0], **tkwargs),
            )
            objective_weights = torch.tensor([-1.0, -1.0, 0.0], **tkwargs)
            with ExitStack() as es:
                _mock_get_X_pending_and_observed = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults."
                        "_get_X_pending_and_observed",
                        wraps=_get_X_pending_and_observed,
                    )
                )
                _mock_infer_reference_point = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults.infer_reference_point",
                        wraps=infer_reference_point,
                    )
                )
                model = SingleTaskGP(train_X=Xs[0], train_Y=torch.rand(2, 3, **tkwargs))
                es.enter_context(
                    mock.patch.object(model, "posterior", return_value=posterior)
                )

                # test passing Xs
                obj_thresholds = infer_objective_thresholds(
                    model,
                    bounds=bounds,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    fixed_features={},
                    linear_constraints=linear_constraints,
                    Xs=Xs + Xs,
                )
            _mock_get_X_pending_and_observed.assert_called_once()
            ckwargs = _mock_get_X_pending_and_observed.call_args[1]
            actual_Xs = ckwargs["Xs"]
            for X in actual_Xs:
                self.assertTrue(torch.equal(X, Xs[0]))
            self.assertEqual(ckwargs["bounds"], bounds)
            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], objective_weights)
            )
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], outcome_constraints[0]))
            self.assertTrue(torch.equal(oc[1], outcome_constraints[1]))
            self.assertEqual(ckwargs["fixed_features"], {})
            lc = ckwargs["linear_constraints"]
            self.assertTrue(torch.equal(lc[0], linear_constraints[0]))
            self.assertTrue(torch.equal(lc[1], linear_constraints[1]))
            _mock_infer_reference_point.assert_called_once()
            ckwargs = _mock_infer_reference_point.call_args[1]
            self.assertEqual(ckwargs["scale"], 0.1)
            self.assertTrue(
                torch.equal(
                    ckwargs["pareto_Y"],
                    torch.tensor([[-9.0, -3.0]], **tkwargs),
                )
            )
            self.assertTrue(
                torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3], **tkwargs))
            )
            self.assertTrue(np.isnan(obj_thresholds[2].item()))

            with ExitStack() as es:
                _mock_get_X_pending_and_observed = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults."
                        "_get_X_pending_and_observed",
                        wraps=_get_X_pending_and_observed,
                    )
                )
                _mock_infer_reference_point = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults.infer_reference_point",
                        wraps=infer_reference_point,
                    )
                )
                es.enter_context(
                    mock.patch.object(model, "posterior", return_value=posterior)
                )

                # test passing X_observed
                obj_thresholds = infer_objective_thresholds(
                    model,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    X_observed=Xs[0],
                )
            _mock_get_X_pending_and_observed.assert_not_called()
            self.assertTrue(
                torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3], **tkwargs))
            )
            self.assertTrue(np.isnan(obj_thresholds[2].item()))

            # test that value error is raised if bounds are not supplied
            with self.assertRaises(ValueError):
                infer_objective_thresholds(
                    model,
                    objective_weights=objective_weights,
                    Xs=Xs + Xs,
                )
            # test that value error is raised if Xs are not supplied
            with self.assertRaises(ValueError):
                infer_objective_thresholds(
                    model,
                    bounds=bounds,
                    objective_weights=objective_weights,
                )
            # test subset_model without subset_idcs
            with mock.patch.object(model, "posterior", return_value=posterior):
                obj_thresholds = infer_objective_thresholds(
                    model,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    X_observed=Xs[0],
                )
            self.assertTrue(
                torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3], **tkwargs))
            )
            self.assertTrue(np.isnan(obj_thresholds[2].item()))
            # test passing subset_idcs
            subset_idcs = torch.tensor(
                [0, 1], dtype=torch.long, device=tkwargs["device"]
            )

            with mock.patch.object(model, "posterior", return_value=posterior):
                obj_thresholds = infer_objective_thresholds(
                    model,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    X_observed=Xs[0],
                    subset_idcs=subset_idcs,
                )
            self.assertTrue(
                torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3], **tkwargs))
            )
            self.assertTrue(np.isnan(obj_thresholds[2].item()))
            # test without subsetting (e.g. if there are
            # 3 metrics for 2 objectives + 1 outcome constraint)
            outcome_constraints = (
                torch.tensor([[0.0, 0.0, 1.0]], **tkwargs),
                torch.tensor([[5.0]], **tkwargs),
            )
            with ExitStack() as es:
                _mock_get_X_pending_and_observed = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults."
                        "_get_X_pending_and_observed",
                        wraps=_get_X_pending_and_observed,
                    )
                )
                _mock_infer_reference_point = es.enter_context(
                    mock.patch(
                        "ax.models.torch.botorch_moo_defaults.infer_reference_point",
                        wraps=infer_reference_point,
                    )
                )
                es.enter_context(
                    mock.patch.object(
                        model,
                        "posterior",
                        return_value=MockPosterior(
                            mean=torch.tensor(
                                [
                                    [11.0, 2.0, 6.0],
                                    [9.0, 3.0, 4.0],
                                ],
                                **tkwargs,
                            )
                        ),
                    )
                )
                # test passing Xs
                obj_thresholds = infer_objective_thresholds(
                    model,
                    bounds=bounds,
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                    fixed_features={},
                    linear_constraints=linear_constraints,
                    Xs=Xs + Xs + Xs,
                )
            self.assertTrue(
                torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3], **tkwargs))
            )
            self.assertTrue(np.isnan(obj_thresholds[2].item()))

    def test_infer_objective_thresholds_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_infer_objective_thresholds(cuda=True)
