#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import ExitStack
from itertools import chain
from typing import Any
from unittest import mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.utils import SubsetModelData
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import (
    _register_acqf_input_constructor,
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    get_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData
from botorch.utils.testing import MockPosterior
from torch import Tensor


ACQUISITION_PATH = Acquisition.__module__
CURRENT_PATH = __name__
SURROGATE_PATH = Surrogate.__module__


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
class DummyAcquisitionFunction(AcquisitionFunction):
    X_pending = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model=None)

    def __call__(self, X, **kwargs: Any) -> None:
        return X.sum(dim=-1)

    def set_X_pending(self, X: Tensor, **kwargs: Any) -> None:
        self.X_pending = X

    def forward(self, X: torch.Tensor) -> None:
        pass


class AcquisitionTest(TestCase):
    def setUp(self):
        qNEI_input_constructor = get_acqf_input_constructor(qNoisyExpectedImprovement)
        self.mock_input_constructor = mock.MagicMock(
            qNEI_input_constructor, side_effect=qNEI_input_constructor
        )
        # Adding wrapping here to be able to count calls and inspect arguments.
        _register_acqf_input_constructor(
            acqf_cls=DummyAcquisitionFunction,
            input_constructor=self.mock_input_constructor,
        )
        tkwargs = {"dtype": torch.double}
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        self.Y = torch.tensor([[3.0], [4.0]], **tkwargs)
        self.Yvar = torch.tensor([[0.0], [2.0]], **tkwargs)
        self.training_data = TrainingData.from_block_design(
            X=self.X, Y=self.Y, Yvar=self.Yvar
        )
        self.fidelity_features = [2]
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_fidelities={2: 1.0},
        )
        self.botorch_acqf_class = DummyAcquisitionFunction
        self.objective_weights = torch.tensor([1.0])
        self.objective_thresholds = None
        self.pending_observations = [torch.tensor([[1.0, 3.0, 4.0]], **tkwargs)]
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {"best_f": 0.0}
        self.inequality_constraints = [
            (torch.tensor([0, 1], **tkwargs), torch.tensor([-1.0, 1.0], **tkwargs), 1)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.tkwargs = tkwargs

    def get_acquisition_function(self, fixed_features=None):
        return Acquisition(
            botorch_acqf_class=self.botorch_acqf_class,
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=fixed_features or {},
            options=self.options,
        )

    def tearDown(self):
        # Avoid polluting the registry for other tests.
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyAcquisitionFunction)

    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model",
        return_value=SubsetModelData(None, torch.ones(1), None, None, None),
    )
    @mock.patch(f"{ACQUISITION_PATH}.get_botorch_objective_and_transform")
    @mock.patch(
        f"{CURRENT_PATH}.Acquisition.compute_model_dependencies",
        return_value={"current_value": 1.2},
    )
    @mock.patch(
        f"{DummyAcquisitionFunction.__module__}.DummyAcquisitionFunction.__init__",
        return_value=None,
    )
    def test_init(
        self,
        mock_botorch_acqf_class,
        mock_compute_model_deps,
        mock_get_objective_and_transform,
        mock_subset_model,
        mock_get_X,
    ):
        with self.assertRaisesRegex(TypeError, ".* missing .* 'botorch_acqf_class'"):
            Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
            )

        botorch_objective = LinearMCObjective(weights=torch.tensor([1.0]))
        mock_get_objective_and_transform.return_value = (botorch_objective, None)
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        acquisition = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
            objective_thresholds=self.objective_thresholds,
        )

        # Check `_get_X_pending_and_observed` kwargs
        mock_get_X.assert_called_with(
            Xs=[self.training_data.X],
            pending_observations=self.pending_observations,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=self.search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        # Call `subset_model` only when needed
        mock_subset_model.assert_called_with(
            model=acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )
        mock_subset_model.reset_mock()
        mock_get_objective_and_transform.reset_mock()
        self.mock_input_constructor.reset_mock()
        mock_botorch_acqf_class.reset_mock()
        self.options[Keys.SUBSET_MODEL] = False
        acquisition = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
        mock_subset_model.assert_not_called()
        # Check `get_botorch_objective_and_transform` kwargs
        mock_get_objective_and_transform.assert_called_once()
        _, ckwargs = mock_get_objective_and_transform.call_args
        self.assertIs(ckwargs["model"], acquisition.surrogate.model)
        self.assertIs(ckwargs["objective_weights"], self.objective_weights)
        self.assertIs(ckwargs["outcome_constraints"], self.outcome_constraints)
        self.assertTrue(torch.equal(ckwargs["X_observed"], self.X[:1]))
        # Check final `acqf` creation
        model_deps = {Keys.CURRENT_VALUE: 1.2}
        self.mock_input_constructor.assert_called_once()
        mock_botorch_acqf_class.assert_called_once()
        _, ckwargs = self.mock_input_constructor.call_args
        self.assertIs(ckwargs["model"], acquisition.surrogate.model)
        self.assertIs(ckwargs["objective"], botorch_objective)
        self.assertTrue(torch.equal(ckwargs["X_pending"], self.pending_observations[0]))
        for k, v in chain(self.options.items(), model_deps.items()):
            self.assertEqual(ckwargs[k], v)

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf")
    def test_optimize(self, mock_optimize_acqf):
        acquisition = self.get_acquisition_function(fixed_features=self.fixed_features)
        acquisition.optimize(
            n=3,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            q=3,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(
            self.search_space_digest.bounds,
            dtype=acquisition.dtype,
            device=acquisition.device,
        ).transpose(0, 1)
        self.assertTrue(
            torch.equal(mock_optimize_acqf.call_args[1]["bounds"], expected_bounds)
        )

    def test_optimize_discrete(self):
        ssd1 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(1, 2, 3), (2, 3, 4)],
            categorical_features=[0, 1, 2],
            discrete_choices={0: [1, 2], 1: [2, 3], 2: [3, 4]},
        )
        # check fixed_feature index validation
        with self.assertRaisesRegex(ValueError, "Invalid fixed_feature index"):
            acquisition = self.get_acquisition_function()
            acquisition.optimize(
                n=3,
                search_space_digest=ssd1,
                fixed_features={3: 2.0},
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )
        # check this works without any fixed_feature specified
        # 2 candidates have acqf value 8, but [1, 3, 4] is pending and thus should
        # not be selected. [2, 3, 4] is the best point, but has already been picked
        acquisition = self.get_acquisition_function()
        X_selected, _ = acquisition.optimize(
            n=2,
            search_space_digest=ssd1,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        expected = torch.tensor([[2, 2, 4], [2, 3, 3]]).to(self.X)
        self.assertTrue(X_selected.shape == (2, 3))
        self.assertTrue(
            all((x.unsqueeze(0) == expected).all(dim=-1).any() for x in X_selected)
        )
        # check with fixed feature
        # Since parameter 1 is fixed to 2, the best 3 candidates are
        # [4, 2, 4], [3, 2, 4], [4, 2, 3]
        ssd2 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 0, 0), (4, 4, 4)],
            categorical_features=[0, 1, 2],
            discrete_choices={k: [0, 1, 2, 3, 4] for k in range(3)},
        )
        X_selected, _ = acquisition.optimize(
            n=3,
            search_space_digest=ssd2,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        expected = torch.tensor([[4, 2, 4], [3, 2, 4], [4, 2, 3]]).to(self.X)
        self.assertTrue(X_selected.shape == (3, 3))
        self.assertTrue(
            all((x.unsqueeze(0) == expected).all(dim=-1).any() for x in X_selected)
        )
        # check with a constraint that -1 * x[0]  -1 * x[1] >= 0 which should make
        # [0, 0, 4] the best candidate.
        X_selected, _ = acquisition.optimize(
            n=1,
            search_space_digest=ssd2,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
            inequality_constraints=[
                [torch.tensor([0, 1], dtype=torch.int64), -torch.ones(2), 0]
            ],
        )
        expected = torch.tensor([[0, 0, 4]]).to(self.X)
        self.assertTrue(torch.equal(expected, X_selected))
        # Same thing but use two constraints instead
        X_selected, _ = acquisition.optimize(
            n=1,
            search_space_digest=ssd2,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
            inequality_constraints=[
                [torch.tensor([0], dtype=torch.int64), -torch.ones(1), 0],
                [torch.tensor([1], dtype=torch.int64), -torch.ones(1), 0],
            ],
        )
        expected = torch.tensor([[0, 0, 4]]).to(self.X)
        self.assertTrue(torch.equal(expected, X_selected))

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf_discrete_local_search")
    def test_optimize_acqf_discrete_local_search(
        self, mock_optimize_acqf_discrete_local_search
    ):
        tkwargs = {"dtype": self.X.dtype, "device": self.X.device}
        ssd = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 0, 0), (1, 1, 1)],
            categorical_features=[0, 1, 2],
            discrete_choices={  # 30 * 60 * 90 > 100,000
                k: np.linspace(0, 1, 30 * (k + 1)).tolist() for k in range(3)
            },
        )
        acquisition = self.get_acquisition_function()
        acquisition.optimize(
            n=3,
            search_space_digest=ssd,
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_discrete_local_search.assert_called_once()
        args, kwargs = mock_optimize_acqf_discrete_local_search.call_args
        self.assertEqual(kwargs["acq_function"], acquisition.acqf)
        self.assertEqual(kwargs["q"], 3)
        self.assertEqual(kwargs["inequality_constraints"], self.inequality_constraints)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])
        self.assertTrue(
            all(
                torch.allclose(torch.linspace(0, 1, 30 * (k + 1), **tkwargs), c)
                for k, c in enumerate(kwargs["discrete_choices"])
            )
        )
        X_avoid_true = torch.cat((self.X, self.pending_observations[0]), dim=0)
        self.assertEqual(kwargs["X_avoid"].shape, X_avoid_true.shape)
        self.assertTrue(  # The order of the rows may not match
            all((X_avoid_true == x).all(dim=-1).any().item() for x in kwargs["X_avoid"])
        )

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf_mixed")
    def test_optimize_mixed(self, mock_optimize_acqf_mixed):
        tkwargs = {"dtype": self.X.dtype, "device": self.X.device}
        ssd = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 1), (0, 2)],
            categorical_features=[1],
            discrete_choices={1: [0, 1, 2]},
        )
        acquisition = self.get_acquisition_function()
        acquisition.optimize(
            n=3,
            search_space_digest=ssd,
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_mixed.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            q=3,
            fixed_features_list=[{1: 0}, {1: 1}, {1: 2}],
            inequality_constraints=self.inequality_constraints,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(ssd.bounds, **tkwargs).transpose(0, 1)
        self.assertTrue(
            torch.equal(
                mock_optimize_acqf_mixed.call_args[1]["bounds"], expected_bounds
            )
        )

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.best_in_sample_point")
    def test_best_point(self, mock_best_point):
        acquisition = self.get_acquisition_function(self.fixed_features)
        acquisition.best_point(
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
        mock_best_point.assert_called_with(
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )

    @mock.patch(
        f"{DummyAcquisitionFunction.__module__}.DummyAcquisitionFunction.__call__",
        return_value=None,
    )
    def test_evaluate(self, mock_call):
        acquisition = self.get_acquisition_function()
        acquisition.evaluate(X=self.X)
        mock_call.assert_called_with(X=self.X)

    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    def test_init_moo(
        self,
        mock_get_X,
    ):
        moo_training_data = TrainingData(
            Xs=[self.X] * 3,
            Ys=[self.Y] * 3,
            Yvars=[self.Yvar] * 3,
        )
        moo_objective_weights = torch.tensor([-1.0, -1.0, 0.0], **self.tkwargs)
        moo_objective_thresholds = torch.tensor(
            [0.5, 1.5, float("nan")], **self.tkwargs
        )
        self.surrogate.construct(
            training_data=moo_training_data,
        )
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        outcome_constraints = (
            torch.tensor([[1.0, 0.0, 0.0]], **self.tkwargs),
            torch.tensor([[10.0]], **self.tkwargs),
        )

        acquisition = Acquisition(
            surrogate=self.surrogate,
            botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
            search_space_digest=self.search_space_digest,
            objective_weights=moo_objective_weights,
            pending_observations=self.pending_observations,
            outcome_constraints=outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
            objective_thresholds=moo_objective_thresholds,
        )
        self.assertTrue(
            torch.equal(
                moo_objective_thresholds[:2], acquisition.objective_thresholds[:2]
            )
        )
        self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
        # test inferred objective_thresholds
        with ExitStack() as es:
            preds = torch.tensor(
                [
                    [11.0, 2.0],
                    [9.0, 3.0],
                ],
                **self.tkwargs,
            )
            es.enter_context(
                mock.patch.object(
                    self.surrogate.model,
                    "posterior",
                    return_value=MockPosterior(
                        mean=preds,
                        samples=preds,
                    ),
                )
            )
            acquisition = Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                objective_weights=moo_objective_weights,
                botorch_acqf_class=self.botorch_acqf_class,
                pending_observations=self.pending_observations,
                outcome_constraints=outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )
            self.assertTrue(
                torch.equal(
                    acquisition.objective_thresholds[:2],
                    torch.tensor([9.9, 3.3], **self.tkwargs),
                )
            )
            self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
