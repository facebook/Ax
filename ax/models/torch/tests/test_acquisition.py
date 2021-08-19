#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from botorch.acquisition.input_constructors import (
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    get_acqf_input_constructor,
    _register_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData
from botorch.utils.testing import MockPosterior


ACQUISITION_PATH = Acquisition.__module__
CURRENT_PATH = __name__
SURROGATE_PATH = Surrogate.__module__


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
class DummyACQFClass:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, **kwargs: Any) -> None:
        pass


class AcquisitionTest(TestCase):
    def setUp(self):
        qNEI_input_constructor = get_acqf_input_constructor(qNoisyExpectedImprovement)
        self.mock_input_constructor = mock.MagicMock(
            qNEI_input_constructor, side_effect=qNEI_input_constructor
        )
        # Adding wrapping here to be able to count calls and inspect arguments.
        _register_acqf_input_constructor(
            acqf_cls=DummyACQFClass,
            input_constructor=self.mock_input_constructor,
        )
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.Y = torch.tensor([[3.0], [4.0]])
        self.Yvar = torch.tensor([[0.0], [2.0]])
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
        self.botorch_acqf_class = DummyACQFClass
        self.objective_weights = torch.tensor([1.0])
        self.objective_thresholds = None
        self.pending_observations = [torch.tensor([[1.0, 3.0, 4.0]])]
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {"best_f": 0.0}
        self.acquisition = Acquisition(
            botorch_acqf_class=self.botorch_acqf_class,
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
        self.inequality_constraints = [
            (torch.tensor([0, 1]), torch.tensor([-1.0, 1.0]), 1)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}

    def tearDown(self):
        # Avoid polluting the registry for other tests.
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyACQFClass)

    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model",
        return_value=SubsetModelData(None, torch.ones(1), None, None, None),
    )
    @mock.patch(f"{ACQUISITION_PATH}.get_botorch_objective")
    @mock.patch(
        f"{CURRENT_PATH}.Acquisition.compute_model_dependencies",
        return_value={"current_value": 1.2},
    )
    @mock.patch(
        f"{DummyACQFClass.__module__}.DummyACQFClass.__init__", return_value=None
    )
    def test_init(
        self,
        mock_botorch_acqf_class,
        mock_compute_model_deps,
        mock_get_objective,
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
        mock_get_objective.return_value = botorch_objective
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
        mock_get_objective.reset_mock()
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
        # Check `get_botorch_objective` kwargs
        mock_get_objective.assert_called_once()
        _, ckwargs = mock_get_objective.call_args
        self.assertIs(ckwargs["model"], self.acquisition.surrogate.model)
        self.assertIs(ckwargs["objective_weights"], self.objective_weights)
        self.assertIs(ckwargs["outcome_constraints"], self.outcome_constraints)
        self.assertTrue(torch.equal(ckwargs["X_observed"], self.X[:1]))
        self.assertFalse(ckwargs["use_scalarized_objective"])
        # Check final `acqf` creation
        model_deps = {Keys.CURRENT_VALUE: 1.2}
        self.mock_input_constructor.assert_called_once()
        mock_botorch_acqf_class.assert_called_once()
        _, ckwargs = self.mock_input_constructor.call_args
        self.assertIs(ckwargs["model"], self.acquisition.surrogate.model)
        self.assertIs(ckwargs["objective"], botorch_objective)
        self.assertTrue(torch.equal(ckwargs["X_pending"], self.pending_observations[0]))
        for k, v in chain(self.options.items(), model_deps.items()):
            self.assertEqual(ckwargs[k], v)

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf")
    def test_optimize(self, mock_optimize_acqf):
        self.acquisition.optimize(
            n=3,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf.assert_called_with(
            acq_function=self.acquisition.acqf,
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
            dtype=self.acquisition.dtype,
            device=self.acquisition.device,
        ).transpose(0, 1)
        self.assertTrue(
            torch.equal(mock_optimize_acqf.call_args[1]["bounds"], expected_bounds)
        )

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf_discrete")
    def test_optimize_discrete(self, mock_optimize_acqf_discrete):
        tkwargs = {
            "dtype": self.acquisition.dtype,
            "device": self.acquisition.device,
        }
        ssd1 = SearchSpaceDigest(
            feature_names=["a"],
            bounds=[(0, 2)],
            categorical_features=[0],
            discrete_choices={0: [0, 1, 2]},
        )
        # check fixed_feature index validation
        with self.assertRaisesRegex(ValueError, "Invalid fixed_feature index"):
            self.acquisition.optimize(
                n=3,
                search_space_digest=ssd1,
                inequality_constraints=self.inequality_constraints,
                fixed_features=self.fixed_features,
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )
        # check this works without any fixed_feature specified
        self.acquisition.optimize(
            n=3,
            search_space_digest=ssd1,
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_discrete.assert_called_with(
            acq_function=self.acquisition.acqf,
            q=3,
            choices=mock.ANY,
            **self.optimizer_options,
        )
        # can't use assert_called_with on choices due to ambiguous bool comparison
        expected_choices = torch.tensor([[0], [1], [2]], **tkwargs)
        self.assertTrue(
            torch.equal(
                mock_optimize_acqf_discrete.call_args[1]["choices"], expected_choices
            )
        )
        # check with fixed feature
        ssd2 = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 2), (0, 1)],
            categorical_features=[0],
            discrete_choices={0: [0, 1, 2]},
        )
        self.acquisition.optimize(
            n=3,
            search_space_digest=ssd2,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_discrete.assert_called_with(
            acq_function=self.acquisition.acqf,
            q=3,
            choices=mock.ANY,
            **self.optimizer_options,
        )
        # can't use assert_called_with on choices due to ambiguous bool comparison
        expected_choices = torch.tensor([[0, 2.0], [1, 2.0], [2, 2.0]], **tkwargs)
        self.assertTrue(
            torch.equal(
                mock_optimize_acqf_discrete.call_args[1]["choices"], expected_choices
            )
        )

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf_mixed")
    def test_optimize_mixed(self, mock_optimize_acqf_mixed):
        tkwargs = {
            "dtype": self.acquisition.dtype,
            "device": self.acquisition.device,
        }
        ssd = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 1), (0, 2)],
            categorical_features=[1],
            discrete_choices={1: [0, 1, 2]},
        )
        self.acquisition.optimize(
            n=3,
            search_space_digest=ssd,
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_mixed.assert_called_with(
            acq_function=self.acquisition.acqf,
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
        self.acquisition.best_point(
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
        f"{DummyACQFClass.__module__}.DummyACQFClass.__call__", return_value=None
    )
    def test_evaluate(self, mock_call):
        self.acquisition.evaluate(X=self.X)
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
        moo_objective_weights = torch.tensor(
            [-1.0, -1.0, 0.0],
        )
        moo_objective_thresholds = torch.tensor(
            [0.5, 1.5, float("nan")],
        )
        self.surrogate.construct(
            training_data=moo_training_data,
        )
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        outcome_constraints = (
            torch.tensor(
                [[1.0, 0.0, 0.0]],
            ),
            torch.tensor(
                [[10.0]],
            ),
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
                    acquisition.objective_thresholds[:2], torch.tensor([9.9, 3.3])
                )
            )
            self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
