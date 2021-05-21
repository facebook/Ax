#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from unittest.mock import patch, MagicMock

import ax.models.torch.botorch_modular.acquisition as acquisition
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.moo_acquisition import MOOAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.input_constructors import (
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    get_acqf_input_constructor,
    _register_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils import get_outcome_constraint_transforms
from botorch.utils.containers import TrainingData
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)


ACQUISITION_PATH = Acquisition.__module__
MOO_ACQUISITION_PATH = MOOAcquisition.__module__
CURRENT_PATH = __name__
SURROGATE_PATH = Surrogate.__module__


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
# Subclassing `qEHVI` to pass instance checks.
class DummyACQFClassMOO(qExpectedHypervolumeImprovement):
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, **kwargs: Any) -> None:
        pass


class MOOAcquisitionTest(TestCase):
    def setUp(self):
        qEHVI_input_constructor = get_acqf_input_constructor(
            qExpectedHypervolumeImprovement
        )
        self.mock_input_constructor = MagicMock(
            qEHVI_input_constructor, side_effect=qEHVI_input_constructor
        )
        _register_acqf_input_constructor(
            acqf_cls=DummyACQFClassMOO,
            input_constructor=self.mock_input_constructor,
        )
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.Y = torch.tensor([[3.0, 4.0, 2.0], [4.0, 3.0, 1.0]])
        self.Yvar = torch.tensor([[0.0, 2.0, 1.0], [2.0, 0.0, 1.0]])
        self.training_data = TrainingData.from_block_design(
            X=self.X, Y=self.Y, Yvar=self.Yvar
        )
        self.fidelity_features = [2]
        self.surrogate.construct(training_data=self.training_data)
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_fidelities={2: 1.0},
        )
        self.botorch_acqf_class = DummyACQFClassMOO
        self.objective_weights = torch.tensor([1.0, -1.0, 0.0])
        self.objective_thresholds = torch.tensor([2.0, 1.0, float("nan")])
        self.pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]]),
            torch.tensor([[1.0, 3.0, 4.0]]),
            torch.tensor([[1.0, 3.0, 4.0]]),
        ]
        self.outcome_constraints = (
            torch.tensor([[1.0, 0.5, 0.5]]),
            torch.tensor([[0.5]]),
        )
        self.con_tfs = get_outcome_constraint_transforms(self.outcome_constraints)
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {}
        self.acquisition = MOOAcquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            botorch_acqf_class=self.botorch_acqf_class,
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
        ACQF_INPUT_CONSTRUCTOR_REGISTRY[DummyACQFClassMOO] = get_acqf_input_constructor(
            qExpectedHypervolumeImprovement
        )

    def tearDown(self):
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyACQFClassMOO)

    @patch(f"{MOO_ACQUISITION_PATH}.get_outcome_constraint_transforms")
    @patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @patch(
        f"{ACQUISITION_PATH}.subset_model",
        wraps=acquisition.subset_model,
    )
    @patch(f"{CURRENT_PATH}.MOOAcquisition._get_botorch_objective")
    @patch(
        f"{DummyACQFClassMOO.__module__}.DummyACQFClassMOO.__init__", return_value=None
    )
    def test_init(
        self,
        mock_botorch_acqf_class,
        mock_get_objective,
        mock_subset_model,
        mock_get_X,
        mock_get_constraints,
    ):
        botorch_objective = WeightedMCMultiOutputObjective(
            weights=self.objective_weights[:2], outcomes=[0, 1]
        )
        mock_get_objective.return_value = botorch_objective
        mock_get_constraints.return_value = self.con_tfs
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        acquisition = MOOAcquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )

        # Check `_get_X_pending_and_observed` kwargs
        mock_get_X.assert_called_with(
            Xs=[self.training_data.X, self.training_data.X, self.training_data.X],
            pending_observations=self.pending_observations,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=self.search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        # Call `subset_model` only when needed
        mock_subset_model.assert_called_with(
            acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )
        mock_subset_model.reset_mock()
        mock_botorch_acqf_class.reset_mock()
        self.options[Keys.SUBSET_MODEL] = False
        acquisition = MOOAcquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
        mock_subset_model.assert_not_called()
        # Check final `acqf` creation
        mock_botorch_acqf_class.assert_called_once()
        mock_botorch_acqf_class.assert_called_once()
        _, ckwargs = mock_botorch_acqf_class.call_args
        self.assertIs(ckwargs["model"], self.acquisition.surrogate.model)
        self.assertIs(ckwargs["objective"], botorch_objective)
        self.assertTrue(torch.equal(ckwargs["X_pending"], self.pending_observations[0]))
        self.assertTrue(
            torch.equal(
                ckwargs["ref_point"],
                (self.objective_thresholds[:2] * self.objective_weights[:2]),
            )
        )
        self.assertIsInstance(ckwargs["partitioning"], BoxDecomposition)

        # qNoisyExpectedImprovement not supported.
        with self.assertRaisesRegex(
            UnsupportedError,
            "Only qExpectedHypervolumeImprovement is currently supported",
        ):
            MOOAcquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
                objective_thresholds=self.objective_thresholds,
                botorch_acqf_class=qNoisyExpectedImprovement,
                pending_observations=self.pending_observations,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )

    @patch(
        f"{DummyACQFClassMOO.__module__}.DummyACQFClassMOO.__call__", return_value=None
    )
    def test_evaluate(self, mock_call):
        self.acquisition.evaluate(X=self.X)
        mock_call.assert_called_with(X=self.X)
