#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
from typing import Any
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData


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
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.Y = torch.tensor([[3.0], [4.0]])
        self.Yvar = torch.tensor([[0.0], [2.0]])
        self.training_data = TrainingData(X=self.X, Y=self.Y, Yvar=self.Yvar)
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

    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model", return_value=(None, None, None, None)
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
        self.acquisition.default_botorch_acqf_class = None
        with self.assertRaisesRegex(
            ValueError, ".*`botorch_acqf_class` argument must be specified."
        ):
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
            acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )
        mock_subset_model.reset_mock()
        mock_get_objective.reset_mock()
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
        mock_botorch_acqf_class.assert_called_once()
        _, ckwargs = mock_botorch_acqf_class.call_args
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
            optimizer_class=None,
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

    def test_extract_training_data(self):
        self.assertEqual(  # Base `Surrogate` case.
            self.acquisition._extract_training_data(surrogate=self.surrogate),
            self.training_data,
        )
        # `ListSurrogate` case.
        list_surrogate = ListSurrogate(botorch_submodel_class=self.botorch_model_class)
        list_surrogate._training_data_per_outcome = {"a": self.training_data}
        self.assertEqual(
            self.acquisition._extract_training_data(surrogate=list_surrogate),
            list_surrogate._training_data_per_outcome,
        )
