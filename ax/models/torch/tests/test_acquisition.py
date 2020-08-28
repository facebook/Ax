#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import TrainingData


ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
class DummyACQFClass:
    def __init__(self, **kwargs: Any) -> None:
        pass


class AcquisitionTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.fidelity_features = [2]
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )

        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self.botorch_acqf_class = DummyACQFClass
        self.objective_weights = torch.tensor([1.0])
        self.pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]]),
            torch.tensor([[2.0, 6.0, 8.0]]),
        ]
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.target_fidelities = {2: 1.0}
        self.options = {"best_f": 0.0}
        self.acquisition = Acquisition(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )

        self.inequality_constraints = [
            (torch.tensor([0, 1]), torch.tensor([-1.0, 1.0]), 1)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {"num_restarts": 40, "raw_samples": 1024}

    @patch(
        f"{ACQUISITION_PATH}._get_X_pending_and_observed",
        return_value=(torch.tensor([2.0]), torch.tensor([3.0])),
    )
    @patch(f"{ACQUISITION_PATH}.subset_model", return_value=(None, None, None, None))
    @patch(f"{ACQUISITION_PATH}.get_botorch_objective")
    @patch(
        f"{CURRENT_PATH}.Acquisition.compute_model_dependencies",
        return_value={"current_value": 1.2},
    )
    @patch(f"{CURRENT_PATH}.Acquisition.compute_data_dependencies", return_value={})
    @patch(f"{DummyACQFClass.__module__}.DummyACQFClass.__init__", return_value=None)
    def test_init(
        self,
        mock_botorch_acqf_class,
        mock_compute_data_deps,
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
                bounds=self.bounds,
                objective_weights=self.objective_weights,
            )

        botorch_objective = LinearMCObjective(weights=torch.tensor([1.0]))
        mock_get_objective.return_value = botorch_objective
        acquisition = Acquisition(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )

        # Check `_get_X_pending_and_observed` kwargs
        mock_get_X.assert_called_with(
            Xs=self.training_data.Xs,
            pending_observations=self.pending_observations,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=self.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        # Call `subset_model` only when needed
        mock_subset_model.assert_called_with(
            acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
        )
        mock_subset_model.reset_mock()
        self.options["subset_model"] = False
        acquisition = Acquisition(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        mock_subset_model.assert_not_called()
        # Check `get_botorch_objective` kwargs
        mock_get_objective.assert_called_with(
            model=self.acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            X_observed=torch.tensor([3.0]),
            use_scalarized_objective=False,
        )
        # Check `compute_model_dependencies` kwargs
        mock_compute_model_deps.assert_called_with(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        # Check `compute_data_dependencies` kwargs
        mock_compute_data_deps.assert_called_with(training_data=self.training_data)
        # Check final `acqf` creation
        model_deps = {"current_value": 1.2}
        data_deps = {}
        mock_botorch_acqf_class.assert_called_with(
            model=self.acquisition.surrogate.model,
            objective=botorch_objective,
            X_pending=torch.tensor([2.0]),
            X_baseline=torch.tensor([3.0]),
            **self.options,
            **model_deps,
            **data_deps,
        )

    @patch(f"{ACQUISITION_PATH}.optimize_acqf")
    def test_optimize(self, mock_optimize_acqf):
        self.acquisition.optimize(
            bounds=self.bounds,
            n=3,
            optimizer_class=None,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf.assert_called_with(
            self.acquisition.acqf,
            bounds=self.bounds,
            q=3,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )

    @patch(f"{SURROGATE_PATH}.Surrogate.best_in_sample_point")
    def test_best_point(self, mock_best_point):
        self.acquisition.best_point(
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        mock_best_point.assert_called_with(
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
