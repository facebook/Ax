#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Any
from unittest import mock
from unittest.mock import Mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.sebo import (
    clamp_to_target,
    get_batch_initial_conditions,
    L1_norm_func,
    SEBOAcquisition,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.penalized import L0Approximation
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import ModelList
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import none_throws


SEBOACQUISITION_PATH: str = SEBOAcquisition.__module__
ACQUISITION_PATH: str = Acquisition.__module__
CURRENT_PATH: str = __name__
SURROGATE_PATH: str = Surrogate.__module__


class TestSebo(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        tkwargs: dict[str, Any] = {"dtype": torch.double}
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        # Function is f(x) = x_1 on [0, 1]^3, target point of [0.5, 0.5, 0.5].
        # Optimal soln is [1.0, 0.5, 0.5]
        self.X = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 1.0],
                [0.9, 1.0, 1.0],
                [0.9, 0.0, 1.0],
                [0.9, 1.0, 0.0],
            ],
            **tkwargs,
        )
        self.target_point = torch.tensor([0.5, 0.5, 0.5], **tkwargs)
        self.Y = self.X[:, 0].unsqueeze(-1)
        self.Yvar = 0.1 * torch.ones_like(self.Y)
        self.training_data = [
            SupervisedDataset(
                X=self.X, Y=self.Y, feature_names=["a", "b", "c"], outcome_names=["m1"]
            )
        ]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            target_values={2: 1.0},
        )
        self.surrogate.fit(
            datasets=self.training_data,
            search_space_digest=self.search_space_digest,
        )

        self.botorch_acqf_class = qNoisyExpectedHypervolumeImprovement
        self.objective_weights = torch.tensor([1.0], **tkwargs)
        # new transformed objective weights
        self.objective_weights_sebo = torch.tensor([1.0, -1.0], **tkwargs)
        self.objective_thresholds = torch.tensor([1.0], **tkwargs)
        self.objective_thresholds_sebo = torch.tensor([1.0, 3.0], **tkwargs)

        self.pending_observations = [torch.tensor([[0.5, 0.5, 0.5]], **tkwargs)]
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[2.0]], **tkwargs),
        )
        self.outcome_constraints_sebo = (
            torch.tensor([[1.0, 0.0]], **tkwargs),
            torch.tensor([[2.0]], **tkwargs),
        )
        self.linear_constraints = None
        self.fixed_features = {1: 1.0}
        self.options = {"best_f": 0.0, "target_point": self.target_point}
        self.inequality_constraints = [
            (torch.tensor([0, 1], **tkwargs), torch.tensor([1.0, -1.0], **tkwargs), 0)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {
            Keys.NUM_RESTARTS: 40,
            Keys.RAW_SAMPLES: 1024,
            "num_homotopy_steps": 3,
        }
        self.tkwargs = tkwargs
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.torch_opt_config_2 = TorchOptConfig(
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
        )

    def get_acquisition_function(
        self,
        fixed_features: dict[int, float] | None = None,
        options: dict[str, str | float] | None = None,
        torch_opt_config: TorchOptConfig | None = None,
    ) -> SEBOAcquisition:
        return SEBOAcquisition(
            botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                torch_opt_config or self.torch_opt_config,
                fixed_features=fixed_features or {},
            ),
            options=options or self.options,
        )

    def test_init(self) -> None:
        acquisition1 = self.get_acquisition_function(
            options={"target_point": self.target_point},
        )
        # Check that determinstic metric is added to surrogate
        surrogate = acquisition1.surrogate
        model_list = none_throws(surrogate._model)
        self.assertIsInstance(model_list, ModelList)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model_list.models[0], SingleTaskGP)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model_list.models[1], GenericDeterministicModel)

        # Check right penalty term is instantiated
        self.assertEqual(acquisition1.penalty_name, "L0_norm")
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model_list.models[1]._f, L0Approximation)
        # `a` needs to be set to something small for the pruning to work as expected
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertEqual(model_list.models[-1]._f.a, 1e-6)

        # Check transformed objective threshold
        self.assertTrue(
            torch.equal(
                # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slic...
                acquisition1.acqf.ref_point[-1],
                -self.objective_thresholds_sebo[-1],
            )
        )
        self.assertTrue(
            torch.equal(
                none_throws(acquisition1.objective_thresholds),
                self.objective_thresholds_sebo,
            )
        )
        self.assertEqual(acquisition1.sparsity_threshold, self.X.shape[-1])

        # Check using non-default penalty
        acquisition2 = self.get_acquisition_function(
            options={"penalty": "L1_norm", "target_point": self.target_point},
        )
        self.assertEqual(acquisition2.penalty_name, "L1_norm")
        surrogate = acquisition2.surrogate
        model_list = none_throws(surrogate._model)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model_list.models[1]._f, functools.partial)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIs(model_list.models[1]._f.func, L1_norm_func)

        # assert error raise when constructing non L0/L1 penalty terms
        with self.assertRaisesRegex(
            NotImplementedError, "L2_norm is not currently implemented."
        ):
            self.get_acquisition_function(
                fixed_features=self.fixed_features,
                options={"penalty": "L2_norm", "target_point": self.target_point},
            )

        # assert error raise if target point is not given
        with self.assertRaisesRegex(ValueError, "please provide target point."):
            self.get_acquisition_function(options={"penalty": "L1_norm"})

        # Cache root catches
        with warnings.catch_warnings(record=True) as ws:
            self.get_acquisition_function(
                fixed_features=self.fixed_features,
                options={"cache_root": True, "target_point": self.target_point},
            )
            self.assertEqual(len(ws), 1)
            self.assertEqual(
                "SEBO doesn't support `cache_root=True`. Changing it to `False`.",
                str(ws[0].message),
            )

        # Test with no outcome constraints
        self.get_acquisition_function(
            options={"target_point": self.target_point},
            torch_opt_config=self.torch_opt_config_2,
        )

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf")
    def test_optimize_l1(self, mock_optimize_acqf: Mock) -> None:
        mock_optimize_acqf.return_value = (
            torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.double),
            torch.tensor([1.0, 2.0], dtype=torch.double),
        )
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L1_norm", "target_point": self.target_point},
        )
        acquisition.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )

        args, kwargs = mock_optimize_acqf.call_args
        self.assertEqual(kwargs["acq_function"], acquisition.acqf)
        self.assertEqual(kwargs["q"], 2)
        self.assertEqual(kwargs["inequality_constraints"], self.inequality_constraints)
        self.assertEqual(kwargs["post_processing_func"], self.rounding_func)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])

    @mock.patch(f"{SEBOACQUISITION_PATH}.optimize_acqf_homotopy")
    def test_optimize_l0(self, mock_optimize_acqf_homotopy: Mock) -> None:
        mock_optimize_acqf_homotopy.return_value = (
            torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.double),
            torch.tensor([1.0, 2.0], dtype=torch.double),
        )
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        acquisition.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )

        args, kwargs = mock_optimize_acqf_homotopy.call_args
        self.assertEqual(kwargs["acq_function"], acquisition.acqf)
        self.assertEqual(kwargs["q"], 2)
        self.assertEqual(kwargs["post_processing_func"], self.rounding_func)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])
        self.assertEqual(
            kwargs["homotopy"]._homotopy_parameters[0].schedule.num_steps,
            self.optimizer_options["num_homotopy_steps"],
        )

        # set self.acqf.cache_pending as False
        acquisition2 = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        # pyre-fixme[16]: `AcquisitionFunction` has no attribute `cache_pending`.
        acquisition2.acqf.cache_pending = torch.tensor(False)
        acquisition2.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        _args, kwargs = mock_optimize_acqf_homotopy.call_args
        self.assertEqual(kwargs["acq_function"], acquisition2.acqf)
        self.assertEqual(kwargs["q"], 2)
        self.assertEqual(kwargs["inequality_constraints"], self.inequality_constraints)
        self.assertEqual(kwargs["post_processing_func"], self.rounding_func)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])

    def test_optimization_result(self) -> None:
        acquisition = self.get_acquisition_function(
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        # Function is f(x) = x_1 on [0, 1]^3, target point of [0.5, 0.5, 0.5].
        # Optimal soln is [1.0, 0.5, 0.5]
        optimizer_options = {Keys.NUM_RESTARTS: 1, Keys.RAW_SAMPLES: 1000}
        xopt, _, _ = acquisition.optimize(
            n=1,
            search_space_digest=self.search_space_digest,
            optimizer_options=optimizer_options,  # pyre-ignore
        )
        self.assertTrue(
            torch.allclose(xopt[0, 1:], torch.tensor([0.5, 0.5], **self.tkwargs))
        )
        self.assertTrue(
            torch.isclose(xopt[0, 0], torch.tensor(1.0, **self.tkwargs), atol=0.3)
        )

    def test_clamp_to_target(self) -> None:
        X = torch.tensor(
            [[0.5, 0.01, 0.5], [0.05, 0.5, 0.95], [0.1, 0.02, 0.06]], **self.tkwargs
        )
        X_true = torch.tensor(
            [[0.5, 0, 0.5], [0, 0.5, 0.95], [0.1, 0, 0.06]], **self.tkwargs
        )
        self.assertTrue(
            torch.allclose(
                clamp_to_target(X, torch.zeros(1, 3, **self.tkwargs), 0.05), X_true
            )
        )

    @mock.patch(f"{SEBOACQUISITION_PATH}.optimize_acqf_homotopy")
    @mock.patch(
        f"{SEBOACQUISITION_PATH}.get_batch_initial_conditions",
        wraps=get_batch_initial_conditions,
    )
    def test_get_batch_initial_conditions(
        self, mock_get_batch_initial_conditions: Mock, mock_optimize_acqf_homotopy: Mock
    ) -> None:
        mock_optimize_acqf_homotopy.return_value = (
            torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=torch.double),
            torch.tensor([1.0, 2.0], dtype=torch.double),
        )
        acquisition = self.get_acquisition_function(
            fixed_features={1: 0.5},
            options={"target_point": self.target_point},
            torch_opt_config=self.torch_opt_config_2,
        )
        acquisition.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options={Keys.NUM_RESTARTS: 3, Keys.RAW_SAMPLES: 32},
        )
        call_args = mock_get_batch_initial_conditions.call_args[1]
        self.assertTrue(
            torch.equal(
                call_args["X_pareto"],
                torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.double),
            )
        )
        self.assertTrue(torch.equal(call_args["target_point"], self.target_point))
        self.assertEqual(call_args["raw_samples"], 32)
        self.assertEqual(call_args["num_restarts"], 3)
        # Check the batch initial conditions
        batch_initial_conditions = mock_optimize_acqf_homotopy.call_args[1][
            "batch_initial_conditions"
        ]
        self.assertEqual(batch_initial_conditions.shape, torch.Size([3, 1, 3]))
        self.assertTrue(torch.all(batch_initial_conditions[:1] != 0.5))
        self.assertTrue(torch.all(batch_initial_conditions[1:, :, 1] == 0.5))

    @mock.patch(f"{SEBOACQUISITION_PATH}.optimize_acqf_homotopy")
    @mock.patch(
        f"{SEBOACQUISITION_PATH}.get_batch_initial_conditions",
        wraps=get_batch_initial_conditions,
    )
    def test_optimize_with_provided_batch_initial_conditions(
        self, mock_get_batch_initial_conditions: Mock, mock_optimize_acqf_homotopy: Mock
    ) -> None:
        mock_optimize_acqf_homotopy.return_value = (
            torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.double),
            torch.tensor([1.0], dtype=torch.double),
        )

        # Create batch initial conditions
        batch_ics = torch.rand(3, 1, 3, dtype=torch.double)

        acquisition = self.get_acquisition_function(
            options={
                "target_point": self.target_point,
                "penalty": "L0_norm",
            },
        )

        acquisition.optimize(
            n=1,
            search_space_digest=self.search_space_digest,
            optimizer_options={
                "batch_initial_conditions": batch_ics,
                Keys.NUM_RESTARTS: 3,
                Keys.RAW_SAMPLES: 32,
            },
        )

        # Verify get_batch_initial_conditions was not called
        mock_get_batch_initial_conditions.assert_not_called()

        # Verify the batch_initial_conditions were passed to optimize_acqf_homotopy
        call_kwargs = mock_optimize_acqf_homotopy.call_args[1]
        self.assertTrue(torch.equal(call_kwargs["batch_initial_conditions"], batch_ics))
