#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Dict
from unittest import mock
from unittest.mock import Mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.sebo import SEBOAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.deterministic_metric import L1_norm_func
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition import PosteriorMean
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.penalized import L0Approximation
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import ModelList
from botorch.optim import Homotopy, HomotopyParameter, LinearHomotopySchedule
from botorch.utils.datasets import SupervisedDataset
from torch.nn import Parameter


SEBOACQUISITION_PATH: str = SEBOAcquisition.__module__
ACQUISITION_PATH: str = Acquisition.__module__
CURRENT_PATH: str = __name__
SURROGATE_PATH: str = Surrogate.__module__


class TestSebo(TestCase):
    def setUp(self) -> None:
        tkwargs: Dict[str, Any] = {"dtype": torch.double}
        self.botorch_model_class = SingleTaskGP
        self.surrogates = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        self.target_point = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        self.Y = torch.tensor([[3.0], [4.0]], **tkwargs)
        self.Yvar = torch.tensor([[0.0], [2.0]], **tkwargs)
        self.training_data = [SupervisedDataset(X=self.X, Y=self.Y)]
        self.fidelity_features = [2]
        self.surrogates.construct(
            datasets=self.training_data,
            metric_names=["m1"],
            fidelity_features=self.fidelity_features,
        )
        self.surrogates._outcomes = ["m1"]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_fidelities={2: 1.0},
        )
        self.botorch_acqf_class = qNoisyExpectedHypervolumeImprovement
        self.objective_weights = torch.tensor([1.0], **tkwargs)
        # new transformed objective weights
        self.objective_weights_sebo = torch.tensor([1.0, -1.0], **tkwargs)
        self.objective_thresholds = torch.tensor([1.0], **tkwargs)
        self.objective_thresholds_sebo = torch.tensor([1.0, 3.0], **tkwargs)

        self.pending_observations = [torch.tensor([[1.0, 3.0, 4.0]], **tkwargs)]
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.outcome_constraints_sebo = (
            torch.tensor([[1.0, 0.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {"best_f": 0.0, "target_point": self.target_point}
        self.inequality_constraints = [
            (torch.tensor([0, 1], **tkwargs), torch.tensor([-1.0, 1.0], **tkwargs), 1)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.tkwargs = tkwargs
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_acquisition_function(self, fixed_features=None, options=None):
        return SEBOAcquisition(
            botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
            surrogates={Keys.ONLY_SURROGATE: self.surrogates},
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, fixed_features=fixed_features or {}
            ),
            options=options or self.options,
        )

    def test_init(self) -> None:
        acquisition1 = self.get_acquisition_function(
            options={"target_point": self.target_point},
        )
        # Check that determinstic metric is added to surrogate
        surrogate = acquisition1.surrogates["sebo"]
        self.assertIsInstance(surrogate._model, ModelList)
        self.assertIsInstance(surrogate._model.models[0], SingleTaskGP)
        self.assertIsInstance(surrogate._model.models[1], GenericDeterministicModel)
        self.assertEqual(acquisition1.det_metric_indx, -1)

        # Check right penalty term is instantiated
        self.assertEqual(acquisition1.penalty_name, "L0_norm")
        self.assertIsInstance(surrogate._model.models[1]._f, L0Approximation)

        # Check transformed objective threshold
        self.assertTrue(
            torch.equal(
                acquisition1.acqf.ref_point[-1], -1 * self.objective_thresholds_sebo[-1]
            )
        )
        self.assertTrue(
            torch.equal(
                acquisition1.objective_thresholds, self.objective_thresholds_sebo
            )
        )
        self.assertEqual(acquisition1.sparsity_threshold, self.X.shape[-1])

        # Check using non-default penalty
        acquisition2 = self.get_acquisition_function(
            options={"penalty": "L1_norm", "target_point": self.target_point},
        )
        self.assertEqual(acquisition2.penalty_name, "L1_norm")
        self.assertIsInstance(
            acquisition2.surrogates["sebo"]._model.models[1]._f, functools.partial
        )
        self.assertIs(
            acquisition2.surrogates["sebo"]._model.models[1]._f.func, L1_norm_func
        )

        # assert error raise when constructing non L0/L1 penalty terms
        with self.assertRaisesRegex(
            NotImplementedError, "L2_norm is not currently implemented."
        ):
            self.get_acquisition_function(
                fixed_features=self.fixed_features,
                options={"penalty": "L2_norm", "target_point": self.target_point},
            )

        # assert error raise if multiple surrogates are given
        with self.assertRaisesRegex(
            ValueError, "SEBO does not support support multiple surrogates."
        ):
            SEBOAcquisition(
                botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
                surrogates={
                    Keys.ONLY_SURROGATE: self.surrogates,
                    "sebo2": self.surrogates,
                },
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
                options=self.options,
            )

        # assert error raise if target point is not given
        with self.assertRaisesRegex(ValueError, "please provide target point."):
            self.get_acquisition_function(options={"penalty": "L1_norm"})

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf")
    def test_optimize_l1(self, mock_optimize_acqf: Mock) -> None:
        mock_optimize_acqf.return_value = (
            # pyre-fixme[6]: For 2nd param expected `bool` but got `dtype`.
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], **{"dtype": torch.double}),
            # pyre-fixme[6]: For 2nd param expected `bool` but got `dtype`.
            torch.tensor([1.0, 2.0], **{"dtype": torch.double}),
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

    @mock.patch(
        f"{SEBOACQUISITION_PATH}.get_batch_initial_conditions", return_value=None
    )
    @mock.patch(f"{SEBOACQUISITION_PATH}.Homotopy")
    def test_optimize_l0_homotopy(
        self,
        mock_homotopy: Mock,
        mock_get_batch_initial_conditions: Mock,
    ) -> None:
        tkwargs = {"dtype": torch.double}
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        # overwrite acqf to validate homotopy
        model = GenericDeterministicModel(f=lambda x: 5 - (x - p) ** 2)
        acqf = PosteriorMean(model=model)
        acquisition.acqf = acqf

        p = Parameter(-2 * torch.ones(1, **tkwargs))  # pyre-ignore
        hp = HomotopyParameter(
            parameter=p,
            schedule=LinearHomotopySchedule(start=4, end=0, num_steps=5),
        )
        mock_homotopy.return_value = Homotopy(homotopy_parameters=[hp])

        search_space_digest = SearchSpaceDigest(
            feature_names=["a"],
            bounds=[(-10.0, 5.0)],
        )
        candidate, acqf_val = acquisition._optimize_with_homotopy(
            n=1,
            search_space_digest=search_space_digest,
            optimizer_options={
                "num_restarts": 2,
                "sequential": True,
                "raw_samples": 16,
            },
        )
        self.assertEqual(candidate, torch.zeros(1, **tkwargs))  # pyre-ignore
        self.assertEqual(acqf_val, 5 * torch.ones(1, **tkwargs))  # pyre-ignore

    @mock.patch(f"{SEBOACQUISITION_PATH}.optimize_acqf_homotopy")
    def test_optimize_l0(self, mock_optimize_acqf_homotopy: Mock) -> None:
        mock_optimize_acqf_homotopy.return_value = (
            # pyre-fixme[6]: For 2nd param expected `bool` but got `dtype`.
            torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], **{"dtype": torch.double}),
            # pyre-fixme[6]: For 2nd param expected `bool` but got `dtype`.
            torch.tensor([1.0, 2.0], **{"dtype": torch.double}),
        )
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        acquisition.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            # does not support in homotopy now
            # inequality_constraints=self.inequality_constraints,
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

        # set self.acqf.cache_pending as False
        acquisition2 = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        acquisition2.acqf.cache_pending = torch.tensor(False)
        acquisition2.optimize(
            n=2,
            search_space_digest=self.search_space_digest,
            # does not support in homotopy now
            # inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        args, kwargs = mock_optimize_acqf_homotopy.call_args
        self.assertEqual(kwargs["acq_function"], acquisition2.acqf)
        self.assertEqual(kwargs["q"], 2)
        self.assertEqual(kwargs["post_processing_func"], self.rounding_func)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])

        # assert error raise with inequality_constraints input
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
            options={"penalty": "L0_norm", "target_point": self.target_point},
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            "Homotopy does not support optimization with inequality "
            + "constraints. Use L1 penalty norm instead.",
        ):
            acquisition.optimize(
                n=2,
                search_space_digest=self.search_space_digest,
                inequality_constraints=self.inequality_constraints,
                fixed_features=self.fixed_features,
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )

        # assert error when using a wrong botorch_acqf_class
        with self.assertRaisesRegex(
            ValueError, "botorch_acqf_class must be qEHVI to use SEBO"
        ):
            acquisition = SEBOAcquisition(
                botorch_acqf_class=qNoisyExpectedImprovement,
                surrogates={Keys.ONLY_SURROGATE: self.surrogates},
                search_space_digest=self.search_space_digest,
                torch_opt_config=dataclasses.replace(
                    self.torch_opt_config, fixed_features=self.fixed_features
                ),
                options=self.options,
            )
