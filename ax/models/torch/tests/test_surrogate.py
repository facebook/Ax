#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.containers import TrainingData
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"


class SurrogateTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.mll_class = ExactMarginalLogLikelihood
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, self.bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype
        )
        self.training_data = TrainingData.from_block_design(
            X=self.Xs[0], Y=self.Ys[0], Yvar=self.Yvars[0]
        )
        self.surrogate_kwargs = self.botorch_model_class.construct_inputs(
            self.training_data
        )
        self.surrogate = Surrogate(
            botorch_model_class=self.botorch_model_class, mll_class=self.mll_class
        )
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2"],
            bounds=self.bounds,
            target_fidelities={1: 1.0},
        )
        self.metric_names = ["y"]
        self.fixed_features = {1: 2.0}
        self.refit = True
        self.objective_weights = torch.tensor(
            [-1.0, 1.0], dtype=self.dtype, device=self.device
        )
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        self.options = {}

    @patch(f"{CURRENT_PATH}.Kernel")
    @patch(f"{CURRENT_PATH}.Likelihood")
    def test_init(self, mock_Likelihood, mock_Kernel):
        self.assertEqual(self.surrogate.botorch_model_class, self.botorch_model_class)
        self.assertEqual(self.surrogate.mll_class, self.mll_class)
        with self.assertRaisesRegex(NotImplementedError, "Customizing likelihood"):
            Surrogate(
                botorch_model_class=self.botorch_model_class, likelihood=Likelihood()
            )
        with self.assertRaisesRegex(NotImplementedError, "Customizing kernel"):
            Surrogate(
                botorch_model_class=self.botorch_model_class, kernel_class=Kernel()
            )

    def test_model_property(self):
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed."
        ):
            self.surrogate.model

    def test_training_data_property(self):
        with self.assertRaisesRegex(
            ValueError,
            "Underlying BoTorch `Model` has not yet received its training_data.",
        ):
            self.surrogate.training_data

    def test_dtype_property(self):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        self.assertEqual(self.dtype, self.surrogate.dtype)

    def test_device_property(self):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        self.assertEqual(self.device, self.surrogate.device)

    def test_from_botorch(self):
        surrogate = Surrogate.from_botorch(
            self.botorch_model_class(**self.surrogate_kwargs)
        )
        self.assertIsInstance(surrogate.model, self.botorch_model_class)
        self.assertTrue(surrogate._constructed_manually)

    @patch(f"{CURRENT_PATH}.SingleTaskGP.__init__", return_value=None)
    def test_construct(self, mock_GP):
        with self.assertRaises(NotImplementedError):
            # Base `Model` does not implement `construct_inputs`.
            Surrogate(botorch_model_class=Model).construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        mock_GP.assert_called_once()
        call_kwargs = mock_GP.call_args[1]
        self.assertTrue(torch.equal(call_kwargs["train_X"], self.Xs[0]))
        self.assertTrue(torch.equal(call_kwargs["train_Y"], self.Ys[0]))
        self.assertFalse(self.surrogate._constructed_manually)

        # Check that `model_options` passed to the `Surrogate` constructor are
        # properly propagated.
        with patch.object(
            SingleTaskGP, "construct_inputs", wraps=SingleTaskGP.construct_inputs
        ) as mock_construct_inputs:
            surrogate = Surrogate(
                botorch_model_class=self.botorch_model_class,
                mll_class=self.mll_class,
                model_options={"some_option": "some_value"},
            )
            surrogate.construct(self.training_data)
            mock_construct_inputs.assert_called_with(
                training_data=self.training_data, some_option="some_value"
            )

    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    def test_fit(self, mock_fit_gpytorch, mock_MLL, mock_state_dict):
        surrogate = Surrogate(
            botorch_model_class=self.botorch_model_class,
            mll_class=ExactMarginalLogLikelihood,
        )
        # Checking that model is None before `fit` (and `construct`) calls.
        self.assertIsNone(surrogate._model)
        # Should instantiate mll and `fit_gpytorch_model` when `state_dict`
        # is `None`.
        surrogate.fit(
            training_data=self.training_data,
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
            refit=self.refit,
        )
        # Check that training data is correctly passed through to the
        # BoTorch `Model`.
        self.assertTrue(
            torch.equal(
                surrogate.model.train_inputs[0],
                self.surrogate_kwargs.get("train_X"),
            )
        )
        self.assertTrue(
            torch.equal(
                surrogate.model.train_targets,
                self.surrogate_kwargs.get("train_Y").squeeze(1),
            )
        )
        mock_state_dict.assert_not_called()
        mock_MLL.assert_called_once()
        mock_fit_gpytorch.assert_called_once()
        mock_state_dict.reset_mock()
        mock_MLL.reset_mock()
        mock_fit_gpytorch.reset_mock()
        # Should `load_state_dict` when `state_dict` is not `None`
        # and `refit` is `False`.
        state_dict = {"state_attribute": "value"}
        surrogate.fit(
            training_data=self.training_data,
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
            refit=False,
            state_dict=state_dict,
        )
        mock_state_dict.assert_called_once()
        mock_MLL.assert_not_called()
        mock_fit_gpytorch.assert_not_called()

    @patch(f"{SURROGATE_PATH}.predict_from_model")
    def test_predict(self, mock_predict):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        self.surrogate.predict(X=self.Xs[0])
        mock_predict.assert_called_with(model=self.surrogate.model, X=self.Xs[0])

    def test_best_in_sample_point(self):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        # `best_in_sample_point` requires `objective_weights`
        with patch(
            f"{SURROGATE_PATH}.best_in_sample_point", return_value=None
        ) as mock_best_in_sample:
            with self.assertRaisesRegex(ValueError, "Could not obtain"):
                self.surrogate.best_in_sample_point(
                    search_space_digest=self.search_space_digest, objective_weights=None
                )
        with patch(
            f"{SURROGATE_PATH}.best_in_sample_point", return_value=(self.Xs[0], 0.0)
        ) as mock_best_in_sample:
            best_point, observed_value = self.surrogate.best_in_sample_point(
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )
            mock_best_in_sample.assert_called_with(
                Xs=[self.training_data.X],
                model=self.surrogate,
                bounds=self.search_space_digest.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )

    @patch(f"{ACQUISITION_PATH}.Acquisition.__init__", return_value=None)
    @patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        return_value=([torch.tensor([0.0])], [torch.tensor([1.0])]),
    )
    @patch(
        f"{SURROGATE_PATH}.pick_best_out_of_sample_point_acqf_class",
        return_value=(qSimpleRegret, {Keys.SAMPLER: SobolQMCNormalSampler}),
    )
    def test_best_out_of_sample_point(
        self, mock_best_point_util, mock_acqf_optimize, mock_acqf_init
    ):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        # currently cannot use function with fixed features
        with self.assertRaisesRegex(NotImplementedError, "Fixed features"):
            self.surrogate.best_out_of_sample_point(
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
                fixed_features=self.fixed_features,
            )
        candidate, acqf_value = self.surrogate.best_out_of_sample_point(
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            options=self.options,
        )
        mock_acqf_init.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=qSimpleRegret,
            search_space_digest=self.search_space_digest,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=None,
            options={Keys.SAMPLER: SobolQMCNormalSampler},
        )
        self.assertTrue(torch.equal(candidate, torch.tensor([0.0])))
        self.assertTrue(torch.equal(acqf_value, torch.tensor([1.0])))

    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    def test_update(self, mock_fit_gpytorch, mock_MLL, mock_state_dict):
        self.surrogate.construct(
            training_data=self.training_data,
            fidelity_features=self.search_space_digest.fidelity_features,
        )
        # Check that correct arguments are passed to `fit`.
        with patch(f"{SURROGATE_PATH}.Surrogate.fit") as mock_fit:
            # Call `fit` by default
            self.surrogate.update(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=self.refit,
                state_dict={"key": "val"},
            )
            mock_fit.assert_called_with(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                candidate_metadata=None,
                refit=self.refit,
                state_dict={"key": "val"},
            )

        # Check that the training data is correctly passed through to the
        # BoTorch `Model`.
        Xs, Ys, Yvars, bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype, offset=1.0
        )
        training_data = TrainingData.from_block_design(X=Xs[0], Y=Ys[0], Yvar=Yvars[0])
        surrogate_kwargs = self.botorch_model_class.construct_inputs(training_data)
        self.surrogate.update(
            training_data=training_data,
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
            refit=self.refit,
            state_dict={"key": "val"},
        )
        self.assertTrue(
            torch.equal(
                self.surrogate.model.train_inputs[0],
                surrogate_kwargs.get("train_X"),
            )
        )
        self.assertTrue(
            torch.equal(
                self.surrogate.model.train_targets,
                surrogate_kwargs.get("train_Y").squeeze(1),
            )
        )

        # If should not be reconstructed, check that error is raised.
        self.surrogate._constructed_manually = True
        with self.assertRaisesRegex(NotImplementedError, ".* constructed manually"):
            self.surrogate.update(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=self.refit,
            )

    def test_serialize_attributes_as_kwargs(self):
        expected = self.surrogate.__dict__
        self.assertEqual(self.surrogate._serialize_attributes_as_kwargs(), expected)
