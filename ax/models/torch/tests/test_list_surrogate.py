#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import (
    NOT_YET_FIT_MSG,
    ListSurrogate,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.models.model import TrainingData
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


SURROGATE_PATH = f"{Surrogate.__module__}"
CURRENT_PATH = f"{__name__}"
ACQUISITION_PATH = f"{Acquisition.__module__}"
RANK = "rank"


class ListSurrogateTest(TestCase):
    def setUp(self):
        self.outcomes = ["outcome_1", "outcome_2"]
        self.mll_class = SumMarginalLogLikelihood
        self.dtype = torch.float
        self.task_features = [0]
        Xs1, Ys1, Yvars1, bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.task_features
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.task_features
        )
        self.botorch_submodel_class_per_outcome = {
            self.outcomes[0]: choose_model_class(
                Yvars=Yvars1, task_features=self.task_features, fidelity_features=[]
            ),
            self.outcomes[1]: choose_model_class(
                Yvars=Yvars2, task_features=self.task_features, fidelity_features=[]
            ),
        }
        self.expected_submodel_type = FixedNoiseMultiTaskGP
        for submodel_cls in self.botorch_submodel_class_per_outcome.values():
            self.assertEqual(submodel_cls, FixedNoiseMultiTaskGP)
        self.Xs = Xs1 + Xs2
        self.Ys = Ys1 + Ys2
        self.Yvars = Yvars1 + Yvars2
        self.training_data = [
            TrainingData(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(self.Xs, self.Ys, self.Yvars)
        ]
        self.submodel_options_per_outcome = {
            self.outcomes[0]: {RANK: 1},
            self.outcomes[1]: {RANK: 2},
        }
        self.surrogate = ListSurrogate(
            botorch_submodel_class_per_outcome=self.botorch_submodel_class_per_outcome,
            mll_class=self.mll_class,
            submodel_options_per_outcome=self.submodel_options_per_outcome,
        )
        self.bounds = [(0.0, 1.0), (1.0, 4.0)]
        self.feature_names = ["x1", "x2"]

    def check_ranks(self, c: ListSurrogate) -> type(None):
        self.assertIsInstance(c, ListSurrogate)
        self.assertIsInstance(c.model, ModelListGP)
        for idx, submodel in enumerate(c.model.models):
            self.assertIsInstance(submodel, self.expected_submodel_type)
            self.assertEqual(
                submodel._rank,
                self.submodel_options_per_outcome[self.outcomes[idx]][RANK],
            )

    def test_init(self):
        self.assertEqual(
            self.surrogate.botorch_submodel_class_per_outcome,
            self.botorch_submodel_class_per_outcome,
        )
        self.assertEqual(self.surrogate.mll_class, self.mll_class)
        with self.assertRaises(NotImplementedError):
            self.surrogate.training_data
        with self.assertRaisesRegex(ValueError, NOT_YET_FIT_MSG):
            self.surrogate.training_data_per_outcome
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed"
        ):
            self.surrogate.model

    @patch(
        f"{CURRENT_PATH}.FixedNoiseMultiTaskGP.construct_inputs",
        # Mock to register calls, but still execute the function.
        side_effect=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_options(self, mock_MTGP_construct_inputs):
        with self.assertRaisesRegex(ValueError, ".* are required"):
            self.surrogate.construct(training_data=self.training_data)
        with self.assertRaisesRegex(ValueError, "No model class specified for"):
            self.surrogate.construct(
                training_data=self.training_data, metric_names=["new_metric"]
            )
        self.surrogate.construct(
            training_data=self.training_data,
            task_features=self.task_features,
            metric_names=self.outcomes,
        )
        self.check_ranks(self.surrogate)
        # Should construct inputs for MTGP twice.
        self.assertEqual(len(mock_MTGP_construct_inputs.call_args_list), 2)
        # First construct inputs should be called for MTGP with training data #0.
        for idx in range(len(mock_MTGP_construct_inputs.call_args_list)):
            self.assertEqual(
                # `call_args` is a tuple of (args, kwargs), and we check kwargs.
                mock_MTGP_construct_inputs.call_args_list[idx][1],
                {
                    "fidelity_features": [],
                    "task_features": self.task_features,
                    "training_data": self.training_data[idx],
                    "rank": self.submodel_options_per_outcome[self.outcomes[idx]][
                        "rank"
                    ],
                },
            )

    @patch(
        f"{CURRENT_PATH}.FixedNoiseMultiTaskGP.construct_inputs",
        # Mock to register calls, but still execute the function.
        side_effect=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_shared_shortcut_options(self, mock_construct_inputs):
        surrogate = ListSurrogate(
            botorch_submodel_class=self.botorch_submodel_class_per_outcome[
                self.outcomes[0]
            ],
            submodel_options={"shared_option": True},
            submodel_options_per_outcome={
                outcome: {"individual_option": f"val_{idx}"}
                for idx, outcome in enumerate(self.outcomes)
            },
        )
        surrogate.construct(
            training_data=self.training_data,
            task_features=self.task_features,
            metric_names=self.outcomes,
        )
        # 2 submodels should've been constructed, both of type `botorch_submodel_class`.
        self.assertEqual(len(mock_construct_inputs.call_args_list), 2)
        first_call_args, second_call_args = mock_construct_inputs.call_args_list
        for idx in range(len(mock_construct_inputs.call_args_list)):
            self.assertEqual(
                mock_construct_inputs.call_args_list[idx][1],
                {
                    "fidelity_features": [],
                    "individual_option": f"val_{idx}",
                    "shared_option": True,
                    "task_features": [0],
                    "training_data": self.training_data[idx],
                },
            )

    @patch(f"{CURRENT_PATH}.ModelListGP.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.SumMarginalLogLikelihood")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    def test_fit(self, mock_fit_gpytorch, mock_MLL, mock_state_dict):
        surrogate = ListSurrogate(
            botorch_submodel_class_per_outcome=self.botorch_submodel_class_per_outcome,
            mll_class=SumMarginalLogLikelihood,
        )
        # Checking that model is None before `fit` (and `construct`) calls.
        self.assertIsNone(surrogate._model)
        # Should instantiate mll and `fit_gpytorch_model` when `state_dict`
        # is `None`.
        surrogate.fit(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.outcomes,
            fidelity_features=[],
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
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.outcomes,
            fidelity_features=[],
            refit=False,
            state_dict=state_dict,
        )
        mock_state_dict.assert_called_once()
        mock_MLL.assert_not_called()
        mock_fit_gpytorch.assert_not_called()

    def test_serialize_attributes_as_kwargs(self):
        expected = self.surrogate.__dict__
        # The two attributes below don't need to be saved as part of state,
        # so we remove them from the expected dict.
        expected.pop("botorch_model_class")
        expected.pop("model_options")
        self.assertEqual(self.surrogate._serialize_attributes_as_kwargs(), expected)
