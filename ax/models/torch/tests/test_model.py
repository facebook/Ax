#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import ANY, patch, MagicMock

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kg import KnowledgeGradient
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.containers import TrainingData


CURRENT_PATH = __name__
MODEL_PATH = BoTorchModel.__module__
SURROGATE_PATH = Surrogate.__module__
UTILS_PATH = choose_model_class.__module__
ACQUISITION_PATH = Acquisition.__module__
LIST_SURROGATE_PATH = ListSurrogate.__module__


class BoTorchModelTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.acquisition_class = KnowledgeGradient
        self.botorch_acqf_class = qKnowledgeGradient
        self.acquisition_options = {Keys.NUM_FANTASIES: 64}
        self.model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            acquisition_options=self.acquisition_options,
        )

        self.dtype = torch.float
        Xs1, Ys1, Yvars1, self.bounds, _, _, _ = get_torch_test_data(dtype=self.dtype)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(dtype=self.dtype, offset=1.0)
        self.Xs = Xs1 + Xs2
        self.Ys = Ys1 + Ys2
        self.Yvars = Yvars1 + Yvars2
        self.X = Xs1[0]
        self.Y = Ys1[0]
        self.Yvar = Yvars1[0]
        self.X2 = Xs2[0]
        self.training_data = TrainingData(X=self.X, Y=self.Y, Yvar=self.Yvar)
        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self.task_features = []
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.metric_names_for_list_surrogate = ["y1", "y2"]
        self.fidelity_features = [2]
        self.target_fidelities = {1: 1.0}
        self.candidate_metadata = []

        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.model_gen_options = {Keys.OPTIMIZER_KWARGS: self.optimizer_options}
        self.objective_weights = torch.tensor([1.0])
        self.outcome_constraints = None
        self.linear_constraints = None
        self.fixed_features = None
        self.pending_observations = None
        self.rounding_func = "func"

    def test_init(self):
        # Default model with no specifications.
        model = BoTorchModel()
        self.assertEqual(model.acquisition_class, Acquisition)
        # Model that specifies `botorch_acqf_class`.
        model = BoTorchModel(botorch_acqf_class=qExpectedImprovement)
        self.assertEqual(model.acquisition_class, Acquisition)
        self.assertEqual(model.botorch_acqf_class, qExpectedImprovement)
        # Model with `Acquisition` that specifies a `default_botorch_acqf_class`.
        model = BoTorchModel(acquisition_class=KnowledgeGradient)
        self.assertEqual(model.acquisition_class, KnowledgeGradient)
        self.assertEqual(model.botorch_acqf_class, qKnowledgeGradient)

        # Check defaults for refitting settings.
        self.assertTrue(model.refit_on_update)
        self.assertFalse(model.refit_on_cv)
        self.assertTrue(model.warm_start_refit)

        # Check setting non-default refitting settings
        mdl2 = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            acquisition_options=self.acquisition_options,
            refit_on_update=False,
            refit_on_cv=True,
            warm_start_refit=False,
        )
        self.assertFalse(mdl2.refit_on_update)
        self.assertTrue(mdl2.refit_on_cv)
        self.assertFalse(mdl2.warm_start_refit)

    def test_surrogate_property(self):
        self.assertEqual(self.surrogate, self.model.surrogate)
        self.model._surrogate = None
        with self.assertRaisesRegex(ValueError, "Surrogate has not yet been set."):
            self.model.surrogate

    def test_botorch_acqf_class_property(self):
        self.assertEqual(self.botorch_acqf_class, self.model.botorch_acqf_class)
        self.model._botorch_acqf_class = None
        with self.assertRaisesRegex(
            ValueError, "`AcquisitionFunction` has not yet been set."
        ):
            self.model.botorch_acqf_class

    @patch(f"{SURROGATE_PATH}.Surrogate.fit")
    @patch(f"{MODEL_PATH}.choose_model_class", return_value=SingleTaskGP)
    def test_fit(self, mock_choose_model_class, mock_fit):
        # If surrogate is not yet set, initialize it with dispatcher functions.
        self.model._surrogate = None
        self.model.fit(
            Xs=[self.X],
            Ys=[self.Y],
            Yvars=[self.Yvar],
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
        )
        # `choose_model_class` is called.
        mock_choose_model_class.assert_called_with(
            Yvars=[self.Yvar],
            task_features=self.task_features,
            fidelity_features=self.fidelity_features,
        )
        # Since we want to refit on updates but not warm start refit, we clear the
        # state dict.
        mock_fit.assert_called_with(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            metric_names=self.metric_names,
            candidate_metadata=self.candidate_metadata,
            state_dict=None,
            refit=True,
        )

    @patch(f"{SURROGATE_PATH}.Surrogate.update")
    def test_update(self, mock_update):
        fit_update_shared_kwargs = {
            "bounds": self.bounds,
            "task_features": self.task_features,
            "feature_names": self.feature_names,
            "metric_names": self.metric_names,
            "fidelity_features": self.fidelity_features,
            "target_fidelities": self.target_fidelities,
            "candidate_metadata": self.candidate_metadata,
        }
        self.model.fit(
            Xs=[self.X], Ys=[self.Y], Yvars=[self.Yvar], **fit_update_shared_kwargs
        )

        for refit_on_update, warm_start_refit in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.model.refit_on_update = refit_on_update
            self.model.warm_start_refit = warm_start_refit
            self.model.update(
                Xs=[self.X], Ys=[self.Y], Yvars=[self.Yvar], **fit_update_shared_kwargs
            )
            expected_state_dict = (
                None
                if refit_on_update and not warm_start_refit
                else self.model.surrogate.model.state_dict()
            )
            # Check correct training data and `fit_update_shared_kwargs` values (can't
            # directly use `assert_called_with` due to tensor comparison ambiguity in
            # state dict).
            self.assertEqual(
                mock_update.call_args_list[-1][1].get("training_data"),
                self.training_data,
            )
            for key in fit_update_shared_kwargs:
                self.assertEqual(
                    mock_update.call_args_list[-1][1].get(key),
                    fit_update_shared_kwargs.get(key),
                )

            # Check correct `refit` and `state_dict` values.
            self.assertEqual(
                mock_update.call_args_list[-1][1].get("refit"), refit_on_update
            )
            if expected_state_dict is None:
                self.assertIsNone(
                    mock_update.call_args_list[-1][1].get("state_dict"),
                    expected_state_dict,
                )
            else:
                self.assertEqual(
                    mock_update.call_args_list[-1][1].get("state_dict").keys(),
                    expected_state_dict.keys(),
                )

    @patch(f"{SURROGATE_PATH}.Surrogate.predict")
    def test_predict(self, mock_predict):
        self.model.predict(X=self.X)
        mock_predict.assert_called_with(X=self.X)

    @patch(f"{MODEL_PATH}.BoTorchModel.fit")
    def test_cross_validate(self, mock_fit):
        fit_cv_shared_kwargs = {
            "bounds": self.bounds,
            "task_features": self.task_features,
            "feature_names": self.feature_names,
            "metric_names": self.metric_names,
            "fidelity_features": self.fidelity_features,
        }
        self.model.fit(
            Xs=[self.X],
            Ys=[self.Y],
            Yvars=[self.Yvar],
            candidate_metadata=self.candidate_metadata,
            **fit_cv_shared_kwargs,
        )

        old_surrogate = self.model.surrogate
        old_surrogate._model = MagicMock()
        old_surrogate._model.state_dict.return_value = {"key": "val"}

        for refit_on_cv, warm_start_refit in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.model.refit_on_cv = refit_on_cv
            self.model.warm_start_refit = warm_start_refit
            with patch(
                f"{SURROGATE_PATH}.Surrogate.clone_reset",
                return_value=MagicMock(spec=Surrogate),
            ) as mock_clone_reset:
                self.model.cross_validate(
                    Xs_train=[self.X],
                    Ys_train=[self.Y],
                    Yvars_train=[self.Yvar],
                    X_test=self.X2,
                    **fit_cv_shared_kwargs,
                )
                # Check that `predict` is called on the cloned surrogate, not
                # on the original one.
                mock_predict = mock_clone_reset.return_value.predict
                mock_predict.assert_called_once()

                # Check correct X_test.
                self.assertTrue(
                    torch.equal(
                        mock_predict.call_args_list[-1][1].get("X"),
                        self.X2,
                    ),
                )

            # Check that surrogate is reset back to `old_surrogate` at the
            # end of cross-validation.
            self.model.surrogate is old_surrogate

            expected_state_dict = (
                None
                if refit_on_cv and not warm_start_refit
                else self.model.surrogate.model.state_dict()
            )

            # Check correct `refit` and `state_dict` values.
            self.assertEqual(mock_fit.call_args_list[-1][1].get("refit"), refit_on_cv)
            if expected_state_dict is None:
                self.assertIsNone(
                    mock_fit.call_args_list[-1][1].get("state_dict"),
                    expected_state_dict,
                )
            else:
                self.assertEqual(
                    mock_fit.call_args_list[-1][1].get("state_dict").keys(),
                    expected_state_dict.keys(),
                )

    @patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @patch(f"{CURRENT_PATH}.KnowledgeGradient")
    @patch(f"{MODEL_PATH}.get_rounding_func", return_value="func")
    @patch(f"{MODEL_PATH}._to_inequality_constraints", return_value=[])
    @patch(f"{MODEL_PATH}.choose_botorch_acqf_class", return_value=qKnowledgeGradient)
    def test_gen(
        self,
        mock_choose_botorch_acqf_class,
        mock_inequality_constraints,
        mock_rounding,
        mock_kg,
        mock_construct_options,
    ):
        mock_kg.return_value.optimize.return_value = (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=KnowledgeGradient,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        model._botorch_acqf_class = None
        model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            rounding_func=self.rounding_func,
            target_fidelities=self.target_fidelities,
        )
        # Assert `construct_acquisition_and_optimizer_options` called with kwargs
        mock_construct_options.assert_called_with(
            acqf_options=self.acquisition_options,
            model_gen_options=self.model_gen_options,
        )
        # Assert `choose_botorch_acqf_class` is called
        mock_choose_botorch_acqf_class.assert_called_once()
        self.assertEqual(model._botorch_acqf_class, qKnowledgeGradient)
        # Assert `acquisition_class` called with kwargs
        mock_kg.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=model.botorch_acqf_class,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            target_fidelities=self.target_fidelities,
            options=self.acquisition_options,
        )
        # Assert `optimize` called with kwargs
        mock_kg.return_value.optimize.assert_called_with(
            bounds=ANY,
            n=1,
            inequality_constraints=[],
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )

    def test_best_point(self):
        with self.assertRaises(NotImplementedError):
            self.model.best_point(
                bounds=self.bounds, objective_weights=self.objective_weights
            )

    @patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @patch(f"{CURRENT_PATH}.KnowledgeGradient", autospec=True)
    def test_evaluate_acquisition_function(self, _mock_kg, _mock_construct_options):
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=KnowledgeGradient,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        model.evaluate_acquisition_function(
            X=self.X,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            acq_options=self.acquisition_options,
            target_fidelities=self.target_fidelities,
        )
        # `_mock_kg` is a mock of class, so to check the mock `evaluate` on
        # instance of that class, we use `_mock_kg.return_value.evaluate`
        _mock_kg.return_value.evaluate.assert_called()

    @patch(f"{LIST_SURROGATE_PATH}.ListSurrogate.__init__", return_value=None)
    @patch(f"{LIST_SURROGATE_PATH}.ListSurrogate.fit", return_value=None)
    def test_surrogate_options_propagation(self, _, mock_init):
        model = BoTorchModel(surrogate_options={"some_option": "some_value"})
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names_for_list_surrogate,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
        )
        mock_init.assert_called_with(
            botorch_submodel_class_per_outcome={
                outcome: FixedNoiseMultiFidelityGP
                for outcome in self.metric_names_for_list_surrogate
            },
            some_option="some_value",
        )

    @patch(
        f"{ACQUISITION_PATH}.Acquisition._extract_training_data",
        # Mock to register calls, but still execute the function.
        side_effect=Acquisition._extract_training_data,
    )
    @patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_list_surrogate_choice(self, _, mock_extract_training_data):
        model = BoTorchModel()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names_for_list_surrogate,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
        )
        # A list surrogate should be chosen, since Xs are not all the same.
        self.assertIsInstance(model.surrogate.model, ModelListGP)
        for submodel in model.surrogate.model.models:
            # There are fidelity features and nonempty Yvars, so
            # fixed noise MFGP should be chosen.
            self.assertIsInstance(submodel, FixedNoiseMultiFidelityGP)
        model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            rounding_func=self.rounding_func,
            target_fidelities=self.target_fidelities,
        )
        mock_extract_training_data.assert_called_once()
        self.assertIsInstance(
            mock_extract_training_data.call_args[1]["surrogate"], ListSurrogate
        )
