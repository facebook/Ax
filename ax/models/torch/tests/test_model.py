#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.models.torch.utils import _filter_X_observed
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.input_constructors import (
    get_acqf_input_constructor,
    _register_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.gp_regression import SingleTaskGP, FixedNoiseGP
from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.containers import TrainingData


CURRENT_PATH = __name__
MODEL_PATH = BoTorchModel.__module__
SURROGATE_PATH = Surrogate.__module__
UTILS_PATH = choose_model_class.__module__
ACQUISITION_PATH = Acquisition.__module__
LIST_SURROGATE_PATH = ListSurrogate.__module__
NEHVI_PATH = qNoisyExpectedHypervolumeImprovement.__module__

ACQ_OPTIONS = {Keys.SAMPLER: SobolQMCNormalSampler(1024)}


class BoTorchModelTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.acquisition_class = Acquisition
        self.botorch_acqf_class = qExpectedImprovement
        self.acquisition_options = ACQ_OPTIONS
        self.model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            botorch_acqf_class=self.botorch_acqf_class,
            acquisition_options=self.acquisition_options,
        )

        self.dtype = torch.float
        self.device = torch.device("cpu")
        tkwargs = {"dtype": self.dtype, "device": self.device}
        Xs1, Ys1, Yvars1, self.bounds, _, _, _ = get_torch_test_data(dtype=self.dtype)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(dtype=self.dtype, offset=1.0)
        self.Xs = Xs1
        self.Ys = Ys1
        self.Yvars = Yvars1
        self.X_test = Xs2[0]
        self.block_design_training_data = TrainingData(
            Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars
        )
        self.non_block_design_training_data = TrainingData(
            Xs=Xs1 + Xs2,
            Ys=Ys1 + Ys2,
            Yvars=Yvars1 + Yvars2,
        )
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2", "x3"],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        )
        self.mf_search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2", "x3"],
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            task_features=[],
            fidelity_features=[2],
            target_fidelities={1: 1.0},
        )
        self.metric_names = ["y"]
        self.metric_names_for_list_surrogate = ["y1", "y2"]
        self.candidate_metadata = []
        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.model_gen_options = {Keys.OPTIMIZER_KWARGS: self.optimizer_options}
        self.objective_weights = torch.tensor([1.0], **tkwargs)
        self.moo_objective_weights = torch.tensor([1.0, 1.5, 0.0], **tkwargs)
        self.moo_objective_thresholds = torch.tensor(
            [0.5, 1.5, float("nan")], **tkwargs
        )
        self.outcome_constraints = None
        self.linear_constraints = None
        self.fixed_features = None
        self.pending_observations = None
        self.rounding_func = "func"
        self.moo_training_data = TrainingData(
            Xs=self.Xs * 3,
            Ys=self.non_block_design_training_data.Ys + self.Ys,
            Yvars=self.Yvars * 3,
        )
        self.moo_metric_names = ["y1", "y2", "y3"]

    def test_init(self):
        # Default model with no specifications.
        model = BoTorchModel()
        self.assertEqual(model.acquisition_class, Acquisition)
        # Model that specifies `botorch_acqf_class`.
        model = BoTorchModel(botorch_acqf_class=qExpectedImprovement)
        self.assertEqual(model.acquisition_class, Acquisition)
        self.assertEqual(model.botorch_acqf_class, qExpectedImprovement)

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
            ValueError, "BoTorch `AcquisitionFunction` has not yet been set."
        ):
            self.model.botorch_acqf_class

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.fit")
    @mock.patch(f"{MODEL_PATH}.choose_model_class", return_value=SingleTaskGP)
    def test_fit(self, mock_choose_model_class, mock_fit):
        # If surrogate is not yet set, initialize it with dispatcher functions.
        self.model._surrogate = None
        self.model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            search_space_digest=self.mf_search_space_digest,
            metric_names=self.metric_names,
            candidate_metadata=self.candidate_metadata,
        )
        # `choose_model_class` is called.
        mock_choose_model_class.assert_called_with(
            Yvars=self.Yvars,
            search_space_digest=self.mf_search_space_digest,
        )
        # Since we want to refit on updates but not warm start refit, we clear the
        # state dict.
        mock_fit.assert_called_with(
            training_data=self.block_design_training_data,
            search_space_digest=self.mf_search_space_digest,
            metric_names=self.metric_names,
            candidate_metadata=self.candidate_metadata,
            state_dict=None,
            refit=True,
        )

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.update")
    def test_update(self, mock_update):
        self.model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            search_space_digest=self.mf_search_space_digest,
            metric_names=self.metric_names,
            candidate_metadata=self.candidate_metadata,
        )
        for refit_on_update, warm_start_refit in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.model.refit_on_update = refit_on_update
            self.model.warm_start_refit = warm_start_refit
            self.model.update(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                search_space_digest=self.mf_search_space_digest,
                metric_names=self.metric_names,
                candidate_metadata=self.candidate_metadata,
            )
            expected_state_dict = (
                None
                if refit_on_update and not warm_start_refit
                else self.model.surrogate.model.state_dict()
            )

            # Check for correct call args
            call_args = mock_update.call_args_list[-1][1]
            self.assertEqual(
                call_args.get("training_data"), self.block_design_training_data
            )
            self.assertEqual(
                call_args.get("search_space_digest"), self.mf_search_space_digest
            )
            self.assertEqual(call_args.get("metric_names"), self.metric_names)
            self.assertEqual(
                call_args.get("candidate_metadata"), self.candidate_metadata
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

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.predict")
    def test_predict(self, mock_predict):
        self.model.predict(X=self.X_test)
        mock_predict.assert_called_with(X=self.X_test)

    @mock.patch(f"{MODEL_PATH}.BoTorchModel.fit")
    def test_cross_validate(self, mock_fit):
        self.model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
            metric_names=self.metric_names,
        )

        old_surrogate = self.model.surrogate
        old_surrogate._model = mock.MagicMock()
        old_surrogate._model.state_dict.return_value = {"key": "val"}

        for refit_on_cv, warm_start_refit in [
            (True, True),
            (True, False),
            (False, True),
        ]:
            self.model.refit_on_cv = refit_on_cv
            self.model.warm_start_refit = warm_start_refit
            with mock.patch(
                f"{SURROGATE_PATH}.Surrogate.clone_reset",
                return_value=mock.MagicMock(spec=Surrogate),
            ) as mock_clone_reset:
                self.model.cross_validate(
                    Xs_train=self.Xs,
                    Ys_train=self.Ys,
                    Yvars_train=self.Yvars,
                    X_test=self.X_test,
                    search_space_digest=self.mf_search_space_digest,
                    metric_names=self.metric_names,
                )
                # Check that `predict` is called on the cloned surrogate, not
                # on the original one.
                mock_predict = mock_clone_reset.return_value.predict
                mock_predict.assert_called_once()

                # Check correct X_test.
                self.assertTrue(
                    torch.equal(
                        mock_predict.call_args_list[-1][1].get("X"),
                        self.X_test,
                    ),
                )

            # Check that surrogate is reset back to `old_surrogate` at the
            # end of cross-validation.
            self.assertTrue(self.model.surrogate is old_surrogate)

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

    @mock.patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=(
            ACQ_OPTIONS,
            {"num_restarts": 40, "raw_samples": 1024},
        ),
    )
    @mock.patch(f"{CURRENT_PATH}.Acquisition")
    @mock.patch(f"{MODEL_PATH}.get_rounding_func", return_value="func")
    @mock.patch(f"{MODEL_PATH}._to_inequality_constraints", return_value=[])
    @mock.patch(
        f"{MODEL_PATH}.choose_botorch_acqf_class", return_value=qExpectedImprovement
    )
    def test_gen(
        self,
        mock_choose_botorch_acqf_class,
        mock_inequality_constraints,
        mock_rounding,
        mock_acquisition,
        mock_construct_options,
    ):
        mock_acquisition.return_value.optimize.return_value = (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.block_design_training_data,
            fidelity_features=self.mf_search_space_digest.fidelity_features,
        )
        model._botorch_acqf_class = None
        # Assert that error is raised if we haven't fit the model
        with self.assertRaises(RuntimeError):
            model.gen(
                n=1,
                bounds=self.mf_search_space_digest.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                pending_observations=self.pending_observations,
                model_gen_options=self.model_gen_options,
                rounding_func=self.rounding_func,
                target_fidelities=self.mf_search_space_digest.target_fidelities,
            )
        # Add search space digest reference to make the model think it's been fit
        model._search_space_digest = self.mf_search_space_digest
        model.gen(
            n=1,
            bounds=self.mf_search_space_digest.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            rounding_func=self.rounding_func,
            target_fidelities=self.mf_search_space_digest.target_fidelities,
        )

        # Assert `construct_acquisition_and_optimizer_options` called with kwargs
        mock_construct_options.assert_called_with(
            acqf_options=self.acquisition_options,
            model_gen_options=self.model_gen_options,
        )
        # Assert `choose_botorch_acqf_class` is called
        mock_choose_botorch_acqf_class.assert_called_once()
        self.assertEqual(model._botorch_acqf_class, qExpectedImprovement)
        # Assert `acquisition_class` called with kwargs
        mock_acquisition.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=model.botorch_acqf_class,
            search_space_digest=self.mf_search_space_digest,
            objective_weights=self.objective_weights,
            objective_thresholds=None,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            options=self.acquisition_options,
        )
        # Assert `optimize` called with kwargs
        mock_acquisition.return_value.optimize.assert_called_with(
            n=1,
            search_space_digest=self.mf_search_space_digest,
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

    @mock.patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @mock.patch(f"{CURRENT_PATH}.Acquisition", autospec=True)
    def test_evaluate_acquisition_function(self, mock_ei, _mock_construct_options):
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.block_design_training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=[],
                bounds=[],
                fidelity_features=self.mf_search_space_digest.fidelity_features,
            ),
        )
        model.evaluate_acquisition_function(
            X=self.X_test,
            search_space_digest=self.mf_search_space_digest,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            acq_options=self.acquisition_options,
        )
        # `mock_ei` is a mock of class, so to check the mock `evaluate` on
        # instance of that class, we use `mock_ei.return_value.evaluate`
        mock_ei.return_value.evaluate.assert_called()

    @mock.patch(f"{LIST_SURROGATE_PATH}.ListSurrogate.__init__", return_value=None)
    @mock.patch(f"{LIST_SURROGATE_PATH}.ListSurrogate.fit", return_value=None)
    def test_surrogate_options_propagation(self, _, mock_init):
        model = BoTorchModel(surrogate_options={"some_option": "some_value"})
        model.fit(
            Xs=self.non_block_design_training_data.Xs,
            Ys=self.non_block_design_training_data.Ys,
            Yvars=self.non_block_design_training_data.Yvars,
            search_space_digest=self.mf_search_space_digest,
            metric_names=self.metric_names_for_list_surrogate,
            candidate_metadata=self.candidate_metadata,
        )
        mock_init.assert_called_with(
            botorch_submodel_class_per_outcome={
                outcome: FixedNoiseMultiFidelityGP
                for outcome in self.metric_names_for_list_surrogate
            },
            some_option="some_value",
        )

    @mock.patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_list_surrogate_choice(self, _):  # , mock_extract_training_data):
        model = BoTorchModel()
        model.fit(
            Xs=self.non_block_design_training_data.Xs,
            Ys=self.non_block_design_training_data.Ys,
            Yvars=self.non_block_design_training_data.Yvars,
            search_space_digest=self.mf_search_space_digest,
            metric_names=self.metric_names_for_list_surrogate,
            candidate_metadata=self.candidate_metadata,
        )
        # A list surrogate should be chosen, since Xs are not all the same.
        self.assertIsInstance(model.surrogate, ListSurrogate)
        self.assertIsInstance(model.surrogate.model, ModelListGP)
        for submodel in model.surrogate.model.models:
            # There are fidelity features and nonempty Yvars, so
            # fixed noise MFGP should be chosen.
            self.assertIsInstance(submodel, FixedNoiseMultiFidelityGP)

    @mock.patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_MOO(self, _):
        # Add mock for qNEHVI input constructor to catch arguments passed to it.
        qNEHVI_input_constructor = get_acqf_input_constructor(
            qNoisyExpectedHypervolumeImprovement
        )
        mock_input_constructor = mock.MagicMock(
            qNEHVI_input_constructor, side_effect=qNEHVI_input_constructor
        )
        _register_acqf_input_constructor(
            acqf_cls=qNoisyExpectedHypervolumeImprovement,
            input_constructor=mock_input_constructor,
        )

        model = BoTorchModel()
        model.fit(
            Xs=self.moo_training_data.Xs,
            Ys=self.moo_training_data.Ys,
            Yvars=self.moo_training_data.Yvars,
            search_space_digest=self.search_space_digest,
            metric_names=self.moo_metric_names,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertIsInstance(model.surrogate.model, FixedNoiseGP)
        _, _, gen_metadata, _ = model.gen(
            n=1,
            bounds=self.search_space_digest.bounds,
            objective_weights=self.moo_objective_weights,
            objective_thresholds=self.moo_objective_thresholds,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            rounding_func=self.rounding_func,
            target_fidelities=self.mf_search_space_digest.target_fidelities,
        )
        ckwargs = mock_input_constructor.call_args[1]
        self.assertIs(model.botorch_acqf_class, qNoisyExpectedHypervolumeImprovement)
        mock_input_constructor.assert_called_once()
        m = ckwargs["model"]
        self.assertIsInstance(m, FixedNoiseGP)
        self.assertEqual(m.num_outputs, 2)
        training_data = ckwargs["training_data"]
        for attr in ("Xs", "Ys", "Yvars"):
            self.assertTrue(
                all(
                    torch.equal(x1, x2)
                    for x1, x2 in zip(
                        getattr(training_data, attr),
                        getattr(self.moo_training_data, attr),
                    )
                )
            )
        self.assertTrue(
            torch.equal(
                ckwargs["objective_thresholds"], self.moo_objective_thresholds[:2]
            )
        )
        self.assertIsNone(
            ckwargs["outcome_constraints"],
        )
        self.assertIsNone(
            ckwargs["X_pending"],
        )
        obj_t = gen_metadata["objective_thresholds"]
        self.assertTrue(torch.equal(obj_t[:2], self.moo_objective_thresholds[:2]))
        self.assertTrue(np.isnan(obj_t[2].item()))

        self.assertIsInstance(
            ckwargs.get("objective"),
            WeightedMCMultiOutputObjective,
        )
        self.assertTrue(
            torch.equal(
                mock_input_constructor.call_args[1].get("objective").weights,
                self.moo_objective_weights[:2],
            )
        )
        expected_X_baseline = _filter_X_observed(
            Xs=self.moo_training_data.Xs,
            objective_weights=self.moo_objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=self.search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.assertTrue(
            torch.equal(
                mock_input_constructor.call_args[1].get("X_baseline"),
                expected_X_baseline,
            )
        )
        # test inferred objective_thresholds
        with ExitStack() as es:
            _mock_model_infer_objective_thresholds = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_modular.acquisition."
                    "infer_objective_thresholds",
                    return_value=torch.tensor([9.9, 3.3, float("nan")]),
                )
            )

            objective_weights = torch.tensor([-1.0, -1.0, 0.0])
            outcome_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[10.0]]),
            )
            linear_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[2.0]]),
            )
            _, _, gen_metadata, _ = model.gen(
                n=1,
                bounds=self.search_space_digest.bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=self.fixed_features,
                pending_observations=self.pending_observations,
                model_gen_options=self.model_gen_options,
                rounding_func=self.rounding_func,
                target_fidelities=self.mf_search_space_digest.target_fidelities,
            )
            expected_X_baseline = _filter_X_observed(
                Xs=self.moo_training_data.Xs,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                bounds=self.search_space_digest.bounds,
                linear_constraints=linear_constraints,
                fixed_features=self.fixed_features,
            )
            ckwargs = _mock_model_infer_objective_thresholds.call_args[1]
            self.assertTrue(
                torch.equal(
                    ckwargs["objective_weights"],
                    objective_weights,
                )
            )
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], outcome_constraints[0]))
            self.assertTrue(torch.equal(oc[1], outcome_constraints[1]))
            m = ckwargs["model"]
            self.assertIsInstance(m, FixedNoiseGP)
            self.assertEqual(m.num_outputs, 2)
            self.assertIn("objective_thresholds", gen_metadata)
            obj_t = gen_metadata["objective_thresholds"]
            self.assertTrue(torch.equal(obj_t[:2], torch.tensor([9.9, 3.3])))
            self.assertTrue(np.isnan(obj_t[2].item()))

        # Avoid polluting the registry for other tests; re-register correct input
        # contructor for qNEHVI.
        _register_acqf_input_constructor(
            acqf_cls=qNoisyExpectedHypervolumeImprovement,
            input_constructor=qNEHVI_input_constructor,
        )
