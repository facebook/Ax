#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import warnings
from contextlib import ExitStack
from typing import Dict
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.model import BoTorchModel, SurrogateSpec
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.models.torch.utils import _filter_X_observed
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.input_constructors import (
    _register_acqf_input_constructor,
    get_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import FixedNoiseMultiFidelityGP
from botorch.models.model import ModelList
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


CURRENT_PATH: str = __name__
MODEL_PATH: str = BoTorchModel.__module__
SURROGATE_PATH: str = Surrogate.__module__
UTILS_PATH: str = choose_model_class.__module__
ACQUISITION_PATH: str = Acquisition.__module__
NEHVI_PATH: str = qNoisyExpectedHypervolumeImprovement.__module__

ACQ_OPTIONS: Dict[str, SobolQMCNormalSampler] = {
    Keys.SAMPLER: SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
}


class BoTorchModelTest(TestCase):
    def setUp(self) -> None:
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
        self.block_design_training_data = [
            SupervisedDataset(X=X, Y=Y) for X, Y in zip(Xs1, Ys1)
        ]
        self.non_block_design_training_data = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(Xs1 + Xs2, Ys1 + Ys2, Yvars1 + Yvars2)
        ]
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
        self.moo_training_data = [  # block design
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(Xs1 * 3, Ys1 + Ys2 + Ys1, Yvars1 * 3)
        ]
        self.moo_metric_names = ["y1", "y2", "y3"]

        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            # pyre-fixme[6]: For 7th param expected
            #  `Optional[typing.Callable[[Tensor], Tensor]]` but got `str`.
            rounding_func=self.rounding_func,
        )
        self.moo_torch_opt_config = dataclasses.replace(
            self.torch_opt_config,
            objective_weights=self.moo_objective_weights,
            objective_thresholds=self.moo_objective_thresholds,
        )

    def test_init(self) -> None:
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

    def test_surrogates_property(self) -> None:
        self.assertEqual(self.surrogate, list(self.model.surrogates.values())[0])

    def test_Xs_property(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        self.assertEqual(len(self.model.Xs), 1)
        self.assertTrue(
            self.model.Xs[0].equal(torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
        )

        with self.assertRaisesRegex(NotImplementedError, "Xs not implemented"):
            self.model._surrogates = {"foo": Surrogate(), "bar": Surrogate()}
            self.model.Xs

    def test_dtype(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertEqual(self.model.dtype, torch.float32)

    def test_device(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertEqual(self.model.device, torch.device("cpu"))

    def test_botorch_acqf_class_property(self) -> None:
        self.assertEqual(self.botorch_acqf_class, self.model.botorch_acqf_class)
        self.model._botorch_acqf_class = None
        with self.assertRaisesRegex(
            ValueError, "BoTorch `AcquisitionFunction` has not yet been set."
        ):
            self.model.botorch_acqf_class

    @mock.patch(f"{SURROGATE_PATH}.choose_model_class", return_value=SingleTaskGP)
    @mock.patch(f"{SURROGATE_PATH}.use_model_list", return_value=False)
    def test_construct(self, _: Mock, mock_choose_model_class: Mock) -> None:
        # Ensure proper error is raised when mixing data w/ and w/o variance
        ds1, ds2 = self.non_block_design_training_data
        with self.assertRaisesRegex(
            UnsupportedError, "Cannot convert mixed data with and without variance"
        ):
            self.model.fit(
                datasets=[ds1, SupervisedDataset(X=ds2.X(), Y=ds2.Y())],
                metric_names=self.metric_names * 2,
                search_space_digest=self.mf_search_space_digest,
            )

        # Ensure non-block design data is converted with warnings
        ds = self.block_design_training_data[0]
        X1 = ds.X()
        X2 = torch.cat((X1[:1], torch.rand_like(X1[1:])))
        with warnings.catch_warnings(record=True) as ws:
            self.model.fit(
                datasets=[ds, SupervisedDataset(X=X2, Y=ds.Y())],
                metric_names=self.metric_names * 2,
                search_space_digest=self.mf_search_space_digest,
            )
        # pyre-fixme[6]: For 1st param expected `Iterable[object]` but got `bool`.
        self.assertTrue(any(issubclass(w.category, AxWarning)) for w in ws)
        self.assertTrue(
            any(
                "Forcing converion of data not complying to a block design"
                in str(w.message)
                for w in ws
            )
        )

        # Test autoset
        self.model._surrogates = {}
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        # `choose_model_class` is called.
        mock_choose_model_class.assert_called_with(
            datasets=self.block_design_training_data,
            search_space_digest=self.mf_search_space_digest,
        )

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.fit")
    def test_fit(self, mock_fit: Mock) -> None:
        # If surrogate is not yet set, initialize it with dispatcher functions.
        self.model._surrogates = {}
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        # Since we want to refit on updates but not warm start refit, we clear the
        # state dict.
        mock_fit.assert_called_with(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
            state_dict=None,
            refit=True,
        )
        # ensure that error is raised when len(metric_names) != len(datasets)
        with self.assertRaisesRegex(
            ValueError, "Length of datasets and metric_names must match"
        ):
            self.model.fit(
                datasets=self.block_design_training_data,
                metric_names=self.metric_names * 2,
                search_space_digest=self.mf_search_space_digest,
            )

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.update")
    def test_update(self, mock_update: Mock) -> None:
        # test assertion that model needs to be fit first
        empty_model = BoTorchModel()
        with self.assertRaisesRegex(
            UnsupportedError, "Cannot update model that has not been fitted."
        ):
            empty_model.update(
                datasets=self.block_design_training_data,
                metric_names=self.metric_names,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
        # test assertion that datasets cannot be None
        empty_model._surrogates = {"key": mock.Mock()}  # mock the Surrogates
        with self.assertRaisesRegex(
            UnsupportedError, "BoTorchModel.update requires data for all outcomes."
        ):
            empty_model.update(
                datasets=[None],
                metric_names=self.metric_names,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
        # fit model and test update
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
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
                datasets=self.block_design_training_data,
                metric_names=self.metric_names,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
            expected_state_dict = (
                None
                if refit_on_update and not warm_start_refit
                else self.model.surrogates[Keys.ONLY_SURROGATE].model.state_dict()
            )

            # Check for correct call args
            call_args = mock_update.call_args_list[-1][1]
            self.assertEqual(call_args.get("datasets"), self.block_design_training_data)
            self.assertEqual(call_args.get("metric_names"), self.metric_names)
            self.assertEqual(
                call_args.get("search_space_digest"), self.mf_search_space_digest
            )
            self.assertEqual(
                call_args.get("candidate_metadata"), self.candidate_metadata
            )

            # Check correct `refit` and `state_dict` values.
            self.assertEqual(
                mock_update.call_args_list[-1][1].get("refit"), refit_on_update
            )
            if expected_state_dict is None:
                self.assertIsNone(mock_update.call_args_list[-1][1].get("state_dict"))
            else:
                self.assertEqual(
                    mock_update.call_args_list[-1][1].get("state_dict").keys(),
                    expected_state_dict.keys(),
                )

        # Test with autoset surrogate.
        autoset_model = BoTorchModel(
            acquisition_class=self.acquisition_class,
            botorch_acqf_class=self.botorch_acqf_class,
            acquisition_options=self.acquisition_options,
        )
        autoset_model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        autoset_model.refit_on_update = True
        autoset_model.warm_start_refit = False
        autoset_model.update(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        # Check for correct call args
        call_args = mock_update.call_args_list[-1][1]
        self.assertEqual(call_args.get("datasets"), self.block_design_training_data)
        self.assertEqual(call_args.get("metric_names"), self.metric_names)
        self.assertEqual(
            call_args.get("search_space_digest"), self.mf_search_space_digest
        )
        self.assertEqual(call_args.get("candidate_metadata"), self.candidate_metadata)

        # Check correct `refit` and `state_dict` values.
        self.assertEqual(mock_update.call_args_list[-1][1].get("refit"), True)
        self.assertIsNone(mock_update.call_args_list[-1][1].get("state_dict"))

    @mock.patch(f"{SURROGATE_PATH}.Surrogate.predict")
    def test_predict(self, mock_predict: Mock) -> None:
        self.model.predict(X=self.X_test)
        mock_predict.assert_called_with(X=self.X_test)

    @mock.patch(f"{MODEL_PATH}.BoTorchModel.fit")
    def test_cross_validate(self, mock_fit: Mock) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        old_surrogate = self.model.surrogates[Keys.ONLY_SURROGATE]
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
                    datasets=self.block_design_training_data,
                    metric_names=self.metric_names,
                    X_test=self.X_test,
                    search_space_digest=self.mf_search_space_digest,
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
            self.assertTrue(self.model.surrogates[Keys.ONLY_SURROGATE] is old_surrogate)

            expected_state_dict = (
                None
                if refit_on_cv and not warm_start_refit
                else self.model.surrogates[Keys.ONLY_SURROGATE].model.state_dict()
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
                    mock_fit.call_args_list[-1][1]
                    .get("state_dicts")
                    .get(Keys.ONLY_SURROGATE)
                    .keys(),
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
        mock_choose_botorch_acqf_class: Mock,
        mock_inequality_constraints: Mock,
        mock_rounding: Mock,
        mock_acquisition: Mock,
        mock_construct_options: Mock,
    ) -> None:
        mock_acquisition.return_value.optimize.return_value = (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        model.surrogates[Keys.ONLY_SURROGATE].construct(
            datasets=self.block_design_training_data,
            metric_names=["metric"],
            fidelity_features=self.mf_search_space_digest.fidelity_features,
        )
        model._botorch_acqf_class = None
        # Assert that error is raised if we haven't fit the model
        with self.assertRaises(RuntimeError):
            model.gen(
                n=1,
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )
        # Add search space digest reference to make the model think it's been fit
        model._search_space_digest = self.mf_search_space_digest
        model.gen(
            n=1,
            search_space_digest=self.mf_search_space_digest,
            torch_opt_config=self.torch_opt_config,
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
            surrogates={Keys.ONLY_SURROGATE: self.surrogate},
            botorch_acqf_class=model.botorch_acqf_class,
            search_space_digest=self.mf_search_space_digest,
            torch_opt_config=self.torch_opt_config,
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

    @fast_botorch_optimize
    def test_best_point(self) -> None:
        self.model._surrogates = {}
        self.model.fit(
            datasets=self.block_design_training_data,
            metric_names=self.metric_names,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertIsNotNone(
            self.model.best_point(
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )
        )
        with mock.patch(f"{SURROGATE_PATH}.best_in_sample_point", return_value=None):
            self.assertIsNone(
                self.model.best_point(
                    search_space_digest=self.mf_search_space_digest,
                    torch_opt_config=self.torch_opt_config,
                )
            )
        with self.assertRaisesRegex(NotImplementedError, "Best observed"):
            self.model.best_point(
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=dataclasses.replace(
                    self.torch_opt_config, is_moo=True
                ),
            )

    @mock.patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @mock.patch(f"{CURRENT_PATH}.Acquisition", autospec=True)
    def test_evaluate_acquisition_function(
        self, mock_ei: Mock, _mock_construct_options: Mock
    ) -> None:
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        model.surrogates[Keys.ONLY_SURROGATE].construct(
            datasets=self.block_design_training_data,
            metric_names=["metric"],
            search_space_digest=SearchSpaceDigest(
                feature_names=[],
                bounds=[],
                fidelity_features=self.mf_search_space_digest.fidelity_features,
            ),
        )
        model.evaluate_acquisition_function(
            X=self.X_test,
            search_space_digest=self.mf_search_space_digest,
            torch_opt_config=self.torch_opt_config,
            acq_options=self.acquisition_options,
        )
        # `mock_ei` is a mock of class, so to check the mock `evaluate` on
        # instance of that class, we use `mock_ei.return_value.evaluate`
        mock_ei.return_value.evaluate.assert_called()

    @mock.patch(f"{MODEL_PATH}.Surrogate.__init__", return_value=None)
    @mock.patch(f"{SURROGATE_PATH}.Surrogate.fit", return_value=None)
    def test_surrogate_options_propagation(self, _: Mock, mock_init: Mock) -> None:
        model = BoTorchModel(
            surrogate_specs={
                "name": SurrogateSpec(
                    botorch_model_kwargs={"some_option": "some_value"}
                )
            }
        )
        model.fit(
            datasets=self.non_block_design_training_data,
            metric_names=self.metric_names_for_list_surrogate,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        mock_init.assert_called_with(
            botorch_model_class=None,
            model_options={"some_option": "some_value"},
            mll_class=ExactMarginalLogLikelihood,
            mll_options={},
            covar_module_class=None,
            covar_module_options=None,
            likelihood_class=None,
            likelihood_options=None,
            input_transform=None,
            outcome_transform=None,
        )

    @mock.patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_model_list_choice(self, _) -> None:  # , mock_extract_training_data):
        model = BoTorchModel()
        model.fit(
            datasets=self.non_block_design_training_data,
            metric_names=self.metric_names_for_list_surrogate,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        # A model list should be chosen, since Xs are not all the same.
        model_list = checked_cast(
            ModelList, model.surrogates[Keys.AUTOSET_SURROGATE].model
        )
        for submodel in model_list.models:
            # There are fidelity features and nonempty Yvars, so
            # fixed noise MFGP should be chosen.
            self.assertIsInstance(submodel, FixedNoiseMultiFidelityGP)

    @mock.patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_MOO(self, _) -> None:
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
            datasets=self.moo_training_data,
            metric_names=self.moo_metric_names,
            search_space_digest=self.search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertIsInstance(
            model.surrogates[Keys.AUTOSET_SURROGATE].model, FixedNoiseGP
        )
        gen_results = model.gen(
            n=1,
            search_space_digest=self.mf_search_space_digest,
            torch_opt_config=self.moo_torch_opt_config,
        )
        ckwargs = mock_input_constructor.call_args[1]
        self.assertIs(model.botorch_acqf_class, qNoisyExpectedHypervolumeImprovement)
        mock_input_constructor.assert_called_once()
        m = ckwargs["model"]
        self.assertIsInstance(m, FixedNoiseGP)
        self.assertEqual(m.num_outputs, 2)
        training_data = ckwargs["training_data"]
        self.assertIsInstance(training_data, FixedNoiseDataset)
        self.assertTrue(torch.equal(training_data.X(), self.Xs[0]))
        self.assertTrue(
            torch.equal(
                training_data.Y(),
                torch.cat([ds.Y() for ds in self.moo_training_data], dim=-1),
            )
        )
        self.assertTrue(
            torch.equal(
                training_data.Yvar(),
                torch.cat([ds.Yvar() for ds in self.moo_training_data], dim=-1),
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
        obj_t = gen_results.gen_metadata["objective_thresholds"]
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
            Xs=[dataset.X() for dataset in self.moo_training_data],
            objective_weights=self.moo_objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=self.search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.assertTrue(
            torch.equal(
                mock_input_constructor.call_args[1].get("X_baseline"),
                # pyre-fixme[6]: For 2nd param expected `Tensor` but got
                #  `Optional[Tensor]`.
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
            gen_results = model.gen(
                n=1,
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=dataclasses.replace(
                    self.moo_torch_opt_config,
                    objective_weights=objective_weights,
                    objective_thresholds=None,
                    outcome_constraints=outcome_constraints,
                    linear_constraints=linear_constraints,
                ),
            )
            expected_X_baseline = _filter_X_observed(
                Xs=[dataset.X() for dataset in self.moo_training_data],
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
            self.assertIn("objective_thresholds", gen_results.gen_metadata)
            obj_t = gen_results.gen_metadata["objective_thresholds"]
            self.assertTrue(torch.equal(obj_t[:2], torch.tensor([9.9, 3.3])))
            self.assertTrue(np.isnan(obj_t[2].item()))

        # Avoid polluting the registry for other tests; re-register correct input
        # contructor for qNEHVI.
        _register_acqf_input_constructor(
            acqf_cls=qNoisyExpectedHypervolumeImprovement,
            input_constructor=qNEHVI_input_constructor,
        )
