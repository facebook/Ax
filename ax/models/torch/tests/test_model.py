#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
from contextlib import ExitStack
from copy import deepcopy
from itertools import product
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.models.torch.botorch_modular.model import (
    BoTorchModel,
    choose_botorch_acqf_class,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.models.torch.botorch_modular.utils import (
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    fit_botorch_model,
    ModelConfig,
)
from ax.models.torch.utils import _filter_X_observed, predict_from_model
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.input_constructors import (
    _register_acqf_input_constructor,
    get_acqf_input_constructor,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.model import Model, ModelList
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.types import DEFAULT
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from pyre_extensions import assert_is_instance, none_throws


CURRENT_PATH: str = __name__
MODEL_PATH: str = BoTorchModel.__module__
SURROGATE_PATH: str = Surrogate.__module__
ACQUISITION_PATH: str = Acquisition.__module__

ACQ_OPTIONS: dict[str, SobolQMCNormalSampler] = {
    Keys.SAMPLER: SobolQMCNormalSampler(sample_shape=torch.Size([1024]))
}


class BoTorchModelTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.tkwargs = tkwargs
        (
            self.Xs,
            self.Ys,
            self.Yvars,
            self.bounds,
            _,
            self.feature_names,  # This is ["x1", "x2", "x3"].
            self.metric_names,  # This is just ["y"].
        ) = get_torch_test_data(dtype=self.dtype)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(dtype=self.dtype, offset=1.0)
        self.X_test = Xs2[0]
        self.block_design_training_data = [
            SupervisedDataset(
                X=self.Xs[0],
                Y=self.Ys[0],
                Yvar=self.Yvars[0],
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
        ]
        self.non_block_design_training_data = self.block_design_training_data + [
            SupervisedDataset(
                X=Xs2[0],
                Y=Ys2[0],
                Yvar=Yvars2[0],
                feature_names=self.feature_names,
                outcome_names=["y2"],
            )
        ]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        )
        self.mf_search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            fidelity_features=[2],
            target_values={1: 1.0},
        )
        self.candidate_metadata = []
        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.model_gen_options = {Keys.OPTIMIZER_KWARGS: self.optimizer_options}
        self.objective_weights = torch.tensor([1.0], **tkwargs)
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[-5.0]], **tkwargs),
        )
        self.moo_objective_weights = torch.tensor([1.0, 1.5, 0.0], **tkwargs)
        self.moo_objective_thresholds = torch.tensor(
            [0.5, 1.5, float("nan")], **tkwargs
        )
        self.moo_outcome_constraints = (
            torch.tensor([[1.0, 0.0, 0.0]], **tkwargs),
            torch.tensor([[-5.0]], **tkwargs),
        )
        self.linear_constraints = None
        self.fixed_features = None
        self.pending_observations = None
        self.moo_metric_names = ["y1", "y2", "y3"]
        self.moo_training_data = [  # block design
            SupervisedDataset(
                X=X,
                Y=Y,
                Yvar=Yvar,
                feature_names=self.feature_names,
                outcome_names=[mn],
            )
            for X, Y, Yvar, mn in zip(
                assert_is_instance(self.Xs, list) * 3,
                self.Ys + Ys2 + self.Ys,
                assert_is_instance(self.Yvars, list) * 3,
                self.moo_metric_names,
            )
        ]

        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
        )
        self.moo_torch_opt_config = dataclasses.replace(
            self.torch_opt_config,
            objective_weights=self.moo_objective_weights,
            objective_thresholds=self.moo_objective_thresholds,
            outcome_constraints=self.moo_outcome_constraints,
            is_moo=True,
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
        self.assertFalse(model.refit_on_cv)
        self.assertTrue(model.warm_start_refit)

        # Check setting non-default refitting settings
        mdl2 = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            acquisition_options=self.acquisition_options,
            refit_on_cv=True,
            warm_start_refit=False,
        )
        self.assertTrue(mdl2.refit_on_cv)
        self.assertFalse(mdl2.warm_start_refit)

    def test_surrogate_property(self) -> None:
        self.assertIs(self.surrogate, self.model.surrogate)

    def test_Xs_property(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        self.assertEqual(len(self.model.Xs), 1)
        self.assertTrue(
            self.model.Xs[0].equal(torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
        )

    def test_dtype(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertEqual(self.model.dtype, torch.float32)

    def test_device(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.search_space_digest,
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

    @mock.patch(f"{SURROGATE_PATH}.use_model_list", return_value=False)
    def test__construct__raises_on_mixed_data(self, _: Mock) -> None:
        """Ensure proper error is raised when mixing data w/ and w/o variance."""
        ds1, ds2 = self.non_block_design_training_data
        datasets = [
            ds1,
            SupervisedDataset(
                X=ds2.X,
                Y=ds2.Y,
                feature_names=ds2.feature_names,
                outcome_names=ds2.outcome_names,
            ),
        ]
        msg = (
            "Mix of known and unknown variances indicates valuation function"
            " errors. Variances should all be specified, or none should be."
        )
        with self.assertRaisesRegex(ValueError, msg):
            self.model.fit(
                datasets=datasets,
                search_space_digest=self.search_space_digest,
            )

    @mock.patch(f"{SURROGATE_PATH}.use_model_list", return_value=False)
    def test__construct__converts_non_block(self, _: Mock) -> None:
        """Ensure non-block design data is converted with warnings."""
        ds = self.block_design_training_data[0]
        X1 = ds.X
        X2 = torch.cat((X1[:1], torch.rand_like(X1[1:])))
        datasets = [
            ds,
            SupervisedDataset(
                X=X2,
                Y=ds.Y,
                Yvar=ds.Yvar,
                feature_names=ds.feature_names,
                outcome_names=["y2"],
            ),
        ]
        with self.assertWarnsRegex(
            AxWarning, "Forcing conversion of data not complying to a block design"
        ):
            self.model.fit(
                datasets=datasets,
                search_space_digest=self.search_space_digest,
            )

    def test__construct(self) -> None:
        """Test autoset."""
        self.model._surrogate = None
        with mock.patch(
            f"{SURROGATE_PATH}.choose_model_class", wraps=choose_model_class
        ) as mock_choose_model_class:
            self.model.fit(
                datasets=self.block_design_training_data,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
        # `choose_model_class` is called.
        mock_choose_model_class.assert_called_with(
            datasets=self.block_design_training_data,
            search_space_digest=self.mf_search_space_digest,
        )

    # This mock is hard to remove since it is mocks a method on a surrogate that
    # is only constructed when `model.fit` is called
    @mock.patch(f"{SURROGATE_PATH}.Surrogate._construct_model")
    def test_fit(self, mock_fit: Mock) -> None:
        # If surrogate is not yet set, initialize it with dispatcher functions.
        self.model._surrogate = None
        with self.assertRaisesRegex(RuntimeError, "is not initialized. Must `fit`"):
            self.model.search_space_digest  # can't access before fit

        with self.assertRaisesRegex(RuntimeError, "manually is disallowed"):
            self.model.search_space_digest = self.mf_search_space_digest

        self.model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        self.assertIsInstance(self.model.search_space_digest, SearchSpaceDigest)
        self.assertEqual(self.model.search_space_digest, self.mf_search_space_digest)

        # Since we want to refit on updates but not warm start refit, we clear the
        # state dict.
        mock_fit.assert_called_with(
            dataset=self.block_design_training_data[0],
            search_space_digest=self.mf_search_space_digest,
            model_config=ModelConfig(
                botorch_model_class=None,
                model_options={},
                mll_class=ExactMarginalLogLikelihood,
                mll_options={},
                input_transform_classes=DEFAULT,
                input_transform_options={},
                outcome_transform_classes=None,
                outcome_transform_options={},
                covar_module_class=None,
                covar_module_options={},
                likelihood_class=None,
                likelihood_options={},
                name="default",
            ),
            default_botorch_model_class=SingleTaskMultiFidelityGP,
            state_dict=None,
            refit=True,
        )

    def test_predict(self) -> None:
        with mock.patch.object(self.model._surrogate, "predict") as mock_predict:
            self.model.predict(X=self.X_test)
        mock_predict.assert_called_with(X=self.X_test, use_posterior_predictive=False)
        with mock.patch.object(self.model._surrogate, "predict") as mock_predict:
            self.model.predict(X=self.X_test, use_posterior_predictive=True)
        mock_predict.assert_called_with(X=self.X_test, use_posterior_predictive=True)

    def test_with_surrogate_specs_input(self) -> None:
        spec1 = SurrogateSpec(
            botorch_model_class=SingleTaskGP,
            outcomes=["y1", "y3"],
        )
        surrogate_specs = {
            "Vanilla": spec1,
            "Bayesian": SurrogateSpec(
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
                outcomes=["y2"],
            ),
        }
        with self.assertRaisesRegex(DeprecationWarning, "Support for multiple"):
            BoTorchModel(surrogate_specs=surrogate_specs)

        with self.assertWarnsRegex(DeprecationWarning, "surrogate_specs"):
            model = BoTorchModel(surrogate_specs={"s": spec1})
        self.assertIs(model.surrogate_spec, spec1)

    @mock_botorch_optimize
    def test_cross_validate(self) -> None:
        self.model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )

        old_surrogate = self.model.surrogate

        for refit_on_cv, warm_start_refit, use_posterior_predictive in product(
            (True, False), (True, False), (True, False)
        ):
            self.model.refit_on_cv = refit_on_cv
            self.model.warm_start_refit = warm_start_refit
            with ExitStack() as es:
                mock_fit = es.enter_context(
                    mock.patch.object(self.model, "fit", wraps=self.model.fit)
                )
                mock_predict_orig_surrogate = es.enter_context(
                    mock.patch.object(
                        self.model.surrogate,
                        "predict",
                        wraps=self.model.surrogate.predict,
                    )
                )
                mock_predict_any_surrogate = es.enter_context(
                    mock.patch(
                        f"{SURROGATE_PATH}.predict_from_model", wraps=predict_from_model
                    )
                )
                self.model.cross_validate(
                    datasets=self.block_design_training_data,
                    X_test=self.X_test,
                    search_space_digest=self.search_space_digest,
                    use_posterior_predictive=use_posterior_predictive,
                )
            # Check that `predict` is called on the cloned surrogate, not
            # on the original one.
            mock_predict_orig_surrogate.assert_not_called()
            mock_predict_any_surrogate.assert_called_once()

            # Check correct X_test.
            kwargs = mock_predict_any_surrogate.call_args.kwargs
            self.assertTrue(torch.equal(kwargs["X"], self.X_test))
            self.assertIs(kwargs["use_posterior_predictive"], use_posterior_predictive)

            # Check that surrogate is reset back to `old_surrogate` at the
            # end of cross-validation.
            self.assertIs(self.model.surrogate, old_surrogate)

            expected_state_dict = (
                None
                if refit_on_cv and not warm_start_refit
                else self.model.surrogate.model.state_dict()
            )

            # Check correct `refit` and `state_dict` values.
            kwargs = mock_fit.call_args.kwargs
            self.assertEqual(kwargs["refit"], refit_on_cv)
            if expected_state_dict is None:
                self.assertIsNone(kwargs["state_dict"], expected_state_dict)
            else:
                self.assertEqual(
                    kwargs["state_dict"].keys(), expected_state_dict.keys()
                )

    @mock_botorch_optimize
    def test_cross_validate_multiple_configs(self) -> None:
        """Test cross-validation with multiple configs."""
        for refit_on_cv in (True, False):
            with self.subTest(refit_on_cv=refit_on_cv):
                self.model = BoTorchModel(
                    surrogate_spec=SurrogateSpec(
                        model_configs=[
                            ModelConfig(),
                            ModelConfig(
                                botorch_model_class=SingleTaskGP,
                                covar_module_class=ScaleMaternKernel,
                            ),
                        ]
                    ),
                    acquisition_class=self.acquisition_class,
                    botorch_acqf_class=self.botorch_acqf_class,
                    acquisition_options=self.acquisition_options,
                    refit_on_cv=refit_on_cv,
                )
                self.model.fit(
                    datasets=self.block_design_training_data,
                    search_space_digest=self.search_space_digest,
                    candidate_metadata=self.candidate_metadata,
                )
                with patch(
                    f"{Surrogate.__module__}.fit_botorch_model", wraps=fit_botorch_model
                ) as mock_fit:
                    self.model.cross_validate(
                        datasets=self.block_design_training_data,
                        X_test=self.X_test,
                        search_space_digest=self.search_space_digest,
                    )
                # check that we don't fit the model during cross_validation
                if refit_on_cv:
                    mock_fit.assert_called()
                else:
                    mock_fit.assert_not_called()

    @mock_botorch_optimize
    @mock.patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        wraps=construct_acquisition_and_optimizer_options,
    )
    @mock.patch(
        f"{MODEL_PATH}.choose_botorch_acqf_class", wraps=choose_botorch_acqf_class
    )
    def _test_gen(
        self,
        mock_choose_botorch_acqf_class: Mock,
        mock_construct_options: Mock,
        botorch_model_class: type[Model],
        search_space_digest: SearchSpaceDigest,
    ) -> None:
        qLogNEI_input_constructor = get_acqf_input_constructor(
            qLogNoisyExpectedImprovement
        )
        mock_input_constructor = mock.MagicMock(
            qLogNEI_input_constructor, side_effect=qLogNEI_input_constructor
        )
        _register_acqf_input_constructor(
            acqf_cls=qLogNoisyExpectedImprovement,
            input_constructor=mock_input_constructor,
        )
        mock_optimize_return_value = (
            torch.tensor([[1.0]]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
        )
        surrogate = Surrogate(botorch_model_class=botorch_model_class)
        model = BoTorchModel(
            surrogate=surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        # Assert that error is raised if we haven't fit the model
        with self.assertRaises(RuntimeError):
            model.gen(
                n=1,
                search_space_digest=search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )
        model.fit(
            datasets=self.block_design_training_data,
            search_space_digest=search_space_digest,
        )
        with ExitStack() as es:
            mock_init_acqf = es.enter_context(
                mock.patch.object(
                    BoTorchModel,
                    "_instantiate_acquisition",
                    wraps=model._instantiate_acquisition,
                )
            )
            mock_optimize = es.enter_context(
                mock.patch(
                    f"{CURRENT_PATH}.Acquisition.optimize",
                    return_value=mock_optimize_return_value,
                )
            )

            gen_results = model.gen(
                n=1,
                search_space_digest=search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )
        self.assertEqual(
            gen_results.gen_metadata["metric_to_model_config_name"],
            {"y": "from deprecated args"},
        )
        # Assert acquisition initialized with expected arguments
        mock_init_acqf.assert_called_once_with(
            search_space_digest=search_space_digest,
            torch_opt_config=self.torch_opt_config,
            acq_options=self.acquisition_options,
        )

        mock_input_constructor.assert_called_once()
        ckwargs = mock_input_constructor.call_args[1]

        # We particularly want to make sure that args that will not be used are
        # not passed
        expected_kwargs = {
            "bounds",
            "constraints",
            "X_baseline",
            "sampler",
            "objective",
            "training_data",
            "model",
        }
        self.assertSetEqual(set(ckwargs.keys()), expected_kwargs)
        for k in expected_kwargs:
            self.assertIsNotNone(ckwargs[k], f"{k} is None")

        m = ckwargs["model"]
        self.assertIsInstance(m, botorch_model_class)
        self.assertEqual(m.num_outputs, 1)
        training_data = ckwargs["training_data"]
        self.assertIsInstance(training_data, SupervisedDataset)
        self.assertTrue(torch.equal(training_data.X, self.Xs[0]))
        self.assertTrue(
            torch.equal(
                training_data.Y,
                torch.cat([ds.Y for ds in self.block_design_training_data], dim=-1),
            )
        )

        self.assertIsInstance(ckwargs["objective"], GenericMCObjective)
        expected_X_baseline = _filter_X_observed(
            Xs=[dataset.X for dataset in self.block_design_training_data],
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            bounds=search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.assertTrue(
            torch.equal(ckwargs["X_baseline"], none_throws(expected_X_baseline))
        )

        # Assert `construct_acquisition_and_optimizer_options` called with kwargs
        mock_construct_options.assert_called_with(
            acqf_options=self.acquisition_options,
            model_gen_options=self.model_gen_options,
        )
        # Assert `choose_botorch_acqf_class` is called
        mock_choose_botorch_acqf_class.assert_called_once()
        self.assertEqual(model._botorch_acqf_class, qLogNoisyExpectedImprovement)

        # Assert `optimize` called with kwargs
        mock_optimize.assert_called_with(
            n=1,
            search_space_digest=search_space_digest,
            inequality_constraints=None,
            fixed_features=self.fixed_features,
            rounding_func=None,
            optimizer_options=self.optimizer_options,
        )

        _register_acqf_input_constructor(
            acqf_cls=qLogNoisyExpectedImprovement,
            input_constructor=qLogNEI_input_constructor,
        )

        # Make sure `gen` runs without mocking out Acquisition.optimize
        with self.subTest("No mocks"):
            gen_results = model.gen(
                n=1,
                search_space_digest=search_space_digest,
                torch_opt_config=self.torch_opt_config,
            )
            self.assertTrue(torch.isfinite(gen_results.points).all())

    def test_gen_SingleTaskGP(self) -> None:
        self._test_gen(
            botorch_model_class=SingleTaskGP,
            search_space_digest=self.search_space_digest,
        )

    def test_gen_SingleTaskMultiFidelityGP(self) -> None:
        self._test_gen(
            botorch_model_class=SingleTaskMultiFidelityGP,
            search_space_digest=self.mf_search_space_digest,
        )

    @mock_botorch_optimize
    def test_feature_importances(self) -> None:
        for botorch_model_class in [SingleTaskGP, SaasFullyBayesianSingleTaskGP]:
            surrogate = Surrogate(botorch_model_class=botorch_model_class)
            model = BoTorchModel(
                surrogate=surrogate,
                acquisition_class=Acquisition,
                acquisition_options=self.acquisition_options,
            )
            model.surrogate.fit(
                datasets=self.block_design_training_data,
                search_space_digest=self.search_space_digest,
            )
            if botorch_model_class == SaasFullyBayesianSingleTaskGP:
                mcmc_samples = {
                    "lengthscale": torch.tensor(
                        [[1, 2, 3], [2, 3, 4], [3, 4, 5]], **self.tkwargs
                    ),
                    "outputscale": torch.rand(3, **self.tkwargs),
                    "mean": torch.randn(3, **self.tkwargs),
                    "noise": torch.rand(3, **self.tkwargs),
                }
                # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
                model.surrogate.model.load_mcmc_samples(mcmc_samples)
                importances = model.feature_importances()
                self.assertTrue(
                    np.allclose(importances, np.array([6 / 13, 4 / 13, 3 / 13]))
                )
                self.assertEqual(importances.shape, (1, 1, 3))
                saas_model = deepcopy(model.surrogate.model)
            else:
                # pyre-fixme[16]: `Tensor` has no attribute `lengthscale`.
                # pyre-fixme[16]: `Module` has no attribute `lengthscale`.
                model.surrogate.model.covar_module.lengthscale = torch.tensor(
                    [1, 2, 3], **self.tkwargs
                )
                importances = model.feature_importances()
                self.assertTrue(
                    np.allclose(importances, np.array([6 / 11, 3 / 11, 2 / 11]))
                )
                self.assertEqual(importances.shape, (1, 1, 3))
                vanilla_model = deepcopy(model.surrogate.model)

        # Mixed model
        model.surrogate._model = ModelList(saas_model, vanilla_model)  # pyre-ignore
        importances = model.feature_importances()
        self.assertTrue(
            np.allclose(
                importances,
                np.expand_dims(
                    np.array([[6 / 13, 4 / 13, 3 / 13], [6 / 11, 3 / 11, 2 / 11]]),
                    axis=1,
                ),
            )
        )
        self.assertEqual(importances.shape, (2, 1, 3))
        # Add model we don't support
        vanilla_model.covar_module = None
        model.surrogate._model = vanilla_model  # pyre-ignore
        with self.assertRaisesRegex(
            NotImplementedError,
            "Failed to extract lengthscales from `m.covar_module` "
            "and `m.covar_module.base_kernel`",
        ):
            model.feature_importances()
        # Test model is None
        model.surrogate._model = None
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed"
        ):
            model.feature_importances()

    @mock_botorch_optimize
    def test_best_point(self) -> None:
        self.model._surrogate = None
        self.model.fit(
            datasets=self.block_design_training_data,
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

    @mock_botorch_optimize
    def test_evaluate_acquisition_function(self) -> None:
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=Acquisition,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.fit(
            datasets=self.block_design_training_data,
            search_space_digest=self.search_space_digest,
        )
        points = model.evaluate_acquisition_function(
            X=self.X_test,
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            acq_options=self.acquisition_options,
        )
        self.assertEqual(points.shape, torch.Size([1]))
        # testing that the new setup chooses qLogNEI by default
        self.assertEqual(model._botorch_acqf_class, qLogNoisyExpectedImprovement)

    @mock_botorch_optimize
    def test_surrogate_model_options_propagation(self) -> None:
        surrogate_spec = SurrogateSpec()
        model = BoTorchModel(surrogate_spec=surrogate_spec)
        with mock.patch(f"{MODEL_PATH}.Surrogate", wraps=Surrogate) as mock_init:
            model.fit(
                datasets=self.non_block_design_training_data,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
        mock_init.assert_called_with(surrogate_spec=surrogate_spec, refit_on_cv=False)

    @mock_botorch_optimize
    def test_surrogate_options_propagation(self) -> None:
        surrogate_spec = SurrogateSpec(allow_batched_models=False)
        model = BoTorchModel(surrogate_spec=surrogate_spec)
        with mock.patch(f"{MODEL_PATH}.Surrogate", wraps=Surrogate) as mock_init:
            model.fit(
                datasets=self.non_block_design_training_data,
                search_space_digest=self.mf_search_space_digest,
                candidate_metadata=self.candidate_metadata,
            )
        mock_init.assert_called_with(surrogate_spec=surrogate_spec, refit_on_cv=False)

    @mock_botorch_optimize
    def test_model_list_choice(self) -> None:
        model = BoTorchModel()
        model.fit(
            datasets=self.non_block_design_training_data,
            search_space_digest=self.mf_search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        # A model list should be chosen, since Xs are not all the same.
        model_list = assert_is_instance(model.surrogate.model, ModelList)
        for submodel in model_list.models:
            # There are fidelity features and nonempty Yvars, so
            # MFGP should be chosen.
            self.assertIsInstance(submodel, SingleTaskMultiFidelityGP)

    @mock_botorch_optimize
    def test_MOO(self) -> None:
        # Add mock for qLogNEHVI input constructor to catch arguments passed to it.
        qLogNEHVI_input_constructor = get_acqf_input_constructor(
            qLogNoisyExpectedHypervolumeImprovement
        )
        mock_input_constructor = mock.MagicMock(
            qLogNEHVI_input_constructor, side_effect=qLogNEHVI_input_constructor
        )
        _register_acqf_input_constructor(
            acqf_cls=qLogNoisyExpectedHypervolumeImprovement,
            input_constructor=mock_input_constructor,
        )

        model = BoTorchModel()
        model.fit(
            datasets=self.moo_training_data,
            search_space_digest=self.search_space_digest,
            candidate_metadata=self.candidate_metadata,
        )
        self.assertIsInstance(model.surrogate.model, SingleTaskGP)
        subset_outcome_constraints = (
            # model is subset since last output is not used
            self.moo_outcome_constraints[0][:, :2],
            self.moo_outcome_constraints[1],
        )
        constraints = get_outcome_constraint_transforms(
            outcome_constraints=subset_outcome_constraints,
        )
        with mock.patch(
            f"{ACQUISITION_PATH}.get_outcome_constraint_transforms",
            # Dummy candidates and acquisition function value.
            # This will return the same value as the original
            return_value=constraints,
        ) as mock_get_outcome_constraint_transforms:
            gen_results = model.gen(
                n=1,
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=self.moo_torch_opt_config,
            )
        mock_get_outcome_constraint_transforms.assert_called_once()
        ckwargs = mock_get_outcome_constraint_transforms.call_args.kwargs
        oc = ckwargs["outcome_constraints"]
        self.assertTrue(torch.equal(oc[0], subset_outcome_constraints[0]))
        self.assertTrue(torch.equal(oc[1], subset_outcome_constraints[1]))

        # Check input constructor args
        ckwargs = mock_input_constructor.call_args.kwargs
        expected_kwargs = {
            "constraints",
            "bounds",
            "objective",
            "training_data",
            "X_baseline",
            "model",
            "objective_thresholds",
        }
        self.assertSetEqual(set(ckwargs.keys()), expected_kwargs)
        for k in expected_kwargs:
            self.assertIsNotNone(ckwargs[k], f"{k} is None")

        self.assertIs(model.botorch_acqf_class, qLogNoisyExpectedHypervolumeImprovement)
        mock_input_constructor.assert_called_once()
        m = ckwargs["model"]
        self.assertIsInstance(m, SingleTaskGP)
        self.assertIsInstance(m.likelihood, FixedNoiseGaussianLikelihood)
        self.assertEqual(m.num_outputs, 2)
        training_data = ckwargs["training_data"]
        self.assertIsNotNone(training_data.Yvar)
        self.assertTrue(torch.equal(training_data.X, self.Xs[0]))
        self.assertTrue(
            torch.equal(
                training_data.Y,
                torch.cat([ds.Y for ds in self.moo_training_data], dim=-1),
            )
        )
        self.assertTrue(
            torch.equal(
                training_data.Yvar,
                torch.cat([ds.Yvar for ds in self.moo_training_data], dim=-1),
            )
        )
        self.assertTrue(
            torch.equal(
                ckwargs["objective_thresholds"], self.moo_objective_thresholds[:2]
            )
        )
        self.assertIs(ckwargs["constraints"], constraints)

        obj_t = gen_results.gen_metadata["objective_thresholds"]
        self.assertTrue(torch.equal(obj_t[:2], self.moo_objective_thresholds[:2]))
        self.assertTrue(np.isnan(obj_t[2].item()))

        self.assertIsInstance(ckwargs["objective"], WeightedMCMultiOutputObjective)
        self.assertTrue(
            torch.equal(
                ckwargs["objective"].weights,
                self.moo_objective_weights[:2],
            )
        )
        expected_X_baseline = _filter_X_observed(
            Xs=[dataset.X for dataset in self.moo_training_data],
            objective_weights=self.moo_objective_weights,
            outcome_constraints=self.moo_outcome_constraints,
            bounds=self.search_space_digest.bounds,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.assertTrue(
            torch.equal(ckwargs["X_baseline"], none_throws(expected_X_baseline))
        )
        # test inferred objective_thresholds
        objective_weights = torch.tensor([-1.0, -1.0, 0.0])
        outcome_constraints = (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[10.0]]),
        )
        linear_constraints = (
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[2.0]]),
        )

        torch_opt_config = dataclasses.replace(
            self.moo_torch_opt_config,
            objective_weights=objective_weights,
            objective_thresholds=None,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
        )

        objective_thresholds = torch.tensor([9.9, 3.3, float("nan")])
        with mock.patch(
            "ax.models.torch.botorch_modular.acquisition.infer_objective_thresholds",
            return_value=objective_thresholds,
        ) as _mock_model_infer_objective_thresholds:
            gen_results = model.gen(
                n=1,
                search_space_digest=self.mf_search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        expected_X_baseline = _filter_X_observed(
            Xs=[dataset.X for dataset in self.moo_training_data],
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
        self.assertIsInstance(m, SingleTaskGP)
        self.assertIsInstance(m.likelihood, FixedNoiseGaussianLikelihood)
        self.assertEqual(m.num_outputs, 2)
        self.assertIn("objective_thresholds", gen_results.gen_metadata)
        obj_t = gen_results.gen_metadata["objective_thresholds"]
        self.assertTrue(torch.equal(obj_t[:2], objective_thresholds[:2]))
        self.assertTrue(np.isnan(obj_t[2].item()))

        # Avoid polluting the registry for other tests; re-register correct input
        # contructor for qLogNEHVI.
        _register_acqf_input_constructor(
            acqf_cls=qLogNoisyExpectedHypervolumeImprovement,
            input_constructor=qLogNEHVI_input_constructor,
        )
