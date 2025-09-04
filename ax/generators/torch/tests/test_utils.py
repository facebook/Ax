#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError, UserInputError
from ax.generators.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.generators.torch.botorch_modular.utils import (
    _get_shared_rows,
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
    get_cv_fold,
    logger,
    ModelConfig,
    subset_state_dict,
    use_model_list,
)
from ax.generators.torch.utils import _to_inequality_constraints, predict_from_model
from ax.generators.torch_base import TorchOptConfig
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors.ensemble import EnsemblePosterior
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance, none_throws


class BoTorchGeneratorUtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dtype = torch.float
        (
            self.Xs,
            self.Ys,
            self.Yvars,
            _,
            _,
            self.feature_names,
            self.metric_names,
        ) = get_torch_test_data(dtype=self.dtype)
        self.Xs2, self.Ys2, self.Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype,
            offset=1.0,  # Making this data different.
        )
        self.fixed_noise_dataset = SupervisedDataset(
            X=self.Xs,
            Y=self.Ys,
            Yvar=self.Yvars,
            feature_names=self.feature_names,
            outcome_names=self.metric_names,
        )
        self.supervised_dataset = SupervisedDataset(
            X=self.Xs,
            Y=self.Ys,
            feature_names=self.feature_names,
            outcome_names=self.metric_names,
        )
        self.none_Yvars = [torch.tensor([[np.nan], [np.nan]])]
        self.task_features = []
        self.objective_thresholds = torch.tensor([0.5, 1.5])

    def test_choose_model_class_fidelity_features(self) -> None:
        # Only a single fidelity feature can be used.
        with self.assertRaisesRegex(
            NotImplementedError, "Only a single fidelity feature"
        ):
            choose_model_class(
                dataset=self.fixed_noise_dataset,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], fidelity_features=[1, 2]
                ),
            )
        # No support for non-empty task & fidelity features yet.
        with self.assertRaisesRegex(NotImplementedError, "Multi-task multi-fidelity"):
            choose_model_class(
                dataset=self.fixed_noise_dataset,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[1],
                    fidelity_features=[1],
                ),
            )
        # With fidelity features, use SingleTaskMultiFidelityGP.
        for ds in [self.supervised_dataset, self.fixed_noise_dataset]:
            self.assertEqual(
                SingleTaskMultiFidelityGP,
                choose_model_class(
                    dataset=ds,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=[],
                        bounds=[],
                        fidelity_features=[2],
                    ),
                ),
            )

    def test_choose_model_class_task_features(self) -> None:
        # Only a single task feature can be used.
        with self.assertRaisesRegex(NotImplementedError, "Only a single task feature"):
            choose_model_class(
                dataset=self.fixed_noise_dataset,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1, 2]
                ),
            )
        # With task features use MultiTaskGP.
        for datasets in (self.supervised_dataset, self.fixed_noise_dataset):
            self.assertEqual(
                MultiTaskGP,
                choose_model_class(
                    dataset=datasets,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=[], bounds=[], task_features=[1]
                    ),
                ),
            )

    def test_choose_model_class_discrete_features(self) -> None:
        # With discrete features, use MixedSingleTaskyGP.
        self.assertEqual(
            MixedSingleTaskGP,
            choose_model_class(
                dataset=self.supervised_dataset,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[],
                    categorical_features=[1],
                ),
            ),
        )

    def test_choose_model_class(self) -> None:
        # Without fidelity/task features, use SingleTaskGP.
        for ds in [self.fixed_noise_dataset, self.supervised_dataset]:
            self.assertEqual(
                SingleTaskGP,
                choose_model_class(
                    dataset=ds,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=[],
                        bounds=[],
                    ),
                ),
            )

    def test_choose_botorch_acqf_class(self) -> None:
        self.assertEqual(
            qLogNoisyExpectedImprovement,
            choose_botorch_acqf_class(
                torch_opt_config=TorchOptConfig(
                    objective_weights=torch.tensor([1.0, 0.0]),
                    is_moo=False,
                ),
                datasets=[self.supervised_dataset],
            ),
        )
        self.assertEqual(
            qLogNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(
                torch_opt_config=TorchOptConfig(
                    objective_weights=torch.tensor([1.0, -1.0]),
                    is_moo=True,
                ),
                datasets=[self.supervised_dataset],
            ),
        )
        # Check that it doesn't error out when given a mix of known and unknown noise.
        ds1 = self.fixed_noise_dataset
        ds1.outcome_names = ["y2"]
        datasets = [self.supervised_dataset, ds1]
        with self.assertLogs(logger=logger, level="DEBUG") as logs:
            choose_botorch_acqf_class(
                torch_opt_config=TorchOptConfig(
                    objective_weights=torch.tensor([1.0, -1.0]),
                    is_moo=True,
                    outcome_constraints=(
                        torch.tensor([[1.0, 0.0]]),
                        torch.tensor([[4.0]]),
                    ),
                ),
                datasets=datasets,
            )
        self.assertTrue(
            any(
                "Only a subset of datasets have noise observations." in str(log)
                for log in logs
            )
        )

    def test_construct_acquisition_and_optimizer_options(self) -> None:
        # Two dicts for `Acquisition` should be concatenated
        acqf_options = {}
        botorch_acqf_options: TConfig = {Keys.NUM_FANTASIES: 64}

        acquisition_function_kwargs = {Keys.CURRENT_VALUE: torch.tensor([1.0])}
        ax_acquisition_kwargs = {Keys.SUBSET_MODEL: False}
        optimizer_kwargs = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        model_gen_options = {
            Keys.AX_ACQUISITION_KWARGS: ax_acquisition_kwargs,
            Keys.ACQF_KWARGS: acquisition_function_kwargs,
            Keys.OPTIMIZER_KWARGS: optimizer_kwargs,
        }

        (
            final_acq_options,
            final_botorch_acqf_options,
            final_opt_options,
            final_botorch_acqf_classes_with_options,
        ) = construct_acquisition_and_optimizer_options(
            acqf_options=acqf_options,
            botorch_acqf_options=botorch_acqf_options,
            # pyre-fixme[6]: Incompatible parameter type [6]:
            # In call `construct_acquisition_and_optimizer_options`, for
            # argument `model_gen_options`, expected `Optional[Dict[str,
            # Union[None, Dict[int, typing.Any], Dict[str, typing.Any],
            # List[int], List[str], OptimizationConfig, WinsorizationConfig,
            # AcquisitionFunction, float, int, str]]]` but got `Dict[Keys,
            # Union[Dict[Keys, bool], Dict[Keys, int], Dict[Keys, Tensor]]]`.
            model_gen_options=model_gen_options,
        )
        self.assertEqual(final_acq_options, {Keys.SUBSET_MODEL: False})
        self.assertEqual(
            final_botorch_acqf_options,
            {Keys.NUM_FANTASIES: 64, Keys.CURRENT_VALUE: torch.tensor([1.0])},
        )
        self.assertEqual(final_opt_options, optimizer_kwargs)
        self.assertIsNone(final_botorch_acqf_classes_with_options)

        with self.assertRaisesRegex(
            ValueError, "Found forbidden keys in `model_gen_options`"
        ):
            construct_acquisition_and_optimizer_options(
                acqf_options=acqf_options,
                botorch_acqf_options=botorch_acqf_options,
                model_gen_options={**model_gen_options, "extra": "key"},
            )

        # test with botorch_acqf_classes_with_options
        botorch_acqf_classes_with_options = [
            (PosteriorMean, {}),
            (qLogNoisyExpectedImprovement, {}),
        ]
        with warnings.catch_warnings(record=True) as ws:
            (
                final_acq_options,
                final_botorch_acqf_options,
                final_opt_options,
                final_botorch_acqf_classes_with_options,
            ) = construct_acquisition_and_optimizer_options(
                acqf_options=acqf_options,
                botorch_acqf_options=botorch_acqf_options,
                # pyre-fixme[6]: Incompatible parameter type [6]:
                # In call `construct_acquisition_and_optimizer_options`, for
                # argument `model_gen_options`, expected `Optional[Dict[str,
                # Union[None, Dict[int, typing.Any], Dict[str, typing.Any],
                # List[int], List[str], OptimizationConfig, WinsorizationConfig,
                # AcquisitionFunction, float, int, str]]]` but got `Dict[Keys,
                # Union[Dict[Keys, bool], Dict[Keys, int], Dict[Keys, Tensor]]]`.
                model_gen_options=model_gen_options,
                # pyre-fixme[6]: Incompatible parameter type [6]: In call
                # `construct_acquisition_and_optimizer_options`, for argument
                # `botorch_acqf_classes_with_options`, expected `Optional[
                # List[Tuple[Type[AcquisitionFunction], Dict[str, Union[None,
                # Dict[int, typing.Any], Dict[str, typing.Any], List[int], List[str],
                # OptimizationConfig, WinsorizationConfig, AcquisitionFunction,
                # float, int, str]]]]]` but got `List[Tuple[Type[
                # qLogNoisyExpectedImprovement], Dict[typing.Any, typing.Any]]]`.
                botorch_acqf_classes_with_options=botorch_acqf_classes_with_options,
            )
            self.assertEqual(
                botorch_acqf_classes_with_options,
                final_botorch_acqf_classes_with_options,
            )
            self.assertEqual(final_botorch_acqf_options, botorch_acqf_options)
            self.assertEqual(final_acq_options, {Keys.SUBSET_MODEL: False})
            self.assertEqual(final_opt_options, optimizer_kwargs)
            self.assertEqual(len(ws), 1)
            warning = ws[0]
            self.assertEqual(warning.category, AxWarning)
            self.assertEqual(
                str(warning.message),
                "botorch_acqf_options are being ignored, due to using "
                "MultiAcquisition. Specify options for each acquistion function"
                "via botorch_acqf_classes_with_options.",
            )

        # test that botorch_acqf_classes_with_options is updated
        botorch_acqf_classes_with_options = [
            (qLogNoisyExpectedImprovement, {Keys.NUM_FANTASIES: 64}),
        ]
        (
            final_acq_options,
            final_botorch_acqf_options,
            final_opt_options,
            final_botorch_acqf_classes_with_options,
        ) = construct_acquisition_and_optimizer_options(
            acqf_options=acqf_options,
            botorch_acqf_options=botorch_acqf_options,
            # pyre-fixme[6]: Incompatible parameter type [6]:
            # In call `construct_acquisition_and_optimizer_options`, for
            # argument `model_gen_options`, expected `Optional[Dict[str,
            # Union[None, Dict[int, typing.Any], Dict[str, typing.Any],
            # List[int], List[str], OptimizationConfig, WinsorizationConfig,
            # AcquisitionFunction, float, int, str]]]` but got `Dict[Keys,
            # Union[Dict[Keys, bool], Dict[Keys, int], Dict[Keys, Tensor]]]`.
            model_gen_options=model_gen_options,
            # pyre-fixme[6]: Incompatible parameter type [6]: In call
            # `construct_acquisition_and_optimizer_options`, for argument
            # `botorch_acqf_classes_with_options`, expected `Optional[
            # List[Tuple[Type[AcquisitionFunction], Dict[str, Union[None,
            # Dict[int, typing.Any], Dict[str, typing.Any], List[int], List[str],
            # OptimizationConfig, WinsorizationConfig, AcquisitionFunction,
            # float, int, str]]]]]` but got `List[Tuple[Type[
            # qLogNoisyExpectedImprovement], Dict[typing.Any, typing.Any]]]`.
            botorch_acqf_classes_with_options=botorch_acqf_classes_with_options,
        )
        self.assertEqual(
            [
                (
                    qLogNoisyExpectedImprovement,
                    {Keys.NUM_FANTASIES: 64, Keys.CURRENT_VALUE: torch.tensor([1.0])},
                )
            ],
            final_botorch_acqf_classes_with_options,
        )
        self.assertEqual(final_botorch_acqf_options, botorch_acqf_options)
        self.assertEqual(final_acq_options, {Keys.SUBSET_MODEL: False})
        self.assertEqual(final_opt_options, optimizer_kwargs)

    def test_use_model_list(self) -> None:
        self.assertFalse(
            use_model_list(
                datasets=[self.supervised_dataset],
                model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
        )
        self.assertFalse(  # Batched multi-output case.
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                ],
                model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
        )
        # Multi-output with allow_batched_models
        self.assertFalse(
            use_model_list(
                datasets=2
                * [
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                ],
                model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
                allow_batched_models=True,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=2
                * [
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                ],
                model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
                allow_batched_models=False,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                    SupervisedDataset(
                        X=self.Xs2,
                        Y=self.Ys2,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                ],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
                model_configs=[ModelConfig(botorch_model_class=SingleTaskGP)],
            )
        )
        self.assertFalse(
            use_model_list(
                datasets=[self.supervised_dataset],
                model_configs=[ModelConfig(botorch_model_class=MultiTaskGP)],
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1, 2]
                ),
            )
        )
        # Not using model list with single outcome.
        self.assertFalse(
            use_model_list(
                datasets=[self.supervised_dataset],
                model_configs=[
                    ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP)
                ],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
        )
        # Using it with multiple outcomes.
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                    SupervisedDataset(
                        X=self.Xs2,
                        Y=self.Ys2,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                ],
                model_configs=[
                    ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP)
                ],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs,
                        Y=self.Ys.repeat(1, 2),
                        feature_names=self.feature_names,
                        outcome_names=["y1", "y2"],
                    ),
                ],
                model_configs=[
                    ModelConfig(botorch_model_class=SaasFullyBayesianSingleTaskGP)
                ],
                search_space_digest=SearchSpaceDigest(feature_names=[], bounds=[]),
            )
        )

    def test_get_shared_rows(self) -> None:
        # test bad input
        with self.assertRaisesRegex(
            UserInputError, "All inputs must be two-dimensional."
        ):
            _get_shared_rows(Xs=[torch.rand(3, 4, 2), torch.rand(2, 4, 2)])

        X1 = torch.rand(4, 2)

        # X1 is subset of X2
        X2 = torch.cat((X1[:2], torch.rand(1, 2), X1[2:]))
        X_shared, shared_idcs = _get_shared_rows([X1, X2])
        # unique() here just makes sure the order is consistent
        self.assertTrue(torch.equal(X1.unique(dim=0), X_shared))
        self.assertTrue(torch.equal(shared_idcs[0], torch.arange(4)))
        self.assertTrue(torch.equal(shared_idcs[1], torch.tensor([0, 1, 3, 4])))
        # X2 is subset of X1
        X2 = X1[:3]
        X_shared, shared_idcs = _get_shared_rows([X1, X2])
        self.assertTrue(torch.equal(X2.unique(dim=0), X_shared))
        self.assertTrue(torch.equal(shared_idcs[0], torch.arange(3)))
        self.assertTrue(torch.equal(shared_idcs[1], torch.arange(3)))

        # no overlap
        X2 = torch.rand(2, 2)
        X_shared, shared_idcs = _get_shared_rows([X1, X2])
        self.assertEqual(X_shared.numel(), 0)
        self.assertEqual(shared_idcs[0].numel(), 0)
        self.assertEqual(shared_idcs[1].numel(), 0)

        # three tensors
        X2 = torch.cat((X1[:2], torch.rand(1, 2), X1[2:]))
        X3 = torch.cat((torch.rand(1, 2), X1[:2], torch.rand(1, 2), X1[3:4]))
        X_shared, shared_idcs = _get_shared_rows([X1, X2, X3])
        self.assertTrue(torch.equal(shared_idcs[0], torch.tensor([0, 1, 3])))
        self.assertTrue(torch.equal(shared_idcs[1], torch.tensor([0, 1, 4])))
        self.assertTrue(
            torch.equal(X_shared, X1[torch.tensor([0, 1, 3])].unique(dim=0))
        )

        # test tensors with duplicate rows
        X4 = torch.cat((X1[:2], X1[:2]))
        X_shared, shared_idcs = _get_shared_rows(Xs=[X1, X4])
        self.assertTrue(torch.equal(X_shared, X1[:2].unique(dim=0)))
        self.assertTrue(torch.equal(shared_idcs[0], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(shared_idcs[1], torch.tensor([2, 3])))

    def test_convert_to_block_design(self) -> None:
        # simple case: block design, supervised
        X = torch.rand(4, 2)
        Ys = [torch.rand(4, 1), torch.rand(4, 1)]
        metric_names = ["y1", "y2"]
        datasets_unknown_noise = [
            SupervisedDataset(
                X=X,
                Y=Ys[i],
                feature_names=["x1", "x2"],
                outcome_names=[metric_names[i]],
            )
            for i in range(2)
        ]
        new_datasets = convert_to_block_design(datasets=datasets_unknown_noise)
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], SupervisedDataset)
        self.assertTrue(torch.equal(new_datasets[0].X, X))
        self.assertTrue(torch.equal(new_datasets[0].Y, torch.cat(Ys, dim=-1)))
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # simple case: block design, fixed
        Yvars = [torch.rand(4, 1), torch.rand(4, 1)]
        datasets_noisy = [
            SupervisedDataset(
                X=X,
                Y=Ys[i],
                Yvar=Yvars[i],
                feature_names=["x1", "x2"],
                outcome_names=[metric_names[i]],
            )
            for i in range(2)
        ]
        new_datasets = convert_to_block_design(datasets=datasets_noisy)
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNotNone(new_datasets[0].Yvar)
        self.assertTrue(torch.equal(new_datasets[0].X, X))
        self.assertTrue(torch.equal(new_datasets[0].Y, torch.cat(Ys, dim=-1)))
        self.assertTrue(
            torch.equal(none_throws(new_datasets[0].Yvar), torch.cat(Yvars, dim=-1))
        )
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # test error is raised if not block design and force=False
        X2 = torch.cat((X[:3], torch.rand(1, 2)))
        datasets = [
            SupervisedDataset(
                X=X, Y=Y, feature_names=["x1", "x2"], outcome_names=[name]
            )
            for X, Y, name in zip((X, X2), Ys, metric_names)
        ]
        with self.assertRaisesRegex(
            UnsupportedError, "Cannot convert data to non-block design data."
        ):
            convert_to_block_design(datasets=datasets)

        # test a log is produced if not block design and force=True (supervised)
        with self.assertLogs(logger=logger, level="DEBUG") as logs:
            new_datasets = convert_to_block_design(datasets=datasets, force=True)
        self.assertTrue(
            any(
                "Forcing conversion of data not complying to a block design" in str(log)
                for log in logs
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNone(new_datasets[0].Yvar)
        # unique() here makes sure the order is consistent
        self.assertTrue(torch.equal(new_datasets[0].X, X[:3].unique(dim=0)))
        self.assertTrue(
            torch.equal(new_datasets[0].Y, torch.cat([Y[:3] for Y in Ys], dim=-1))
        )
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # test a log is produced if not block design and force=True (fixed)
        datasets = [
            SupervisedDataset(
                X=X, Y=Y, Yvar=Yvar, feature_names=["x1", "x2"], outcome_names=[name]
            )
            for X, Y, Yvar, name in zip((X, X2), Ys, Yvars, metric_names)
        ]
        with self.assertLogs(logger=logger, level="DEBUG") as logs:
            new_datasets = convert_to_block_design(datasets=datasets, force=True)
        self.assertTrue(
            any(
                "Forcing conversion of data not complying to a block design" in str(log)
                for log in logs
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNotNone(new_datasets[0].Yvar)
        self.assertTrue(torch.equal(new_datasets[0].X, X[:3].unique(dim=0)))
        self.assertTrue(
            torch.equal(new_datasets[0].Y, torch.cat([Y[:3] for Y in Ys], dim=-1))
        )
        self.assertTrue(
            torch.equal(
                none_throws(new_datasets[0].Yvar),
                torch.cat([Yvar[:3] for Yvar in Yvars], dim=-1),
            )
        )
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # Test that known and unknown noise can be merged if force=True.
        datasets = [datasets_unknown_noise[0], datasets_noisy[1]]
        with self.assertLogs(logger=logger, level="DEBUG") as logs:
            new_datasets = convert_to_block_design(datasets=datasets, force=True)
        self.assertTrue(
            any(
                "Only a subset of datasets have noise observations." in str(log)
                for log in logs
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNone(new_datasets[0].Yvar)
        # Errors out if force=False.
        with self.assertRaisesRegex(
            UnsupportedError, "Cannot convert mixed data with and without variance"
        ):
            convert_to_block_design(datasets=datasets, force=False)

    def test_to_inequality_constraints(self) -> None:
        A = torch.tensor([[0, 1, -2, 3], [0, 1, 0, 0]])
        b = torch.tensor([[1], [2]])
        ineq_constraints = none_throws(
            _to_inequality_constraints(linear_constraints=(A, b))
        )
        self.assertEqual(len(ineq_constraints), 2)
        self.assertAllClose(ineq_constraints[0][0], torch.tensor([1, 2, 3]))
        self.assertAllClose(ineq_constraints[0][1], torch.tensor([-1, 2, -3]))
        self.assertEqual(ineq_constraints[0][2], -1.0)
        self.assertAllClose(ineq_constraints[1][0], torch.tensor([1]))
        self.assertAllClose(ineq_constraints[1][1], torch.tensor([-1]))
        self.assertEqual(ineq_constraints[1][2], -2.0)

    def test_subset_state_dict(self) -> None:
        m0 = SingleTaskGP(train_X=torch.rand(5, 2), train_Y=torch.rand(5, 1))
        m1 = SingleTaskGP(train_X=torch.rand(5, 2), train_Y=torch.rand(5, 1))
        model_list = ModelListGP(m0, m1)
        model_list_state_dict = assert_is_instance(model_list.state_dict(), OrderedDict)
        # Subset the model dict from model list and check that it is correct.
        m0_state_dict = model_list.models[0].state_dict()
        subsetted_m0_state_dict = subset_state_dict(
            state_dict=model_list_state_dict, submodel_index=0
        )
        self.assertEqual(m0_state_dict.keys(), subsetted_m0_state_dict.keys())
        for k in m0_state_dict:
            self.assertTrue(torch.equal(m0_state_dict[k], subsetted_m0_state_dict[k]))
        # Check that it can be loaded on the model.
        m0.load_state_dict(subsetted_m0_state_dict)

    def test_get_folds(self) -> None:
        X = torch.rand(10, 2)
        Y = torch.rand(10, 1)
        Yvar = torch.rand(10, 1)
        dataset = SupervisedDataset(
            X=X, Y=Y, Yvar=Yvar, feature_names=["x1", "x2"], outcome_names=["y"]
        )
        # CV
        cv_fold = get_cv_fold(
            dataset=dataset,
            X=X,
            Y=Y,
            idcs=torch.arange(3),
        )
        self.assertTrue(torch.equal(cv_fold.test_X, X[:3]))
        self.assertTrue(torch.equal(cv_fold.test_Y, Y[:3]))
        self.assertTrue(torch.equal(cv_fold.train_dataset.X, X[3:]))
        self.assertTrue(torch.equal(cv_fold.train_dataset.Y, Y[3:]))
        self.assertTrue(
            torch.equal(cv_fold.train_dataset.Yvar, Yvar[3:])  # pyre-ignore[6]
        )

    def test_model_config(self) -> None:
        # Test that model identifier is correctly computed.
        mc1 = ModelConfig(
            botorch_model_class=SingleTaskGP,
            covar_module_class=ScaleMaternKernel,
            covar_module_options={"ard_num_dims": 1},
            name="GP",
        )
        self.assertEqual(mc1.identifier, "GP")
        mc2 = ModelConfig(
            botorch_model_class=SingleTaskGP,
            covar_module_class=ScaleMaternKernel,
            covar_module_options={"ard_num_dims": 1},
        )
        mc_str = (
            "ModelConfig("
            "botorch_model_class=<class 'botorch.models.gp_regression.SingleTaskGP'>, "
            "model_options={}, mll_class=None, mll_options={}, "
            "input_transform_classes=<class 'botorch.utils.types.DEFAULT'>, "
            "input_transform_options={}, outcome_transform_classes=None, "
            "outcome_transform_options={}, "
            "covar_module_class=<class 'ax.generators.torch.botorch_modular.kernels."
            "ScaleMaternKernel'>, covar_module_options={'ard_num_dims': 1}, "
            "likelihood_class=None, likelihood_options={}, name=None)"
        )
        self.assertEqual(mc2.identifier, mc_str)

    def test_predict_from_model_ensemble_posterior(self) -> None:
        """Test predict_from_model with EnsemblePosterior support."""
        X = torch.rand(2, 3)

        # Create a mock EnsemblePosterior
        mock_posterior = Mock(spec=EnsemblePosterior)
        mock_posterior.mixture_mean = torch.rand(2, 2)
        mock_posterior.mixture_variance = torch.rand(2, 2)

        # Create a mock model
        mock_model = Mock()
        mock_model.posterior.return_value = mock_posterior

        # Test prediction
        mean, cov = predict_from_model(mock_model, X, use_posterior_predictive=False)

        # Verify shapes
        self.assertEqual(mean.shape, (2, 2))  # (n_points, n_outputs)
        self.assertEqual(cov.shape, (2, 2, 2))  # (n_points, n_outputs, n_outputs)
        self.assertTrue(torch.all(cov >= 0))  # Ensure covariance is positive
