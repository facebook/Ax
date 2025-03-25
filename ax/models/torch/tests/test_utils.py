#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections import OrderedDict

import numpy as np

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError
from ax.models.torch.botorch_modular.utils import (
    _get_shared_rows,
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
    subset_state_dict,
    use_model_list,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
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
        self.fixed_noise_datasets = [
            SupervisedDataset(
                X=X,
                Y=Y,
                Yvar=Yvar,
                feature_names=self.feature_names,
                outcome_names=[mn],
            )
            for X, Y, Yvar, mn in zip(self.Xs, self.Ys, self.Yvars, self.metric_names)
        ]
        self.supervised_datasets = [
            SupervisedDataset(
                X=X, Y=Y, feature_names=self.feature_names, outcome_names=[mn]
            )
            for X, Y, mn in zip(self.Xs, self.Ys, self.metric_names)
        ]
        self.none_Yvars = [torch.tensor([[np.nan], [np.nan]])]
        self.task_features = []
        self.objective_thresholds = torch.tensor([0.5, 1.5])

    def test_choose_model_class_fidelity_features(self) -> None:
        # Only a single fidelity feature can be used.
        with self.assertRaisesRegex(
            NotImplementedError, "Only a single fidelity feature"
        ):
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], fidelity_features=[1, 2]
                ),
            )
        # No support for non-empty task & fidelity features yet.
        with self.assertRaisesRegex(NotImplementedError, "Multi-task multi-fidelity"):
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[1],
                    fidelity_features=[1],
                ),
            )
        # With fidelity features, use SingleTaskMultiFidelityGP.
        for ds in [self.supervised_datasets, self.fixed_noise_datasets]:
            self.assertEqual(
                SingleTaskMultiFidelityGP,
                choose_model_class(
                    datasets=ds,
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
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1, 2]
                ),
            )
        # With task features use MultiTaskGP.
        for datasets in (self.supervised_datasets, self.fixed_noise_datasets):
            self.assertEqual(
                MultiTaskGP,
                choose_model_class(
                    datasets=datasets,
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
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    task_features=[],
                    categorical_features=[1],
                ),
            ),
        )

    def test_choose_model_class(self) -> None:
        # Mix of known and unknown variances.
        with self.assertRaisesRegex(
            ValueError, "Variances should all be specified, or none should be."
        ):
            choose_model_class(
                datasets=[self.fixed_noise_datasets[0], self.supervised_datasets[0]],
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            )
        # Without fidelity/task features, use SingleTaskGP.
        for ds in [self.fixed_noise_datasets, self.supervised_datasets]:
            self.assertEqual(
                SingleTaskGP,
                choose_model_class(
                    datasets=ds,
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
                )
            ),
        )
        self.assertEqual(
            qLogNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(
                torch_opt_config=TorchOptConfig(
                    objective_weights=torch.tensor([1.0, -1.0]),
                    is_moo=True,
                )
            ),
        )

    def test_construct_acquisition_and_optimizer_options(self) -> None:
        # Two dicts for `Acquisition` should be concatenated
        acqf_options = {Keys.NUM_FANTASIES: 64}

        acquisition_function_kwargs = {Keys.CURRENT_VALUE: torch.tensor([1.0])}
        optimizer_kwargs = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        model_gen_options = {
            Keys.ACQF_KWARGS: acquisition_function_kwargs,
            Keys.OPTIMIZER_KWARGS: optimizer_kwargs,
        }

        (
            final_acq_options,
            final_opt_options,
        ) = construct_acquisition_and_optimizer_options(
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, Dict[str,
            #  typing.Any], OptimizationConfig, AcquisitionFunction, float, int, str]]`
            #  but got `Dict[Keys, int]`.
            # pyre-fixme[6]: For 2nd param expected `Optional[Dict[str, Union[None,
            #  Dict[str, typing.Any], OptimizationConfig, AcquisitionFunction, float,
            #  int, str]]]` but got `Dict[Keys, Union[Dict[Keys, int], Dict[Keys,
            #  Tensor]]]`.
            acqf_options=acqf_options,
            # pyre-fixme[6]: For 2nd param expected `Optional[Dict[str, Union[None,
            #  Dict[str, typing.Any], OptimizationConfig, AcquisitionFunction, float,
            #  int, str]]]` but got `Dict[Keys, Union[Dict[Keys, int], Dict[Keys,
            #  Tensor]]]`.
            model_gen_options=model_gen_options,
        )
        self.assertEqual(
            final_acq_options,
            {Keys.NUM_FANTASIES: 64, Keys.CURRENT_VALUE: torch.tensor([1.0])},
        )
        self.assertEqual(final_opt_options, optimizer_kwargs)

        with self.assertRaisesRegex(
            ValueError, "Found forbidden keys in `model_gen_options`"
        ):
            construct_acquisition_and_optimizer_options(
                # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, Dict[str,
                #  typing.Any], OptimizationConfig, AcquisitionFunction, float, int,
                #  str]]` but got `Dict[Keys, int]`.
                # pyre-fixme[6]: For 2nd param expected `Optional[Dict[str, Union[None,
                #  Dict[str, typing.Any], OptimizationConfig, AcquisitionFunction,
                #  float, int, str]]]` but got `Dict[Keys, Union[Dict[Keys, int],
                #  Dict[Keys, Tensor]]]`.
                acqf_options=acqf_options,
                model_gen_options={**model_gen_options, "extra": "key"},
            )

    def test_use_model_list(self) -> None:
        self.assertFalse(
            use_model_list(
                datasets=self.supervised_datasets, botorch_model_class=SingleTaskGP
            )
        )
        self.assertFalse(  # Batched multi-output case.
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=Y,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                    for Y in self.Ys
                ],
                botorch_model_class=SingleTaskGP,
            )
        )
        # Multi-output with allow_batched_models
        self.assertFalse(
            use_model_list(
                datasets=2
                * [
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=Y,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                    for Y in self.Ys
                ],
                botorch_model_class=SingleTaskGP,
                allow_batched_models=True,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=2
                * [
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=Y,
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    )
                    for Y in self.Ys
                ],
                botorch_model_class=SingleTaskGP,
                allow_batched_models=False,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=self.Ys[0],
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                    SupervisedDataset(
                        X=self.Xs2[0],
                        Y=self.Ys2[0],
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                ],
                botorch_model_class=SingleTaskGP,
            )
        )
        self.assertFalse(
            use_model_list(
                datasets=self.supervised_datasets, botorch_model_class=MultiTaskGP
            )
        )
        # Not using model list with single outcome.
        self.assertFalse(
            use_model_list(
                datasets=self.supervised_datasets,
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
            )
        )
        # Using it with multiple outcomes.
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=self.Ys[0],
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                    SupervisedDataset(
                        X=self.Xs2[0],
                        Y=self.Ys2[0],
                        feature_names=self.feature_names,
                        outcome_names=["y"],
                    ),
                ],
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(
                        X=self.Xs[0],
                        Y=self.Ys[0].repeat(1, 2),
                        feature_names=self.feature_names,
                        outcome_names=["y1", "y2"],
                    ),
                ],
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
            )
        )

    def test_get_shared_rows(self) -> None:
        X1 = torch.rand(4, 2)

        # X1 is subset of X2
        X2 = torch.cat((X1[:2], torch.rand(1, 2), X1[2:]))
        X_shared, shared_idcs = _get_shared_rows([X1, X2])
        self.assertTrue(torch.equal(X1, X_shared))
        self.assertTrue(torch.equal(shared_idcs[0], torch.arange(4)))
        self.assertTrue(torch.equal(shared_idcs[1], torch.tensor([0, 1, 3, 4])))

        # X2 is subset of X1
        X2 = X1[:3]
        X_shared, shared_idcs = _get_shared_rows([X1, X2])
        self.assertTrue(torch.equal(X2, X_shared))
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
        self.assertTrue(torch.equal(shared_idcs[2], torch.tensor([1, 2, 4])))
        self.assertTrue(torch.equal(X_shared, X1[torch.tensor([0, 1, 3])]))

    def test_convert_to_block_design(self) -> None:
        # simple case: block design, supervised
        X = torch.rand(4, 2)
        Ys = [torch.rand(4, 1), torch.rand(4, 1)]
        metric_names = ["y1", "y2"]
        datasets = [
            SupervisedDataset(
                X=X,
                Y=Ys[i],
                feature_names=["x1", "x2"],
                outcome_names=[metric_names[i]],
            )
            for i in range(2)
        ]
        new_datasets = convert_to_block_design(datasets=datasets)
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], SupervisedDataset)
        self.assertTrue(torch.equal(new_datasets[0].X, X))
        self.assertTrue(torch.equal(new_datasets[0].Y, torch.cat(Ys, dim=-1)))
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # simple case: block design, fixed
        Yvars = [torch.rand(4, 1), torch.rand(4, 1)]
        datasets = [
            SupervisedDataset(
                X=X,
                Y=Ys[i],
                Yvar=Yvars[i],
                feature_names=["x1", "x2"],
                outcome_names=[metric_names[i]],
            )
            for i in range(2)
        ]
        new_datasets = convert_to_block_design(datasets=datasets)
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

        # test warning is issued if not block design and force=True (supervised)
        with warnings.catch_warnings(record=True) as ws:
            new_datasets = convert_to_block_design(datasets=datasets, force=True)
        # pyre-fixme[6]: For 1st param expected `Iterable[object]` but got `bool`.
        self.assertTrue(any(issubclass(w.category, AxWarning)) for w in ws)
        self.assertTrue(
            any(
                "Forcing conversion of data not complying to a block design"
                in str(w.message)
                for w in ws
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNone(new_datasets[0].Yvar)
        self.assertTrue(torch.equal(new_datasets[0].X, X[:3]))
        self.assertTrue(
            torch.equal(new_datasets[0].Y, torch.cat([Y[:3] for Y in Ys], dim=-1))
        )
        self.assertEqual(new_datasets[0].outcome_names, metric_names)

        # test warning is issued if not block design and force=True (fixed)
        datasets = [
            SupervisedDataset(
                X=X, Y=Y, Yvar=Yvar, feature_names=["x1", "x2"], outcome_names=[name]
            )
            for X, Y, Yvar, name in zip((X, X2), Ys, Yvars, metric_names)
        ]
        with warnings.catch_warnings(record=True) as ws:
            new_datasets = convert_to_block_design(datasets=datasets, force=True)
        self.assertTrue(any(issubclass(w.category, AxWarning) for w in ws))
        self.assertTrue(
            any(
                "Forcing conversion of data not complying to a block design"
                in str(w.message)
                for w in ws
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsNotNone(new_datasets[0].Yvar)
        self.assertTrue(torch.equal(new_datasets[0].X, X[:3]))
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

    def test_to_inequality_constraints(self) -> None:
        A = torch.tensor([[0, 1, -2, 3], [0, 1, 0, 0]])
        b = torch.tensor([[1], [2]])
        ineq_constraints = none_throws(
            _to_inequality_constraints(linear_constraints=(A, b))
        )
        self.assertEqual(len(ineq_constraints), 2)
        self.assertTrue(torch.allclose(ineq_constraints[0][0], torch.tensor([1, 2, 3])))
        self.assertTrue(
            torch.allclose(ineq_constraints[0][1], torch.tensor([-1, 2, -3]))
        )
        self.assertEqual(ineq_constraints[0][2], -1.0)
        self.assertTrue(torch.allclose(ineq_constraints[1][0], torch.tensor([1])))
        self.assertTrue(torch.allclose(ineq_constraints[1][1], torch.tensor([-1])))
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
