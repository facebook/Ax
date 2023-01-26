#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError
from ax.models.torch.botorch_modular.utils import (
    _get_shared_rows,
    _tensor_difference,
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
    disable_one_to_many_transforms,
    use_model_list,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import ModelList
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputPerturbation,
    InputTransform,
    Normalize,
)
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from botorch.utils.testing import MockModel, MockPosterior


class BoTorchModelUtilsTest(TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, _, _, _, _ = get_torch_test_data(dtype=self.dtype)
        self.Xs2, self.Ys2, self.Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, offset=1.0  # Making this data different.
        )
        self.fixed_noise_datasets = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(self.Xs, self.Ys, self.Yvars)
        ]
        self.supervised_datasets = [
            SupervisedDataset(X=X, Y=Y) for X, Y, in zip(self.Xs, self.Ys)
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
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            SingleTaskMultiFidelityGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                    fidelity_features=[2],
                ),
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiFidelityGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
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
        # With fidelity features and unknown variances, use SingleTaskMultiFidelityGP.
        self.assertEqual(
            MultiTaskGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[], bounds=[], task_features=[1]
                ),
            ),
        )
        # With fidelity features and known variances, use FixedNoiseMultiFidelityGP.
        self.assertEqual(
            FixedNoiseMultiTaskGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
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
        # Without fidelity/task features but with Yvar specifications, use FixedNoiseGP.
        self.assertEqual(
            FixedNoiseGP,
            choose_model_class(
                datasets=self.fixed_noise_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            ),
        )
        # W/out fidelity/task features and w/out Yvar specifications, use SingleTaskGP.
        self.assertEqual(
            SingleTaskGP,
            choose_model_class(
                datasets=self.supervised_datasets,
                search_space_digest=SearchSpaceDigest(
                    feature_names=[],
                    bounds=[],
                ),
            ),
        )

    def test_choose_botorch_acqf_class(self) -> None:
        self.assertEqual(qNoisyExpectedImprovement, choose_botorch_acqf_class())
        self.assertEqual(
            qNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(objective_thresholds=self.objective_thresholds),
        )
        self.assertEqual(
            qNoisyExpectedHypervolumeImprovement,
            choose_botorch_acqf_class(objective_weights=torch.tensor([0.5, 0.5])),
        )
        self.assertEqual(
            qNoisyExpectedImprovement,
            choose_botorch_acqf_class(objective_weights=torch.tensor([1.0, 0.0])),
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

    def test_use_model_list(self) -> None:
        self.assertFalse(
            use_model_list(
                datasets=self.supervised_datasets, botorch_model_class=SingleTaskGP
            )
        )
        self.assertFalse(  # Batched multi-output case.
            use_model_list(
                datasets=[SupervisedDataset(X=self.Xs[0], Y=Y) for Y in self.Ys],
                botorch_model_class=SingleTaskGP,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(X=self.Xs[0], Y=self.Ys[0]),
                    SupervisedDataset(X=self.Xs2[0], Y=self.Ys2[0]),
                ],
                botorch_model_class=SingleTaskGP,
            )
        )
        self.assertTrue(
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
                    SupervisedDataset(X=self.Xs[0], Y=self.Ys[0]),
                    SupervisedDataset(X=self.Xs2[0], Y=self.Ys2[0]),
                ],
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
            )
        )
        self.assertTrue(
            use_model_list(
                datasets=[
                    SupervisedDataset(X=self.Xs[0], Y=self.Ys[0].repeat(1, 2)),
                ],
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
            )
        )

    def test_tensor_difference(self) -> None:
        n, m = 3, 2
        A = torch.arange(n * m).reshape(n, m)
        B = torch.cat((A[: n - 1], torch.randn(2, m)), dim=0)
        # permute B
        B = B[torch.randperm(len(B))]

        C = _tensor_difference(A=A, B=B)

        self.assertEqual(C.size(dim=0), 2)


class ConvertToBlockDesignTest(TestCase):
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
        datasets = [SupervisedDataset(X=X, Y=Y) for Y in Ys]
        metric_names = ["y1", "y2"]
        new_datasets, new_metric_names = convert_to_block_design(
            datasets=datasets,
            metric_names=metric_names,
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], SupervisedDataset)
        self.assertTrue(torch.equal(new_datasets[0].X(), X))
        self.assertTrue(torch.equal(new_datasets[0].Y(), torch.cat(Ys, dim=-1)))
        self.assertEqual(new_metric_names, ["y1_y2"])

        # simple case: block design, fixed
        Yvars = [torch.rand(4, 1), torch.rand(4, 1)]
        datasets = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar) for Y, Yvar in zip(Ys, Yvars)
        ]
        new_datasets, new_metric_names = convert_to_block_design(
            datasets=datasets,
            metric_names=metric_names,
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], FixedNoiseDataset)
        self.assertTrue(torch.equal(new_datasets[0].X(), X))
        self.assertTrue(torch.equal(new_datasets[0].Y(), torch.cat(Ys, dim=-1)))
        # pyre-fixme[16]: `SupervisedDataset` has no attribute `Yvar`.
        self.assertTrue(torch.equal(new_datasets[0].Yvar(), torch.cat(Yvars, dim=-1)))
        self.assertEqual(new_metric_names, ["y1_y2"])

        # test error is raised if not block design and force=False
        X2 = torch.cat((X[:3], torch.rand(1, 2)))
        datasets = [SupervisedDataset(X=X, Y=Y) for X, Y in zip((X, X2), Ys)]
        with self.assertRaisesRegex(
            UnsupportedError, "Cannot convert data to non-block design data."
        ):
            convert_to_block_design(datasets=datasets, metric_names=metric_names)

        # test warning is issued if not block design and force=True (supervised)
        with warnings.catch_warnings(record=True) as ws:
            new_datasets, new_metric_names = convert_to_block_design(
                datasets=datasets, metric_names=metric_names, force=True
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
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], SupervisedDataset)
        self.assertTrue(torch.equal(new_datasets[0].X(), X[:3]))
        self.assertTrue(
            torch.equal(new_datasets[0].Y(), torch.cat([Y[:3] for Y in Ys], dim=-1))
        )
        self.assertEqual(new_metric_names, ["y1_y2"])

        # test warning is issued if not block design and force=True (fixed)
        datasets = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip((X, X2), Ys, Yvars)
        ]
        with warnings.catch_warnings(record=True) as ws:
            new_datasets, new_metric_names = convert_to_block_design(
                datasets=datasets, metric_names=metric_names, force=True
            )
        self.assertTrue(any(issubclass(w.category, AxWarning) for w in ws))
        self.assertTrue(
            any(
                "Forcing converion of data not complying to a block design"
                in str(w.message)
                for w in ws
            )
        )
        self.assertEqual(len(new_datasets), 1)
        self.assertIsInstance(new_datasets[0], FixedNoiseDataset)
        self.assertTrue(torch.equal(new_datasets[0].X(), X[:3]))
        self.assertTrue(
            torch.equal(new_datasets[0].Y(), torch.cat([Y[:3] for Y in Ys], dim=-1))
        )
        self.assertTrue(
            torch.equal(
                new_datasets[0].Yvar(), torch.cat([Yvar[:3] for Yvar in Yvars], dim=-1)
            )
        )
        self.assertEqual(new_metric_names, ["y1_y2"])

    def test_disable_one_to_many_transforms(self) -> None:
        mm = MockModel(posterior=MockPosterior())
        # No input transforms.
        with disable_one_to_many_transforms(model=mm):
            pass
        # Error with Chained intf.
        normalize = Normalize(d=2)
        perturbation = InputPerturbation(perturbation_set=torch.rand(2, 2))
        chained_tf = ChainedInputTransform(
            normalize=normalize, perturbation=perturbation
        )
        # pyre-ignore [16]
        mm.input_transform = deepcopy(chained_tf)
        with self.assertRaisesRegex(UnsupportedError, "ChainedInputTransforms"):
            with disable_one_to_many_transforms(model=mm):
                pass
        # The transform is not modified.
        self.assertTrue(
            checked_cast(InputTransform, mm.input_transform).equals(chained_tf)
        )
        # With one-to-many transform.
        mm.input_transform = deepcopy(perturbation)
        with disable_one_to_many_transforms(model=mm):
            self.assertFalse(
                checked_cast(InputTransform, mm.input_transform).transform_on_eval
            )
        self.assertTrue(
            checked_cast(InputTransform, mm.input_transform).transform_on_eval
        )
        self.assertTrue(
            checked_cast(InputTransform, mm.input_transform).equals(perturbation)
        )
        # With ModelList.
        mm_list = ModelList(mm, deepcopy(mm))
        with disable_one_to_many_transforms(model=mm_list):
            for mm in mm_list.models:
                self.assertFalse(mm.input_transform.transform_on_eval)
        for mm in mm_list.models:
            self.assertTrue(mm.input_transform.transform_on_eval)
