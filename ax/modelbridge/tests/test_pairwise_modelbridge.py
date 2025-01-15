#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import torch
from ax.core import Metric, Objective, OptimizationConfig
from ax.core.observation import ObservationData, ObservationFeatures
from ax.modelbridge.pairwise import (
    _binary_pref_to_comp_pair,
    _consolidate_comparisons,
    PairwiseModelBridge,
)
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.preference_stubs import get_pbo_experiment
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.utils.datasets import RankingDataset
from pyre_extensions import assert_is_instance


class PairwiseModelBridgeTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        experiment = get_pbo_experiment()
        self.experiment = experiment
        self.data = experiment.lookup_data()

    @TestCase.ax_long_test(
        reason="TODO[T199510629] Fix: break up test into one test per case"
    )
    def test_PairwiseModelBridge(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=PairwiseGP,
            mll_class=PairwiseLaplaceMarginalLogLikelihood,
            input_transform_classes=[Normalize],
            input_transform_options={
                "Normalize": {"d": len(self.experiment.parameters)}
            },
        )

        cases = [
            (qNoisyExpectedImprovement, None, 3),
            (qExpectedUtilityOfBestOption, None, 3),
            (
                AnalyticExpectedUtilityOfBestOption,
                # Analytic Acqfs do not support pending points and sequential opt
                {"optimizer_kwargs": {"sequential": False}},
                2,  # analytic EUBO only supports n=2
            ),
        ]
        for botorch_acqf_class, model_gen_options, n in cases:
            pmb = PairwiseModelBridge(
                experiment=self.experiment,
                search_space=self.experiment.search_space,
                data=self.data,
                model=BoTorchModel(
                    botorch_acqf_class=botorch_acqf_class,
                    surrogate=surrogate,
                ),
                transforms=[],
                optimization_config=OptimizationConfig(
                    Objective(
                        Metric(Keys.PAIRWISE_PREFERENCE_QUERY.value), minimize=False
                    )
                ),
                fit_tracking_metrics=False,
            )
            # Can generate candidates correctly
            # pyre-ignore: Incompatible parameter type [6]
            generator_run = pmb.gen(n=n, model_gen_options=model_gen_options)
            self.assertEqual(len(generator_run.arms), n)

        observation_data = [
            ObservationData(
                metric_names=[Keys.PAIRWISE_PREFERENCE_QUERY.value],
                means=np.array([0]),
                covariance=np.array([[np.nan]]),
            ),
            ObservationData(
                metric_names=[Keys.PAIRWISE_PREFERENCE_QUERY.value],
                means=np.array([1]),
                covariance=np.array([[np.nan]]),
            ),
        ]
        observation_features = [
            ObservationFeatures(parameters={"x1": 0.1, "x2": 0.2}, trial_index=0),
            ObservationFeatures(parameters={"x1": 0.3, "x2": 0.4}, trial_index=0),
        ]
        observation_features_with_metadata = [
            ObservationFeatures(parameters={"x1": 0.1, "x2": 0.2}, trial_index=0),
            ObservationFeatures(
                parameters={"x1": 0.3, "x2": 0.4},
                trial_index=0,
                metadata={"metadata_key": "metadata_val"},
            ),
        ]
        parameter_names = list(self.experiment.parameters.keys())
        outcomes = [assert_is_instance(Keys.PAIRWISE_PREFERENCE_QUERY.value, str)]

        datasets, _, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=outcomes,
            parameters=parameter_names,
            search_space_digest=None,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertTrue(isinstance(datasets[0], RankingDataset))
        self.assertTrue(candidate_metadata is None)

        datasets, _, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features_with_metadata,
            outcomes=outcomes,
            parameters=parameter_names,
            search_space_digest=None,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertTrue(isinstance(datasets[0], RankingDataset))
        self.assertTrue(candidate_metadata is not None)

        # Test individual helper methods
        X = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.1, 3.1, 4.1]]
        )
        Y = torch.tensor([[1, 0, 0, 1]])
        expected_X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [2.1, 3.1, 4.1]])
        ordered_Y = torch.tensor([[0, 1], [3, 2]])
        expected_Y = torch.tensor([[0, 1], [2, 0]])

        # `_binary_pref_to_comp_pair`.
        comp_pair_Y = _binary_pref_to_comp_pair(Y=Y)
        self.assertTrue(torch.equal(comp_pair_Y, ordered_Y))

        # test `_binary_pref_to_comp_pair` with invalid data
        bad_Y = torch.tensor([[1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            _binary_pref_to_comp_pair(Y=bad_Y)

        # `_consolidate_comparisons`.
        consolidated_X, consolidated_Y = _consolidate_comparisons(X=X, Y=comp_pair_Y)
        self.assertTrue(torch.equal(consolidated_X, expected_X))
        self.assertTrue(torch.equal(consolidated_Y, expected_Y))

        with self.assertRaises(ValueError):
            _consolidate_comparisons(
                X=X.expand(2, *X.shape), Y=comp_pair_Y.expand(2, *comp_pair_Y.shape)
            )
