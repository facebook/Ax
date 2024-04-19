#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict

import numpy as np
import torch
from ax.core import Arm, GeneratorRun
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import RangeParameter
from ax.core.types import TEvaluationOutcome, TParameterization
from ax.modelbridge.pairwise import (
    _binary_pref_to_comp_pair,
    _consolidate_comparisons,
    PairwiseModelBridge,
)
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.utils.instantiation import InstantiationBase
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.utils.datasets import RankingDataset


class PairwiseModelBridgeTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        def evaluate(
            parameters: Dict[str, TParameterization]
        ) -> Dict[str, TEvaluationOutcome]:
            # A pair at a time
            assert len(parameters.keys()) == 2
            arm1, arm2 = list(parameters.keys())
            arm1_outcome_values = [
                checked_cast(float, v) for v in parameters[arm1].values()
            ]
            arm2_outcome_values = [
                checked_cast(float, v) for v in parameters[arm2].values()
            ]
            arm1_sum = float(sum(arm1_outcome_values))
            arm2_sum = float(sum(arm2_outcome_values))
            is_arm1_preferred = int(arm1_sum - arm2_sum > 0)
            return {
                arm1: {Keys.PAIRWISE_PREFERENCE_QUERY: is_arm1_preferred},
                arm2: {Keys.PAIRWISE_PREFERENCE_QUERY: 1 - is_arm1_preferred},
            }

        experiment = InstantiationBase.make_experiment(
            name="pref_experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.0, 0.6],
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [0.0, 0.7],
                },
            ],
            objectives={Keys.PAIRWISE_PREFERENCE_QUERY: "minimize"},
            is_test=True,
        )

        for _ in range(3):
            gr = GeneratorRun(
                [
                    Arm(
                        {
                            pn: np.random.uniform(
                                low=checked_cast(RangeParameter, p).lower,
                                high=checked_cast(RangeParameter, p).upper,
                            )
                            for pn, p in experiment.search_space.parameters.items()
                        }
                    )
                    for _ in range(2)
                ]
            )
            trial = experiment.new_batch_trial(generator_run=gr)
            trial.attach_batch_trial_data(
                raw_data=evaluate({a.name: a.parameters for a in trial.arms})
            )
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()

        # Manually add arms from previous trials
        trial = experiment.new_batch_trial()
        trial.add_arm(experiment.trials[1].arms[0])
        trial.add_arm(experiment.trials[2].arms[0])
        trial.attach_batch_trial_data(
            raw_data=evaluate({a.name: a.parameters for a in trial.arms})
        )
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()

        self.experiment = experiment
        self.data = experiment.lookup_data()

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
            (qNoisyExpectedImprovement, None),
            (
                AnalyticExpectedUtilityOfBestOption,
                # Analytic Acqfs do not support pending points and sequential opt
                {"optimizer_kwargs": {"sequential": False}},
            ),
        ]
        for botorch_acqf_class, model_gen_options in cases:
            pmb = PairwiseModelBridge(
                experiment=self.experiment,
                search_space=self.experiment.search_space,
                data=self.data,
                model=BoTorchModel(
                    botorch_acqf_class=botorch_acqf_class,
                    surrogate=surrogate,
                ),
                transforms=[],
            )
            # Can generate candidates correctly
            # pyre-ignore: Incompatible parameter type [6]
            generator_run = pmb.gen(n=2, model_gen_options=model_gen_options)
            self.assertEqual(len(generator_run.arms), 2)

        observation_data = [
            ObservationData(
                metric_names=[Keys.PAIRWISE_PREFERENCE_QUERY],
                means=np.array([0]),
                covariance=np.array([[np.nan]]),
            ),
            ObservationData(
                metric_names=[Keys.PAIRWISE_PREFERENCE_QUERY],
                means=np.array([1]),
                covariance=np.array([[np.nan]]),
            ),
        ]
        observation_features = [
            ObservationFeatures(parameters={"X1": 0.1, "X2": 0.2}, trial_index=0),
            ObservationFeatures(parameters={"X1": 0.3, "X2": 0.4}, trial_index=0),
        ]
        observation_features_with_metadata = [
            ObservationFeatures(parameters={"X1": 0.1, "X2": 0.2}, trial_index=0),
            ObservationFeatures(
                parameters={"X1": 0.3, "X2": 0.4},
                trial_index=0,
                metadata={"metadata_key": "metadata_val"},
            ),
        ]
        parameters = ["X1", "X2"]
        outcomes = [checked_cast(str, Keys.PAIRWISE_PREFERENCE_QUERY)]

        datasets, _, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=outcomes,
            parameters=parameters,
            search_space_digest=None,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertTrue(isinstance(datasets[0], RankingDataset))
        self.assertTrue(candidate_metadata is None)

        datasets, _, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features_with_metadata,
            outcomes=outcomes,
            parameters=parameters,
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
