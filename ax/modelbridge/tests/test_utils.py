#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.types import ComparisonOp
from ax.core.utils import get_pending_observation_features
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    extract_outcome_constraints,
    feasible_hypervolume,
    observation_data_to_array,
    pending_observations_as_array_list,
)
from ax.modelbridge.registry import Models
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_hierarchical_search_space_experiment,
)
from pyre_extensions import none_throws


TEST_PARAMETERIZATON_LIST = ["5", "foo", "True", "5"]


class TestModelbridgeUtils(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_experiment()
        self.arm = Arm({"x": 5, "y": "foo", "z": True, "w": 5})
        self.trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        self.experiment_2 = get_experiment()
        self.batch_trial = self.experiment_2.new_batch_trial(GeneratorRun([self.arm]))
        self.batch_trial.set_status_quo_with_weight(self.experiment_2.status_quo, 1)
        self.obs_feat = ObservationFeatures.from_arm(
            arm=self.trial.arm, trial_index=self.trial.index
        )
        self.hss_exp = get_hierarchical_search_space_experiment()
        self.hss_sobol = Models.SOBOL(search_space=self.hss_exp.search_space)
        self.hss_gr = self.hss_sobol.gen(n=1)
        self.hss_trial = self.hss_exp.new_trial(self.hss_gr)
        self.hss_arm = none_throws(self.hss_trial.arm)
        self.hss_cand_metadata = self.hss_trial._get_candidate_metadata(
            arm_name=self.hss_arm.name
        )
        self.hss_full_parameterization = self.hss_cand_metadata.get(
            Keys.FULL_PARAMETERIZATION
        ).copy()
        self.assertTrue(
            all(
                p_name in self.hss_full_parameterization
                for p_name in self.hss_exp.search_space.parameters
            )
        )
        self.hss_obs_feat = ObservationFeatures.from_arm(
            arm=self.hss_arm,
            trial_index=self.hss_trial.index,
            metadata=self.hss_cand_metadata,
        )
        self.hss_obs_feat_all_params = ObservationFeatures.from_arm(
            arm=Arm(self.hss_full_parameterization),
            trial_index=self.hss_trial.index,
            metadata={Keys.FULL_PARAMETERIZATION: self.hss_full_parameterization},
        )

    def test_extract_outcome_constraints(self) -> None:
        outcomes = ["m1", "m2", "m3"]
        # pass no outcome constraints
        self.assertIsNone(extract_outcome_constraints([], outcomes))

        outcome_constraints = [
            OutcomeConstraint(metric=Metric("m1"), op=ComparisonOp.LEQ, bound=0)
        ]
        res = extract_outcome_constraints(outcome_constraints, outcomes)
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertEqual(res[0].shape, (1, 3))
        self.assertListEqual(list(res[0][0]), [1, 0, 0])
        self.assertEqual(res[1][0][0], 0)

        outcome_constraints = [
            OutcomeConstraint(metric=Metric("m1"), op=ComparisonOp.LEQ, bound=0),
            ScalarizedOutcomeConstraint(
                metrics=[Metric("m2"), Metric("m3")],
                weights=[0.5, 0.5],
                op=ComparisonOp.GEQ,
                bound=1,
            ),
        ]
        res = extract_outcome_constraints(outcome_constraints, outcomes)
        self.assertEqual(res[0].shape, (2, 3))
        self.assertListEqual(list(res[0][0]), [1, 0, 0])
        self.assertListEqual(list(res[0][1]), [0, -0.5, -0.5])
        self.assertEqual(res[1][0][0], 0)
        self.assertEqual(res[1][1][0], -1)

    def test_extract_objective_thresholds(self) -> None:
        outcomes = ["m1", "m2", "m3", "m4"]
        objective = MultiObjective(
            objectives=[
                Objective(metric=Metric(name), minimize=False) for name in outcomes[:3]
            ]
        )
        objective_thresholds = [
            ObjectiveThreshold(
                metric=Metric(name),
                op=ComparisonOp.LEQ,
                bound=float(i + 2),
                relative=False,
            )
            for i, name in enumerate(outcomes[:3])
        ]

        # None of no thresholds
        self.assertIsNone(
            extract_objective_thresholds(
                objective_thresholds=[], objective=objective, outcomes=outcomes
            )
        )

        # Working case
        obj_t = extract_objective_thresholds(
            objective_thresholds=objective_thresholds,
            objective=objective,
            outcomes=outcomes,
        )
        expected_obj_t_not_nan = np.array([2.0, 3.0, 4.0])
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertTrue(np.array_equal(obj_t[:3], expected_obj_t_not_nan[:3]))
        self.assertTrue(np.isnan(obj_t[-1]))
        # pyre-fixme[16]: Optional type has no attribute `shape`.
        self.assertEqual(obj_t.shape[0], 4)

        # Returns NaN for objectives without a threshold.
        obj_t = extract_objective_thresholds(
            objective_thresholds=objective_thresholds[:2],
            objective=objective,
            outcomes=outcomes,
        )
        self.assertTrue(np.array_equal(obj_t[:2], expected_obj_t_not_nan[:2]))
        self.assertTrue(np.isnan(obj_t[-2:]).all())

        # Fails if a threshold does not have a corresponding metric.
        objective2 = Objective(Metric("m1"), minimize=False)
        with self.assertRaisesRegex(ValueError, "corresponding metrics"):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective2,
                outcomes=outcomes,
            )

        # Works with a single objective, single threshold
        obj_t = extract_objective_thresholds(
            objective_thresholds=objective_thresholds[:1],
            objective=objective2,
            outcomes=outcomes,
        )
        self.assertEqual(obj_t[0], 2.0)
        self.assertTrue(np.all(np.isnan(obj_t[1:])))
        self.assertEqual(obj_t.shape[0], 4)

        # Fails if relative
        objective_thresholds[2] = ObjectiveThreshold(
            metric=Metric("m3"), op=ComparisonOp.LEQ, bound=3
        )
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective,
                outcomes=outcomes,
            )
        objective_thresholds[2] = ObjectiveThreshold(
            metric=Metric("m3"), op=ComparisonOp.LEQ, bound=3, relative=True
        )
        with self.assertRaises(ValueError):
            extract_objective_thresholds(
                objective_thresholds=objective_thresholds,
                objective=objective,
                outcomes=outcomes,
            )

    def test_observation_data_to_array(self) -> None:
        outcomes = ["a", "b", "c"]
        obsd = ObservationData(
            metric_names=["c", "a", "b"],
            means=np.array([1, 2, 3]),
            covariance=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        )
        Y, Ycov = observation_data_to_array(outcomes=outcomes, observation_data=[obsd])
        self.assertTrue(np.array_equal(Y, np.array([[2, 3, 1]])))
        self.assertTrue(
            np.array_equal(Ycov, np.array([[[5, 6, 4], [8, 9, 7], [2, 3, 1]]]))
        )

        # With missing metrics.
        obsd2 = ObservationData(
            metric_names=["c", "a"],
            means=np.array([1, 2]),
            covariance=np.array([[1, 2], [4, 5]]),
        )
        Y, Ycov = observation_data_to_array(
            outcomes=outcomes, observation_data=[obsd, obsd2]
        )
        nan = float("nan")
        self.assertTrue(
            np.array_equal(Y, np.array([[2, 3, 1], [2, nan, 1]]), equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(
                Ycov,
                np.array(
                    [
                        [[5, 6, 4], [8, 9, 7], [2, 3, 1]],
                        [[5, nan, 4], [nan, nan, nan], [2, nan, 1]],
                    ]
                ),
                equal_nan=True,
            )
        )

    def test_feasible_hypervolume(self) -> None:
        ma = Metric(name="a", lower_is_better=False)
        mb = Metric(name="b", lower_is_better=True)
        mc = Metric(name="c", lower_is_better=False)
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(metrics=[ma, mb]),
            outcome_constraints=[
                OutcomeConstraint(
                    mc,
                    op=ComparisonOp.GEQ,
                    bound=0,
                    relative=False,
                )
            ],
            objective_thresholds=[
                ObjectiveThreshold(
                    ma,
                    bound=1.0,
                ),
                ObjectiveThreshold(
                    mb,
                    bound=1.0,
                ),
            ],
        )
        feas_hv = feasible_hypervolume(
            optimization_config,
            values={
                "a": np.array(
                    [
                        1.0,
                        3.0,
                        2.0,
                        2.0,
                    ]
                ),
                "b": np.array(
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ]
                ),
                "c": np.array(
                    [
                        0.0,
                        -0.0,
                        1.0,
                        -2.0,
                    ]
                ),
            },
        )
        self.assertEqual(list(feas_hv), [0.0, 0.0, 1.0, 1.0])

    def test_pending_observations_as_array_list(self) -> None:
        # Mark a trial dispatched so that there are pending observations.
        self.trial.mark_running(no_runner_required=True)
        # If outcome names are respected, unlisted metrics should be filtered out.
        self.assertEqual(
            [
                x.tolist()
                for x in none_throws(
                    pending_observations_as_array_list(
                        pending_observations=none_throws(
                            get_pending_observation_features(self.experiment)
                        ),
                        outcome_names=["m2", "m1"],
                        param_names=["x", "y", "z", "w"],
                    )
                )
            ],
            [[TEST_PARAMETERIZATON_LIST], [TEST_PARAMETERIZATON_LIST]],
        )
        self.experiment.attach_data(
            Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            )
        )
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            ),
        ):
            pending = none_throws(get_pending_observation_features(self.experiment))
        # There should be no pending observations for metric m2 now, since the
        # only trial there is, has been updated with data for it.
        self.assertEqual(
            [
                x.tolist()
                for x in none_throws(
                    pending_observations_as_array_list(
                        pending_observations=pending,
                        outcome_names=["m2", "m1"],
                        param_names=["x", "y", "z", "w"],
                    )
                )
            ],
            [[], [TEST_PARAMETERIZATON_LIST]],
        )
