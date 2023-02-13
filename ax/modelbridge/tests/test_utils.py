#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.types import ComparisonOp
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    extract_outcome_constraints,
    get_pending_observation_features,
    get_pending_observation_features_based_on_trial_status as get_pending_status,
    observation_data_to_array,
    pending_observations_as_array_list,
)
from ax.modelbridge.registry import Models
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_hierarchical_search_space_experiment,
)


class TestModelbridgeUtils(TestCase):
    def setUp(self) -> None:
        self.experiment = get_experiment()
        self.arm = Arm({"x": 1, "y": "foo", "z": True, "w": 4})
        self.trial = self.experiment.new_trial(GeneratorRun([self.arm]))
        self.experiment_2 = get_experiment()
        self.batch_trial = self.experiment_2.new_batch_trial(GeneratorRun([self.arm]))
        self.batch_trial.set_status_quo_with_weight(self.experiment_2.status_quo, 1)
        self.obs_feat = ObservationFeatures.from_arm(
            arm=self.trial.arm, trial_index=np.int64(self.trial.index)
        )
        self.hss_exp = get_hierarchical_search_space_experiment()
        self.hss_sobol = Models.SOBOL(search_space=self.hss_exp.search_space)
        self.hss_gr = self.hss_sobol.gen(n=1)
        self.hss_trial = self.hss_exp.new_trial(self.hss_gr)
        self.hss_arm = not_none(self.hss_trial.arm)
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
            trial_index=np.int64(self.hss_trial.index),
            metadata=self.hss_cand_metadata,
        )
        self.hss_obs_feat_all_params = ObservationFeatures.from_arm(
            arm=Arm(self.hss_full_parameterization),
            trial_index=np.int64(self.hss_trial.index),
            metadata={Keys.FULL_PARAMETERIZATION: self.hss_full_parameterization},
        )

    def test_get_pending_observation_features(self) -> None:
        # Pending observations should be none if there aren't any.
        self.assertIsNone(get_pending_observation_features(self.experiment))
        self.trial.mark_running(no_runner_required=True)
        # Now that the trial is deployed, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_observation_features(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        with patch.object(
            self.trial,
            "lookup_data",
            return_value=Data.from_evaluations(
                {self.trial.arm.name: {"m2": (1, 0)}}, trial_index=self.trial.index
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {"tracking": [self.obs_feat], "m2": [], "m1": [self.obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending...
        self.trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.experiment))
        # ... unless specified to include failed trials in pending observations.
        self.assertEqual(
            get_pending_observation_features(
                self.experiment, include_failed_as_pending=True
            ),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When a trial is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_observation_features(
                self.experiment, include_failed_as_pending=True
            ),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When an arm is abandoned, it should appear in pending features whether
        # or not there is data for it.
        self.batch_trial.mark_arm_abandoned(arm_name="0_0")
        # Checking with data for all metrics.
        with patch.object(
            self.batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        self.batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                            "tracking": (1, 0),
                        }
                    },
                    trial_index=self.trial.index,
                ),
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(
                    self.experiment, include_failed_as_pending=True
                ),
                {
                    "tracking": [self.obs_feat],
                    "m2": [self.obs_feat],
                    "m1": [self.obs_feat],
                },
            )
        # Checking with data for all metrics.
        with patch.object(
            self.trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        self.trial.arm.name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                            "tracking": (1, 0),
                        }
                    },
                    trial_index=self.trial.index,
                ),
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.experiment),
                {
                    "tracking": [self.obs_feat],
                    "m2": [self.obs_feat],
                    "m1": [self.obs_feat],
                },
            )

    def test_get_pending_observation_features_hss(self) -> None:
        # Pending observations should be none if there aren't any.
        self.assertIsNone(get_pending_observation_features(self.hss_exp))
        self.hss_trial.mark_running(no_runner_required=True)
        # Now that the trial is deployed, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        pending = get_pending_observation_features(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # Check that transforming observation features works correctly since this
        # is applying `Cast` transform, it should inject full parameterization into
        # resulting obs.feats.). Therefore, transforming the extracted pending features
        #  and observation features made from full parameterization should be the same.
        obsd = ObservationData(
            metric_names=["m1"], means=np.array([1.0]), covariance=np.array([[1.0]])
        )
        self.assertEqual(
            self.hss_sobol._transform_data(
                observations=[
                    Observation(data=obsd, features=pending["m1"][0])  # pyre-ignore
                ],
                search_space=self.hss_exp.search_space,
                transforms=self.hss_sobol._raw_transforms,
                transform_configs=None,
            ),
            self.hss_sobol._transform_data(
                observations=[
                    Observation(
                        data=obsd, features=self.hss_obs_feat_all_params.clone()
                    )
                ],
                search_space=self.hss_exp.search_space,
                transforms=self.hss_sobol._raw_transforms,
                transform_configs=None,
            ),
        )
        # With `fetch_data` on trial returning data for metric "m2", that metric
        # should no longer have pending observation features.
        with patch.object(
            self.hss_trial,
            "lookup_data",
            return_value=Data.from_evaluations(
                {self.hss_trial.arm.name: {"m2": (1, 0)}},
                trial_index=self.hss_trial.index,
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(self.hss_exp),
                {"m2": [], "m1": [self.hss_obs_feat]},
            )
        # When a trial is marked failed, it should no longer appear in pending...
        self.hss_trial.mark_failed()
        self.assertIsNone(get_pending_observation_features(self.hss_exp))
        # ... unless specified to include failed trials in pending observations.
        self.assertEqual(
            get_pending_observation_features(
                self.hss_exp, include_failed_as_pending=True
            ),
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # When an arm is abandoned, it should appear in pending features whether
        # or not there is data for it.
        hss_exp = get_hierarchical_search_space_experiment()
        hss_batch_trial = hss_exp.new_batch_trial(generator_run=self.hss_gr)
        hss_batch_trial.mark_arm_abandoned(hss_batch_trial.arms[0].name)
        # Checking with data for all metrics.
        with patch.object(
            hss_batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        hss_batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                        }
                    },
                    trial_index=hss_batch_trial.index,
                ),
            ),
        ):
            pending = get_pending_observation_features(
                hss_exp, include_failed_as_pending=True
            )
            self.assertEqual(
                pending,
                {
                    "m1": [self.hss_obs_feat],
                    "m2": [self.hss_obs_feat],
                },
            )
            # Check that candidate metadata is property propagated for abandoned arm.
            self.assertEqual(
                self.hss_sobol._transform_data(
                    observations=[Observation(data=obsd, features=pending["m1"][0])],
                    search_space=hss_exp.search_space,
                    transforms=self.hss_sobol._raw_transforms,
                    transform_configs=None,
                ),
                self.hss_sobol._transform_data(
                    observations=[
                        Observation(
                            data=obsd, features=self.hss_obs_feat_all_params.clone()
                        )
                    ],
                    search_space=hss_exp.search_space,
                    transforms=self.hss_sobol._raw_transforms,
                    transform_configs=None,
                ),
            )
        # Checking with data for all metrics.
        with patch.object(
            hss_batch_trial,
            "fetch_data",
            return_value=Metric._wrap_trial_data_multi(
                data=Data.from_evaluations(
                    {
                        hss_batch_trial.arms[0].name: {
                            "m1": (1, 0),
                            "m2": (1, 0),
                        }
                    },
                    trial_index=hss_batch_trial.index,
                ),
            ),
        ):
            self.assertEqual(
                get_pending_observation_features(hss_exp),
                {
                    "m2": [self.hss_obs_feat],
                    "m1": [self.hss_obs_feat],
                },
            )

    def test_get_pending_observation_features_batch_trial(self) -> None:
        # Check the same functionality for batched trials.
        self.assertIsNone(get_pending_observation_features(self.experiment_2))
        self.batch_trial.mark_running(no_runner_required=True)
        sq_obs_feat = ObservationFeatures.from_arm(
            self.batch_trial.arms_by_name.get("status_quo"),
            trial_index=self.batch_trial.index,
        )
        self.assertEqual(
            get_pending_observation_features(self.experiment_2),
            {
                "tracking": [self.obs_feat, sq_obs_feat],
                "m2": [self.obs_feat, sq_obs_feat],
                "m1": [self.obs_feat, sq_obs_feat],
            },
        )

    def test_get_pending_observation_features_based_on_trial_status(self) -> None:
        # Pending observations should be none if there aren't any as trial is
        # candidate.
        self.assertTrue(self.trial.status.is_candidate)
        self.assertIsNone(get_pending_status(self.experiment))
        self.trial.mark_staged()
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_status(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # Same should be true for running trial.
        # NOTE: Can't mark a staged trial running unless it uses a runner that
        # specifically requires staging; hacking around that here since the marking
        # logic does not matter for this test.
        self.trial._status = TrialStatus.RUNNING
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        self.assertEqual(
            get_pending_status(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )
        # When a trial is marked failed, it should no longer appear in pending.
        self.trial.mark_failed()
        self.assertIsNone(get_pending_status(self.experiment))
        # And if the trial is abandoned, it should always appear in pending features.
        self.trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            get_pending_status(self.experiment),
            {"tracking": [self.obs_feat], "m2": [self.obs_feat], "m1": [self.obs_feat]},
        )

    def test_get_pending_observation_features_based_on_trial_status_hss(self) -> None:
        self.assertTrue(self.hss_trial.status.is_candidate)
        self.assertIsNone(get_pending_status(self.hss_exp))
        self.hss_trial.mark_staged()
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        pending = get_pending_status(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # Same should be true for running trial.
        # NOTE: Can't mark a staged trial running unless it uses a runner that
        # specifically requires staging; hacking around that here since the marking
        # logic does not matter for this test.
        self.hss_trial._status = TrialStatus.RUNNING
        # Now that the trial is staged, it should become a pending trial on the
        # experiment and appear as pending for all metrics.
        pending = get_pending_status(self.hss_exp)
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )
        # When a trial is marked failed, it should no longer appear in pending.
        self.hss_trial.mark_failed()
        self.assertIsNone(get_pending_status(self.hss_exp))
        # And if the trial is abandoned, it should always appear in pending features.
        self.hss_trial._status = TrialStatus.ABANDONED  # Cannot re-mark a failed trial.
        self.assertEqual(
            pending,
            {
                "m1": [self.hss_obs_feat],
                "m2": [self.hss_obs_feat],
            },
        )

        # Check that transforming observation features works correctly since this
        # is applying `Cast` transform, it should inject full parameterization into
        # resulting obs.feats.). Therefore, transforming the extracted pending features
        #  and observation features made from full parameterization should be the same.
        obsd = ObservationData(
            metric_names=["m1"], means=np.array([1.0]), covariance=np.array([[1.0]])
        )
        self.assertEqual(
            self.hss_sobol._transform_data(
                observations=[
                    Observation(data=obsd, features=pending["m1"][0])  # pyre-ignore
                ],
                search_space=self.hss_exp.search_space,
                transforms=self.hss_sobol._raw_transforms,
                transform_configs=None,
            ),
            self.hss_sobol._transform_data(
                observations=[
                    Observation(
                        data=obsd, features=self.hss_obs_feat_all_params.clone()
                    )
                ],
                search_space=self.hss_exp.search_space,
                transforms=self.hss_sobol._raw_transforms,
                transform_configs=None,
            ),
        )

    def test_pending_observations_as_array_list(self) -> None:
        # Mark a trial dispatched so that there are pending observations.
        self.trial.mark_running(no_runner_required=True)
        # If outcome names are respected, unlisted metrics should be filtered out.
        self.assertEqual(
            [
                x.tolist()
                # pyre-fixme[16]: Optional type has no attribute `__iter__`.
                for x in pending_observations_as_array_list(
                    # pyre-fixme[6]: For 1st param expected `Dict[str,
                    #  List[ObservationFeatures]]` but got `Optional[Dict[str,
                    #  List[ObservationFeatures]]]`.
                    pending_observations=get_pending_observation_features(
                        self.experiment
                    ),
                    outcome_names=["m2", "m1"],
                    param_names=["x", "y", "z", "w"],
                )
            ],
            [[["1", "foo", "True", "4"]], [["1", "foo", "True", "4"]]],
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
            pending = get_pending_observation_features(self.experiment)
        # There should be no pending observations for metric m2 now, since the
        # only trial there is, has been updated with data for it.
        self.assertEqual(
            [
                x.tolist()
                for x in pending_observations_as_array_list(
                    # pyre-fixme[6]: For 1st param expected `Dict[str,
                    #  List[ObservationFeatures]]` but got `Optional[Dict[str,
                    #  List[ObservationFeatures]]]`.
                    pending_observations=pending,
                    outcome_names=["m2", "m1"],
                    param_names=["x", "y", "z", "w"],
                )
            ],
            [[], [["1", "foo", "True", "4"]]],
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
            objectives=[Objective(metric=Metric(name)) for name in outcomes[:3]]
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
        objective2 = Objective(Metric("m1"))
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

    def testObservationDataToArray(self) -> None:
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
