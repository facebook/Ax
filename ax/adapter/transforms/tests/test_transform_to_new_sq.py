# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from copy import deepcopy
from unittest import mock

import numpy as np
import numpy.typing as npt
from ax.adapter import Adapter
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.tests.test_relativize_transform import RelativizeDataTest
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.core.batch_trial import BatchTrial
from ax.core.observation import observations_from_data
from ax.exceptions.core import DataRequiredError
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.stats.statstools import relativize
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_branin_optimization_config,
    get_sobol,
)
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class TransformToNewSQTest(RelativizeDataTest):
    # pyre-ignore [15]: `relativize_classes` overrides attribute
    # defined in `RelativizeDataTest` inconsistently. Type `List
    # [Type[TransformToNewSQ]]` is not a subtype of the
    # overridden attribute `List[Type[Transform]]`
    relativize_classes = [TransformToNewSQ]
    cases: list[tuple[type[Transform], list[tuple[npt.NDArray, npt.NDArray]]]] = [
        (
            TransformToNewSQ,
            [
                (
                    np.array([1.6, 10.0]),
                    np.array([[0.16, 0.0], [0.0, 0.2892562]]),
                ),
                (np.array([2.0, 5.0]), np.array([[0.1, 0.0], [0.0, 0.2]])),
                (np.array([1.0, 10.0]), np.array([[0.3, 0.0], [0.0, 0.4]])),
            ],
        )
    ]

    # these tests are defined by RelativizeDataTest, but it is irrelevant
    # for TransformToNewSQ, so we don't need to run it here.
    def test_bad_relativize(self) -> None:
        pass

    def test_transform_status_quos_always_zero(self) -> None:
        pass


class TransformToNewSQSpecificTest(TestCase):
    def setUp(self) -> None:
        self.exp = get_branin_experiment(
            with_batch=True,
            with_status_quo=True,
        )
        t = self.exp.trials[0]
        t.mark_running(no_runner_required=True)
        self.exp.attach_data(
            get_branin_data_batch(batch=assert_is_instance(t, BatchTrial))
        )
        t.mark_completed()
        self.data = self.exp.fetch_data()

        self._refresh_adapter()

    def _refresh_adapter(self) -> None:
        self.adapter = Adapter(
            search_space=self.exp.search_space,
            generator=Generator(),
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )

    def test_adapter_without_status_quo_name(self) -> None:
        self.adapter._status_quo = None
        self.adapter._status_quo_name = None

        with self.assertRaisesRegex(
            DataRequiredError, "TransformToNewSQ requires status quo data."
        ):
            TransformToNewSQ(
                search_space=None,
                observations=[],
                adapter=self.adapter,
            )

    def test_transform_optimization_config(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            adapter=self.adapter,
        )
        oc = get_branin_optimization_config()
        new_oc = tf.transform_optimization_config(optimization_config=oc)
        self.assertIs(new_oc, oc)

    def test_untransform_outcome_constraints(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            adapter=self.adapter,
        )
        oc = get_branin_optimization_config()
        new_outcome_constraints = tf.untransform_outcome_constraints(
            outcome_constraints=oc.outcome_constraints
        )
        self.assertIs(new_outcome_constraints, oc.outcome_constraints)

    def test_custom_target_trial(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            adapter=self.adapter,
        )
        self.assertEqual(tf.default_trial_idx, 0)

        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            adapter=self.adapter,
            config={"target_trial_index": 1},
        )
        self.assertEqual(tf.default_trial_idx, 1)

    def test_single_trial_is_not_transformed(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            adapter=self.adapter,
        )
        obs = observations_from_data(
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )[:1]
        obs2 = tf.transform_observations(observations=deepcopy(obs))
        self.assertEqual(obs, obs2)

    def test_target_trial_index(self) -> None:
        sobol = get_sobol(search_space=self.exp.search_space)
        self.exp.new_batch_trial(
            generator_run=sobol.gen(2), should_add_status_quo_arm=True
        )
        t = self.exp.trials[1]
        t = assert_is_instance(t, BatchTrial)
        t.mark_running(no_runner_required=True)
        self.exp.attach_data(
            get_branin_data_batch(batch=assert_is_instance(t, BatchTrial))
        )

        self._refresh_adapter()

        observations = observations_from_data(
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )

        t = TransformToNewSQ(
            search_space=self.exp.search_space,
            observations=observations,
            adapter=self.adapter,
        )

        self.assertEqual(t.default_trial_idx, 1)

        with mock.patch(
            "ax.adapter.transforms.transform_to_new_sq.get_target_trial_index",
            return_value=0,
        ):
            t = TransformToNewSQ(
                search_space=self.exp.search_space,
                observations=observations,
                adapter=self.adapter,
            )

        self.assertEqual(t.default_trial_idx, 0)
        # test falling back to latest trial with SQ data
        with mock.patch(
            "ax.adapter.transforms.transform_to_new_sq.get_target_trial_index",
            return_value=10,
        ):
            t = TransformToNewSQ(
                search_space=self.exp.search_space,
                observations=observations,
                adapter=self.adapter,
            )

        self.assertEqual(t.default_trial_idx, 1)

    def test_transform_experiment_data(self) -> None:
        # Create two more trials with different SQ observations.
        sobol = get_sobol(search_space=self.exp.search_space)
        for sq_val in (2.0, 3.0):
            t = self.exp.new_batch_trial(
                generator_run=sobol.gen(2), should_add_status_quo_arm=True
            ).mark_completed(unsafe=True)
            data = get_branin_data_batch(batch=t)
            data.df.loc[(data.df["arm_name"] == "status_quo"), "mean"] = sq_val
            self.exp.attach_data(data=data)
        self._refresh_adapter()

        experiment_data = extract_experiment_data(
            experiment=self.exp, data_loader_config=DataLoaderConfig()
        )

        # Create the transform with trial 2 as the target.
        tf = TransformToNewSQ(
            search_space=None,
            adapter=self.adapter,
            config={"target_trial_index": 2},
        )
        transformed_data = tf.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Verify that status quo observations are dropped except for the target trial.
        tf_obs_data = transformed_data.observation_data
        self.assertEqual(
            tf_obs_data[
                tf_obs_data.index.get_level_values("arm_name")
                == self.adapter.status_quo_name
            ]
            .index.get_level_values("trial_index")
            .item(),
            2,  # target trial index from the config.
        )
        # Verify that data from the target trial is not transformed.
        target_trial_data = experiment_data.observation_data.loc[2]
        transformed_target_trial_data = transformed_data.observation_data.loc[2]
        assert_frame_equal(target_trial_data, transformed_target_trial_data)

        # Check that the data for trials 0 and 1 are transformed correctly.
        sq_data_target = self.adapter.status_quo_data_by_trial[2]
        for t_idx in (0, 1):
            sq_data = self.adapter.status_quo_data_by_trial[t_idx]
            # Get the data for the non-sq arms.
            trial_data = experiment_data.observation_data.loc[t_idx]
            trial_data = trial_data[
                trial_data.index.get_level_values("arm_name")
                != self.adapter.status_quo_name
            ]
            # Relativize the data with respect to the SQ for the trial.
            means_rel, sems_rel = relativize(
                means_t=trial_data["mean", "branin"],
                sems_t=trial_data["sem", "branin"],
                mean_c=sq_data.means[0],
                sem_c=sq_data.covariance[0, 0] ** 0.5,
                as_percent=False,
                control_as_constant=tf.control_as_constant,
            )
            # Derelativize using target SQ.
            target_mean = sq_data_target.means[0]
            abs_target_mean = np.abs(target_mean)
            means = means_rel * abs_target_mean + target_mean
            sems = sems_rel * abs_target_mean
            self.assertTrue(
                np.allclose(
                    means,
                    transformed_data.observation_data.loc[t_idx]["mean", "branin"],
                )
            )
            self.assertTrue(
                np.allclose(
                    sems, transformed_data.observation_data.loc[t_idx]["sem", "branin"]
                )
            )
