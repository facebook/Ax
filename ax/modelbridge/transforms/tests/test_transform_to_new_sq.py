# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest import mock

import numpy as np
import numpy.typing as npt
from ax.core.batch_trial import BatchTrial
from ax.core.observation import observations_from_data
from ax.modelbridge import ModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.tests.test_relativize_transform import RelativizeDataTest
from ax.modelbridge.transforms.transform_to_new_sq import TransformToNewSQ
from ax.models.base import Model
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_branin_optimization_config,
    get_sobol,
)
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

        self._refresh_modelbridge()

    def _refresh_modelbridge(self) -> None:
        self.modelbridge = ModelBridge(
            search_space=self.exp.search_space,
            model=Model(),
            experiment=self.exp,
            data=self.exp.lookup_data(),
            status_quo_name="status_quo",
        )

    def test_modelbridge_without_status_quo_name(self) -> None:
        self.modelbridge._status_quo = None
        self.modelbridge._status_quo_name = None

        with self.assertRaisesRegex(
            AssertionError, "TransformToNewSQ requires status quo data."
        ):
            TransformToNewSQ(
                search_space=None,
                observations=[],
                modelbridge=self.modelbridge,
            )

    def test_transform_optimization_config(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            modelbridge=self.modelbridge,
        )
        oc = get_branin_optimization_config()
        new_oc = tf.transform_optimization_config(optimization_config=oc)
        self.assertIs(new_oc, oc)

    def test_untransform_outcome_constraints(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            modelbridge=self.modelbridge,
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
            modelbridge=self.modelbridge,
        )
        self.assertEqual(tf.default_trial_idx, 0)

        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            modelbridge=self.modelbridge,
            config={"target_trial_index": 1},
        )
        self.assertEqual(tf.default_trial_idx, 1)

    def test_single_trial_is_not_transformed(self) -> None:
        tf = TransformToNewSQ(
            search_space=None,
            observations=[],
            modelbridge=self.modelbridge,
        )
        obs = self.modelbridge._prepare_observations(
            experiment=self.exp, data=self.data
        )
        obs2 = tf.transform_observations(obs)
        self.assertEqual(obs, obs2)

    def test_target_trial_index(self) -> None:
        sobol = get_sobol(search_space=self.exp.search_space)
        self.exp.new_batch_trial(generator_run=sobol.gen(2), optimize_for_power=True)
        t = self.exp.trials[1]
        t = assert_is_instance(t, BatchTrial)
        t.mark_running(no_runner_required=True)
        self.exp.attach_data(
            get_branin_data_batch(batch=assert_is_instance(t, BatchTrial))
        )

        self._refresh_modelbridge()

        observations = observations_from_data(
            experiment=self.exp,
            data=self.exp.lookup_data(),
        )

        t = TransformToNewSQ(
            search_space=self.exp.search_space,
            observations=observations,
            modelbridge=self.modelbridge,
        )

        self.assertEqual(t.default_trial_idx, 1)

        with mock.patch(
            "ax.modelbridge.transforms.transform_to_new_sq.get_target_trial_index",
            return_value=0,
        ):
            t = TransformToNewSQ(
                search_space=self.exp.search_space,
                observations=observations,
                modelbridge=self.modelbridge,
            )

        self.assertEqual(t.default_trial_idx, 0)
        # test falling back to latest trial with SQ data
        with mock.patch(
            "ax.modelbridge.transforms.transform_to_new_sq.get_target_trial_index",
            return_value=10,
        ):
            t = TransformToNewSQ(
                search_space=self.exp.search_space,
                observations=observations,
                modelbridge=self.modelbridge,
            )

        self.assertEqual(t.default_trial_idx, 1)
