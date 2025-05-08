# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.analysis.plotly.utils import truncate_label
from ax.analysis.utils import _relativize_data, prepare_arm_data
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.arm import Arm
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


class TestUtils(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()

        self.client = Client()
        self.client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        self.client.configure_optimization(
            objective="foo",
            outcome_constraints=["bar >= 11", "baz <= 18", "qux >= 1998"],
        )

        # Get two trials and fail one, giving us a ragged structure
        self.client.get_next_trials(max_trials=2)
        self.client.complete_trial(
            trial_index=0, raw_data={"foo": 1.0, "bar": 2.0, "baz": 3.0, "qux": 4.0}
        )
        self.client.mark_trial_failed(trial_index=1)

        # Complete 5 trials successfully
        for _ in range(5):
            for trial_index, parameterization in self.client.get_next_trials(
                max_trials=1
            ).items():
                self.client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "foo": assert_is_instance(parameterization["x1"], float),
                        "bar": assert_is_instance(parameterization["x1"], float)
                        - 2 * assert_is_instance(parameterization["x2"], float),
                        "baz": 3.0,
                        "qux": 4.0,
                    },
                )

    def test_prepare_arm_data_validation(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Must provide at least one metric name"
        ):
            prepare_arm_data(
                experiment=self.client._experiment,
                metric_names=[],
                use_model_predictions=False,
            )

        with self.assertRaisesRegex(
            UserInputError, "Requested metrics .* are not present in the experiment."
        ):
            prepare_arm_data(
                experiment=self.client._experiment,
                metric_names=["foo", "bar", "zed"],
                use_model_predictions=False,
            )

        with self.assertRaisesRegex(
            UserInputError, "Trial with index .* not found in experiment."
        ):
            prepare_arm_data(
                experiment=self.client._experiment,
                metric_names=["foo", "bar"],
                trial_index=1998,
                use_model_predictions=False,
            )

        with self.assertRaisesRegex(UserInputError, "Must provide an adapter"):
            prepare_arm_data(
                experiment=self.client._experiment,
                metric_names=["foo", "bar"],
                use_model_predictions=True,
            )

        with self.assertRaisesRegex(
            UserInputError,
            "Cannot provide additional arms when use_model_predictions=False.",
        ):
            prepare_arm_data(
                experiment=self.client._experiment,
                metric_names=["foo", "bar"],
                use_model_predictions=False,
                additional_arms=[Arm(parameters={"x1": 0.5, "x2": 0.5})],
            )

    def test_prepare_arm_data_raw(self) -> None:
        df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar", "baz", "qux"],
            use_model_predictions=False,
        )

        # Check that the columns are correct
        self.assertEqual(
            set(df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
                "baz_mean",
                "baz_sem",
                "qux_mean",
                "qux_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((df["arm_name"] == arm_name).sum(), 1)

        # Check that the FAILED trial has no mean or sem
        self.assertTrue(np.isnan(df.loc[1]["foo_mean"]))
        self.assertTrue(np.isnan(df.loc[1]["foo_sem"]))

        # Check that all SEMs are NaN
        self.assertTrue(df["foo_sem"].isna().all())
        self.assertTrue(df["bar_sem"].isna().all())

        # Check that p_feasible is NaN for the arm without data and not NaN for the
        # other arms.
        self.assertTrue(np.isnan(df.loc[1]["p_feasible"]))
        self.assertFalse(df[df["arm_name"] != "1_0"]["p_feasible"].isna().any())

        only_foo_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo"],
            use_model_predictions=False,
        )
        # Check that the columns are correct
        self.assertEqual(
            set(only_foo_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(only_foo_df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((only_foo_df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are NaN
        self.assertTrue(only_foo_df["foo_sem"].isna().all())

        only_trial_0_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=False,
            trial_index=0,
        )

        # Check that the columns are correct
        self.assertEqual(
            set(only_trial_0_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )

        # Check that we have one row per arm and that the arm appears only once
        self.assertEqual(len(only_trial_0_df), 1)
        self.assertEqual(only_trial_0_df.loc[0]["arm_name"], "0_0")

        # Check that all means are not NaN
        self.assertFalse(only_trial_0_df["foo_mean"].isna().any())

        # Check that all SEMs are NaN
        self.assertTrue(only_trial_0_df["foo_sem"].isna().all())

        only_completed_trials_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=False,
            trial_statuses=[TrialStatus.COMPLETED],
        )

        # Check that we have two rows per arm and that each arm appears only once
        self.assertEqual(
            len(only_completed_trials_df), len(self.client._experiment.arms_by_name) - 1
        )

        # Check that all means are not NaN
        self.assertFalse(only_completed_trials_df["foo_mean"].isna().any())

        # Check that all SEMs are NaN
        self.assertTrue(only_completed_trials_df["foo_sem"].isna().all())

    def test_prepare_arm_data_use_model_predictions(self) -> None:
        df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar", "baz", "qux"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
        )

        # Check that the columns are correct
        self.assertEqual(
            set(df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
                "baz_mean",
                "baz_sem",
                "qux_mean",
                "qux_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((df["arm_name"] == arm_name).sum(), 1)

        # Check that all means and SEMs are not NaN
        self.assertFalse(df["foo_mean"].isna().any())
        self.assertFalse(df["foo_sem"].isna().any())
        self.assertFalse(df["bar_mean"].isna().any())
        self.assertFalse(df["bar_sem"].isna().any())

        # Check that all p_feasible are not NaN
        self.assertFalse(df["p_feasible"].isna().any())

        only_foo_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
        )
        # Check that the columns are correct
        self.assertEqual(
            set(only_foo_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(only_foo_df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((only_foo_df["arm_name"] == arm_name).sum(), 1)

        # Check that all means and SEMs are not NaN
        self.assertFalse(only_foo_df["foo_mean"].isna().any())
        self.assertFalse(only_foo_df["foo_sem"].isna().any())

        only_trial_0_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
            trial_index=0,
        )

        # Check that the columns are correct
        self.assertEqual(
            set(only_trial_0_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )

        # Check that we have one row per arm and that the arm appears only once
        self.assertEqual(len(only_trial_0_df), 1)
        self.assertEqual(only_trial_0_df.loc[0]["arm_name"], "0_0")

        # Check that all means and SEMs are not NaN
        self.assertFalse(only_trial_0_df["foo_mean"].isna().any())
        self.assertFalse(only_trial_0_df["foo_sem"].isna().any())
        self.assertFalse(only_trial_0_df["bar_mean"].isna().any())
        self.assertFalse(only_trial_0_df["bar_sem"].isna().any())

        with_additional_arms_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
            additional_arms=[
                Arm(parameters={"x1": 0.5, "x2": 0.5}),
                Arm(parameters={"x1": 0.25, "x2": 0.25}),
            ],
        )

        # Check that the columns are correct
        self.assertEqual(
            set(with_additional_arms_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(
            len(with_additional_arms_df), len(self.client._experiment.arms_by_name) + 2
        )
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((with_additional_arms_df["arm_name"] == arm_name).sum(), 1)

        # Check that all means and SEMs are not NaN
        self.assertFalse(with_additional_arms_df["foo_mean"].isna().any())
        self.assertFalse(with_additional_arms_df["foo_sem"].isna().any())
        self.assertFalse(with_additional_arms_df["bar_mean"].isna().any())
        self.assertFalse(with_additional_arms_df["bar_sem"].isna().any())

        with_only_additional_arms_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
            trial_index=-1,
            additional_arms=[
                Arm(parameters={"x1": 0.5, "x2": 0.5}),
                Arm(parameters={"x1": 0.25, "x2": 0.25}),
            ],
        )

        # Check that the columns are correct
        self.assertEqual(
            set(with_only_additional_arms_df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(with_only_additional_arms_df), 2)

        # Check that all means and SEMs are not NaN
        self.assertFalse(with_additional_arms_df["foo_mean"].isna().any())
        self.assertFalse(with_additional_arms_df["foo_sem"].isna().any())
        self.assertFalse(with_additional_arms_df["bar_mean"].isna().any())
        self.assertFalse(with_additional_arms_df["bar_sem"].isna().any())

        only_completed_trials_df = prepare_arm_data(
            experiment=self.client._experiment,
            metric_names=["foo", "bar"],
            use_model_predictions=True,
            adapter=self.client._generation_strategy.model,
            trial_statuses=[TrialStatus.COMPLETED],
        )

        # Check that we have two rows per arm and that each arm appears only once
        self.assertEqual(
            len(only_completed_trials_df), len(self.client._experiment.arms_by_name) - 1
        )

        # Check that all means are not NaN
        self.assertFalse(only_completed_trials_df["foo_mean"].isna().any())

        # Check that all SEMs are not NaN
        self.assertFalse(only_completed_trials_df["foo_sem"].isna().any())

    def test_relativize_data(self) -> None:
        df = pd.DataFrame(
            {
                "trial_index": [0, 0, 0],
                "arm_name": ["status_quo", "arm1", "arm2"],
                "foo_mean": [10.0, 12.0, 15.0],
                "foo_sem": [1.0, 1.2, 1.5],
                "bar_mean": [20.0, 22.0, 25.0],
                "bar_sem": [2.0, 2.2, 2.5],
            }
        )

        rel_df = _relativize_data(
            df=df, status_quo_df=df[df["arm_name"] == "status_quo"]
        )

        np.testing.assert_almost_equal(
            rel_df.loc[0, "foo_mean"], 0.0, decimal=1
        )  # status quo
        np.testing.assert_almost_equal(rel_df.loc[1, "foo_mean"], 0.2, decimal=1)
        np.testing.assert_almost_equal(rel_df.loc[2, "foo_mean"], 0.5, decimal=1)
        np.testing.assert_almost_equal(rel_df.loc[0, "foo_sem"], 0.1, decimal=1)
        np.testing.assert_almost_equal(rel_df.loc[1, "foo_sem"], 0.2, decimal=1)
        np.testing.assert_almost_equal(rel_df.loc[1, "bar_mean"], 0.1, decimal=1)
        np.testing.assert_almost_equal(rel_df.loc[1, "bar_sem"], 0.2, decimal=1)

    def test_relativize_data_multiple_trials(self) -> None:
        df = pd.DataFrame(
            {
                "trial_index": [0, 0, 1, 1],
                "arm_name": ["status_quo", "arm1", "status_quo", "arm2"],
                "foo_mean": [10.0, 12.0, 10.0, 15.0],
                "foo_sem": [1.0, 1.2, 1.0, 1.5],
                "bar_mean": [20.0, 22.0, 20.0, 25.0],
                "bar_sem": [2.0, 2.2, 2.0, 2.5],
            }
        )

        rel_df = _relativize_data(
            df=df, status_quo_df=df[df["arm_name"] == "status_quo"]
        )

        np.testing.assert_almost_equal(
            rel_df[(rel_df["trial_index"] == 0) & (rel_df["arm_name"] == "status_quo")][
                "foo_mean"
            ].iloc[0],
            0.0,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            rel_df[(rel_df["trial_index"] == 0) & (rel_df["arm_name"] == "arm1")][
                "foo_mean"
            ].iloc[0],
            0.2,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            rel_df[(rel_df["trial_index"] == 1) & (rel_df["arm_name"] == "status_quo")][
                "foo_mean"
            ].iloc[0],
            0.0,
            decimal=1,
        )
        np.testing.assert_almost_equal(
            rel_df[(rel_df["trial_index"] == 1) & (rel_df["arm_name"] == "arm2")][
                "foo_mean"
            ].iloc[0],
            0.5,
            decimal=1,
        )

    def test_truncate_label(self) -> None:
        with self.subTest("No truncation"):
            self.assertEqual(
                truncate_label(label="short:overall_END_ALL_CAP"),
                "short:overall_END_ALL_CAP",
            )
        with self.subTest("remove suffix"):
            self.assertEqual(
                truncate_label(label="not_as_short:overall_END_ALL_CAP", n=20),
                "not_as_short:overall",
            )
        with self.subTest("remove prefix"):
            self.assertEqual(
                truncate_label(
                    label="long_prefix:not_as_short:overall_END_ALL_CAP", n=20
                ),
                "not_as_short:overall",
            )
        with self.subTest("remove common words and keep two segments"):
            self.assertEqual(
                truncate_label(
                    label="long_prefix:short_prefix:not_as_short:overall_END_ALL_CAP",
                    n=30,
                ),
                "short_prefix:not_as_short",
            )
        with self.subTest("only one long label"):
            self.assertEqual(
                truncate_label(
                    label="thisismyextrabiglabelthatcantbehandled",
                    n=20,
                ),
                "thisismyextrabigl...",
            )

    @TestCase.ax_long_test(
        reason=(
            "Adapter.predict still too slow under @mock_botorch_optimize for this test"
        )
    )
    @mock_botorch_optimize
    def test_online(self) -> None:
        # Test ArmEffectsPlot can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for experiment in get_online_experiments():
            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        if use_model_predictions and with_additional_arms:
                            additional_arms = [
                                Arm(
                                    parameters={
                                        parameter_name: 0
                                        for parameter_name in (
                                            experiment.search_space.parameters.keys()
                                        )
                                    }
                                )
                            ]
                        else:
                            additional_arms = None

                        generation_strategy = (
                            get_default_generation_strategy_at_MBM_node(
                                experiment=experiment
                            )
                        )
                        generation_strategy.current_node._fit(experiment=experiment)
                        adapter = none_throws(generation_strategy.model)

                        _ = prepare_arm_data(
                            experiment=experiment,
                            metric_names=[*adapter.metric_names],
                            use_model_predictions=use_model_predictions,
                            adapter=adapter,
                            trial_index=trial_index,
                            additional_arms=additional_arms,
                        )

    @TestCase.ax_long_test(
        reason=(
            "Adapter.predict still too slow under @mock_botorch_optimize for this test"
        )
    )
    @mock_botorch_optimize
    def test_offline(self) -> None:
        # Test compute_arm_data can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        for experiment in get_offline_experiments():
            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        if use_model_predictions and with_additional_arms:
                            additional_arms = [
                                Arm(
                                    parameters={
                                        parameter_name: 0
                                        for parameter_name in (
                                            experiment.search_space.parameters.keys()
                                        )
                                    }
                                )
                            ]
                        else:
                            additional_arms = None

                        generation_strategy = (
                            get_default_generation_strategy_at_MBM_node(
                                experiment=experiment
                            )
                        )
                        generation_strategy.current_node._fit(experiment=experiment)
                        adapter = none_throws(generation_strategy.model)

                        _ = prepare_arm_data(
                            experiment=experiment,
                            metric_names=[*adapter.metric_names],
                            use_model_predictions=use_model_predictions,
                            adapter=adapter,
                            trial_index=trial_index,
                            additional_arms=additional_arms,
                        )
