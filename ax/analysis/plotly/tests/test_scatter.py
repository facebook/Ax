# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import json
from itertools import product

from ax.adapter.registry import Generators
from ax.analysis.plotly.scatter import compute_scatter_adhoc, ScatterPlot
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


class TestScatterPlot(TestCase):
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
            objective="foo", outcome_constraints=["bar >= -0.5"]
        )

        # Get two trials and fail one, giving us a ragged structure
        self.client.get_next_trials(max_trials=2)
        self.client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 2.0})
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
                    },
                )

    def test_trial_statuses_behavior(self) -> None:
        # When neither trial_statuses nor trial_index is provided,
        # should use default statuses (excluding ABANDONED and STALE)
        analysis = ScatterPlot(x_metric_name="foo", y_metric_name="bar")
        expected_statuses = {*TrialStatus} - {TrialStatus.ABANDONED, TrialStatus.STALE}
        self.assertEqual(set(none_throws(analysis.trial_statuses)), expected_statuses)

        # When trial_statuses is explicitly provided, it should be used
        explicit_statuses = [TrialStatus.COMPLETED, TrialStatus.RUNNING]
        analysis = ScatterPlot(
            x_metric_name="foo", y_metric_name="bar", trial_statuses=explicit_statuses
        )
        self.assertEqual(analysis.trial_statuses, explicit_statuses)

        # When trial_index is provided (and trial_statuses is None),
        # trial_statuses should be None to allow filtering by trial_index
        analysis = ScatterPlot(x_metric_name="foo", y_metric_name="bar", trial_index=0)
        self.assertIsNone(analysis.trial_statuses)

    def test_validation(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Requested metrics .* are not present in the experiment."
        ):
            ScatterPlot(x_metric_name="foo", y_metric_name="baz").compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

        with self.assertRaisesRegex(
            UserInputError, "Trial with index .* not found in experiment."
        ):
            ScatterPlot(
                x_metric_name="foo", y_metric_name="bar", trial_index=1998
            ).compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

    def test_compute_raw(self) -> None:
        default_analysis = ScatterPlot(
            x_metric_name="foo", y_metric_name="bar", use_model_predictions=False
        )

        card = default_analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        self.assertEqual(
            set(card.df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "fail_reason",
                "generation_node",
                "p_feasible_mean",
                "p_feasible_sem",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )
        self.assertIsNotNone(card.blob)

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(card.df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are NaN
        self.assertTrue(card.df["foo_sem"].isna().all())
        self.assertTrue(card.df["bar_sem"].isna().all())

    def test_show_pareto_frontier(self) -> None:
        analysis = ScatterPlot(
            x_metric_name="foo",
            y_metric_name="bar",
            show_pareto_frontier=True,
            use_model_predictions=False,
        )
        card = analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )
        fig_data = json.loads(none_throws(card.blob))
        pareto_traces = [
            trace
            for trace in fig_data.get("data", [])
            if trace.get("name") == "Pareto Frontier"
        ]
        self.assertEqual(len(pareto_traces), 1)
        self.assertTrue(pareto_traces[0].get("showlegend"))

    def test_compute_with_modeled(self) -> None:
        default_analysis = ScatterPlot(
            x_metric_name="foo", y_metric_name="bar", use_model_predictions=True
        )

        card = default_analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        self.assertEqual(
            set(card.df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "fail_reason",
                "generation_node",
                "p_feasible_mean",
                "p_feasible_sem",
                "foo_mean",
                "foo_sem",
                "bar_mean",
                "bar_sem",
            },
        )

        self.assertIsNotNone(card.blob)

        # Check that we have one row per arm and that each arm appears only once
        self.assertEqual(len(card.df), len(self.client._experiment.arms_by_name))
        for arm_name in self.client._experiment.arms_by_name:
            self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are not NaN
        self.assertFalse(card.df["foo_sem"].isna().any())
        self.assertFalse(card.df["bar_sem"].isna().any())

    def test_compute_adhoc(self) -> None:
        # Use the same kwargs for typical and adhoc
        kwargs = {
            "x_metric_name": "foo",
            "y_metric_name": "bar",
            "use_model_predictions": True,
            "additional_arms": [Arm(parameters={"x1": 0, "x2": 0})],
            "labels": {"foo": "f"},
        }
        # pyre-ignore[6]: Unsafe kwargs usage on purpose
        analysis = ScatterPlot(**kwargs)

        cards = analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        adhoc_cards = compute_scatter_adhoc(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
            # pyre-ignore[6]: Unsafe kwargs usage on purpose
            **kwargs,
        )

        self.assertEqual(cards, adhoc_cards)

    @TestCase.ax_long_test(
        reason=(
            "Adapter.predict still too slow under @mock_botorch_optimize for this test"
        )
    )
    @mock_botorch_optimize
    def test_online(self) -> None:
        # Test ScatterPlot can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for experiment in get_online_experiments():
            # Skip experiments with fewer than 2 metrics
            if len(experiment.metrics) < 2:
                continue
            arm = Generators.SOBOL(experiment=experiment).gen(n=1).arms[0]
            arm.name = "additional_arm"
            for (
                use_model_predictions,
                trial_index,
                with_additional_arms,
                show_pareto_frontier,
            ) in product(
                [True, False],
                [None, 0],
                [True, False],
                [True, False],
            ):
                if use_model_predictions and with_additional_arms:
                    additional_arms = [arm]
                else:
                    additional_arms = None

                generation_strategy = get_default_generation_strategy_at_MBM_node(
                    experiment=experiment
                )
                generation_strategy.current_node._fit(experiment=experiment)
                adapter = none_throws(generation_strategy.adapter)

                metric_names = [
                    experiment.signature_to_metric[signature].name
                    for signature in adapter.metric_signatures
                ]

                x_metric_name, y_metric_name = metric_names[:2]

                analysis = ScatterPlot(
                    x_metric_name=x_metric_name,
                    y_metric_name=y_metric_name,
                    use_model_predictions=use_model_predictions,
                    trial_index=trial_index,
                    additional_arms=additional_arms,
                    show_pareto_frontier=show_pareto_frontier,
                )

                cards = analysis.compute(
                    experiment=experiment,
                    adapter=adapter,
                )
                if with_additional_arms and use_model_predictions:
                    # validate that we plotted the additional arm
                    self.assertTrue(
                        all(
                            any(
                                arm.name in dat["text"][0]
                                for dat in json.loads(card.blob)["data"]
                            )
                            for card in cards.flatten()
                        )
                    )

    @TestCase.ax_long_test(
        reason=(
            "Adapter.predict still too slow under @mock_botorch_optimize for this test"
        )
    )
    @mock_botorch_optimize
    def test_offline(self) -> None:
        # Test ScatterPlot can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        for experiment in get_offline_experiments():
            # Skip experiments with fewer than 2 metrics
            if len(experiment.metrics) < 2:
                continue

            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        for show_pareto_frontier in [True, False]:
                            if use_model_predictions and with_additional_arms:
                                additional_arms = [
                                    Arm(
                                        parameters={
                                            parameter_name: 0
                                            for parameter_name in (
                                                experiment.search_space.parameters.keys()  # noqa E501
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
                            adapter = none_throws(generation_strategy.adapter)

                            metric_names = [
                                experiment.signature_to_metric[signature].name
                                for signature in adapter.metric_signatures
                            ]
                            x_metric_name, y_metric_name = metric_names[:2]

                            analysis = ScatterPlot(
                                x_metric_name=x_metric_name,
                                y_metric_name=y_metric_name,
                                use_model_predictions=use_model_predictions,
                                trial_index=trial_index,
                                additional_arms=additional_arms,
                                show_pareto_frontier=show_pareto_frontier,
                            )

                            _ = analysis.compute(
                                experiment=experiment,
                                adapter=adapter,
                            )
