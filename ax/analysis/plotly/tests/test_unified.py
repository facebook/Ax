# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from ax.analysis.plotly.arm_effects.unified import (
    ArmEffectsPlot,
    compute_arm_effects_adhoc,
)
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.arm import Arm
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_offline_experiments,
    get_online_experiments,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import (
    get_default_generation_strategy_at_MBM_node,
    get_sobol_MBM_MTGP_gs,
)
from pyre_extensions import assert_is_instance, none_throws


class TestArmEffectsPlot(TestCase):
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
        self.client.configure_optimization(objective="foo, bar")

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

    def test_validation(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Requested metrics .* are not present in the experiment."
        ):
            ArmEffectsPlot(metric_names=["foo", "bar", "baz"]).compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

        with self.assertRaisesRegex(
            UserInputError, "Trial with index .* not found in experiment."
        ):
            ArmEffectsPlot(trial_index=1998).compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

    def test_compute_raw(self) -> None:
        default_analysis = ArmEffectsPlot(use_model_predictions=False)

        cards = default_analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        # Check that we have cards for both metrics
        self.assertEqual(len(cards), 2)

        self.assertEqual(
            set(cards[0].df.columns),
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

        self.assertEqual(
            set(cards[1].df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "bar_mean",
                "bar_sem",
            },
        )

        for card in cards:
            # Check that we have one row per arm and that each arm appears only once
            self.assertEqual(len(card.df), len(self.client._experiment.arms_by_name))
            for arm_name in self.client._experiment.arms_by_name:
                self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are NaN
        self.assertTrue(cards[0].df["foo_sem"].isna().all())
        self.assertTrue(cards[1].df["bar_sem"].isna().all())

    def test_compute_with_modeled(self) -> None:
        default_analysis = ArmEffectsPlot(use_model_predictions=True)

        cards = default_analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        # Check that we have cards for both metrics
        self.assertEqual(len(cards), 2)

        self.assertEqual(
            set(cards[0].df.columns),
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

        self.assertEqual(
            set(cards[1].df.columns),
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "p_feasible",
                "bar_mean",
                "bar_sem",
            },
        )

        for card in cards:
            # Check that we have one row per arm and that each arm appears only once
            self.assertEqual(len(card.df), len(self.client._experiment.arms_by_name))
            for arm_name in self.client._experiment.arms_by_name:
                self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are not NaN
        self.assertFalse(cards[0].df["foo_sem"].isna().any())
        self.assertFalse(cards[1].df["bar_sem"].isna().any())

    def test_compute_adhoc(self) -> None:
        # Use the same kwargs for typical and adhoc
        kwargs = {
            "metric_names": ["foo", "bar"],
            "use_model_predictions": True,
            "additional_arms": [Arm(parameters={"x1": 0, "x2": 0})],
            "labels": {"foo": "f"},
        }
        # pyre-ignore[6]: Unsafe kwargs usage on purpose
        analysis = ArmEffectsPlot(**kwargs)

        cards = analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        adhoc_cards = compute_arm_effects_adhoc(
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
        # Test ArmEffectsPlot can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for experiment in get_online_experiments():
            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        for show_cumulative_best in [True, False]:
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
                            adapter = none_throws(generation_strategy.model)

                            analysis = ArmEffectsPlot(
                                metric_names=[*adapter.metric_names],
                                use_model_predictions=use_model_predictions,
                                trial_index=trial_index,
                                additional_arms=additional_arms,
                                show_cumulative_best=show_cumulative_best,
                            )

                            _ = analysis.compute(
                                experiment=experiment,
                                adapter=adapter,
                            )

    @TestCase.ax_long_test(
        reason=(
            "Adapter.predict still too slow under @mock_botorch_optimize for this test"
        )
    )
    @mock_botorch_optimize
    def test_offline(self) -> None:
        # Test ArmEffectsPlot can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        for experiment in get_offline_experiments():
            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        for show_cumulative_best in [True, False]:
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
                            adapter = none_throws(generation_strategy.model)

                            analysis = ArmEffectsPlot(
                                metric_names=[*adapter.metric_names],
                                use_model_predictions=use_model_predictions,
                                trial_index=trial_index,
                                additional_arms=additional_arms,
                                show_cumulative_best=show_cumulative_best,
                            )

                            _ = analysis.compute(
                                experiment=experiment,
                                adapter=adapter,
                            )


class TestArmEffectsPlotRel(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment(with_status_quo=True)
        self.generation_strategy = get_sobol_MBM_MTGP_gs()
        self.generation_strategy.experiment = self.experiment
        # Run 2 trials
        for _ in range(2):
            self.experiment.new_batch_trial(
                generator_runs=self.generation_strategy._gen_with_multiple_nodes(
                    experiment=self.experiment, n=3
                )
            ).set_status_quo_with_weight(
                status_quo=self.experiment.status_quo, weight=1.0
            ).mark_completed(unsafe=True)
            self.experiment.fetch_data()

    def test_compute_with_relativize(self) -> None:
        for use_model_predictions in [True, False]:
            with self.subTest(use_model_predictions=use_model_predictions):
                analysis = ArmEffectsPlot(
                    use_model_predictions=use_model_predictions, relativize=True
                )

                cards = analysis.compute(
                    experiment=self.experiment,
                    generation_strategy=self.generation_strategy,
                )

                self.assertEqual(len(cards), 1)

                self.assertEqual(
                    set(cards[0].df.columns),
                    {
                        "trial_index",
                        "arm_name",
                        "trial_status",
                        "generation_node",
                        "p_feasible",
                        "branin_mean",
                        "branin_sem",
                    },
                )

                for card in cards:
                    # Check that we have one row per arm and that each arm appears only
                    # once. Exclude status_quo since that is repeated between trials
                    card_arms = card.df[card.df.arm_name != "status_quo"].arm_name
                    experiment_arms = self.experiment.arms_by_name.copy()
                    experiment_arms.pop("status_quo")
                    self.assertEqual(len(card_arms), len(experiment_arms))
                    for arm_name in experiment_arms:
                        self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

                    self.assertFalse(card.df["branin_mean"].isna().any())
                    self.assertFalse(card.df["branin_sem"].isna().any())
