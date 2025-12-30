# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

import json
from itertools import product

from ax.adapter.registry import Generators
from ax.analysis.plotly.arm_effects import ArmEffectsPlot, compute_arm_effects_adhoc
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.arm import Arm
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_non_failed_arm_names,
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
        # should use default statuses (excluding ABANDONED, STALE, and FAILED)
        analysis = ArmEffectsPlot(metric_name="foo")
        expected_statuses = {*TrialStatus} - {
            TrialStatus.ABANDONED,
            TrialStatus.STALE,
            TrialStatus.FAILED,
        }
        self.assertEqual(set(none_throws(analysis.trial_statuses)), expected_statuses)

        # When trial_statuses is explicitly provided, it should be used
        explicit_statuses = [TrialStatus.COMPLETED, TrialStatus.RUNNING]
        analysis = ArmEffectsPlot(metric_name="foo", trial_statuses=explicit_statuses)
        self.assertEqual(analysis.trial_statuses, explicit_statuses)

        # When trial_index is provided (and trial_statuses is None),
        # trial_statuses should be None to allow filtering by trial_index
        analysis = ArmEffectsPlot(metric_name="foo", trial_index=0)
        self.assertIsNone(analysis.trial_statuses)

    def test_validation(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Requested metrics .* are not present in the experiment."
        ):
            ArmEffectsPlot(metric_name="baz").compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

        with self.assertRaisesRegex(
            UserInputError, "Trial with index .* not found in experiment."
        ):
            ArmEffectsPlot(metric_name="foo", trial_index=1998).compute(
                experiment=self.client._experiment,
                generation_strategy=self.client._generation_strategy,
            )

    def test_compute_raw(self) -> None:
        default_analysis = ArmEffectsPlot(
            metric_name="foo", use_model_predictions=False
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
                "foo_mean",
                "foo_sem",
            },
        )

        # Check that we have one row per arm from non-failed trials and that each
        # arm appears only once
        non_failed_arms = get_non_failed_arm_names(self.client._experiment)
        self.assertEqual(len(card.df), len(non_failed_arms))
        for arm_name in non_failed_arms:
            self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are NaN
        self.assertTrue(card.df["foo_sem"].isna().all())

    def test_compute_with_modeled(self) -> None:
        default_analysis = ArmEffectsPlot(metric_name="foo", use_model_predictions=True)

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
                "foo_mean",
                "foo_sem",
            },
        )

        # Check that we have one row per arm from non-failed trials and that each
        # arm appears only once
        non_failed_arms = get_non_failed_arm_names(self.client._experiment)
        self.assertEqual(len(card.df), len(non_failed_arms))
        for arm_name in non_failed_arms:
            self.assertEqual((card.df["arm_name"] == arm_name).sum(), 1)

        # Check that all SEMs are not NaN
        self.assertFalse(card.df["foo_sem"].isna().any())

    def test_compute_adhoc(self) -> None:
        # Use the same kwargs for typical and adhoc
        kwargs = {
            "metric_name": "foo",
            "use_model_predictions": True,
            "additional_arms": [Arm(parameters={"x1": 0, "x2": 0})],
            "label": "f",
        }
        # pyre-ignore[6]: Unsafe kwargs usage on purpose
        analysis = ArmEffectsPlot(**kwargs)

        cards = analysis.compute(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
        )

        metric_name = assert_is_instance(kwargs.pop("metric_name"), str)
        adhoc_cards = compute_arm_effects_adhoc(
            experiment=self.client._experiment,
            generation_strategy=self.client._generation_strategy,
            metric_names=[metric_name],
            labels={metric_name: assert_is_instance(kwargs.pop("label"), str)},
            # pyre-ignore[6]: Unsafe kwargs usage on purpose
            **kwargs,
        )

        self.assertEqual(cards, adhoc_cards.children[0])

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
            arm = Generators.SOBOL(experiment=experiment).gen(n=1).arms[0]
            arm.name = "additional_arm"
            for (
                use_model_predictions,
                trial_index,
                with_additional_arms,
            ) in product([True, False], [None, 0], [True, False]):
                if use_model_predictions and with_additional_arms:
                    additional_arms = [arm]
                else:
                    additional_arms = None
                generation_strategy = get_default_generation_strategy_at_MBM_node(
                    experiment=experiment
                )
                generation_strategy.current_node._fit(experiment=experiment)
                adapter = none_throws(generation_strategy.adapter)

                for signature in adapter.metric_signatures:
                    metric_name = adapter._experiment.signature_to_metric[
                        signature
                    ].name

                    analysis = ArmEffectsPlot(
                        metric_name=metric_name,
                        use_model_predictions=use_model_predictions,
                        trial_index=trial_index,
                        additional_arms=additional_arms,
                    )

                    card = analysis.compute(
                        experiment=experiment,
                        adapter=adapter,
                    )
                    if with_additional_arms and use_model_predictions:
                        # validate that we plotted the additional arm
                        self.assertIn(
                            arm.name,
                            json.loads(card.blob)["layout"]["xaxis"]["ticktext"],
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

                        model_metric_names = [
                            adapter._experiment.signature_to_metric[signature].name
                            for signature in adapter.metric_signatures
                        ]
                        for metric_name in model_metric_names:
                            analysis = ArmEffectsPlot(
                                metric_name=metric_name,
                                use_model_predictions=use_model_predictions,
                                trial_index=trial_index,
                                additional_arms=additional_arms,
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
                generator_runs=self.generation_strategy.gen(
                    experiment=self.experiment, n=3
                )[0]
            ).add_status_quo_arm(weight=1.0).mark_completed(unsafe=True)
            self.experiment.fetch_data()

    def test_compute_with_relativize(self) -> None:
        for use_model_predictions in [True, False]:
            with self.subTest(use_model_predictions=use_model_predictions):
                analysis = ArmEffectsPlot(
                    metric_name="branin",
                    use_model_predictions=use_model_predictions,
                    relativize=True,
                )

                cards = analysis.compute(
                    experiment=self.experiment,
                    generation_strategy=self.generation_strategy,
                ).flatten()

                self.assertEqual(len(cards), 1)

                self.assertEqual(
                    set(cards[0].df.columns),
                    {
                        "trial_index",
                        "arm_name",
                        "trial_status",
                        "fail_reason",
                        "generation_node",
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

                    # The title should include status quo name when relativize is True
                    expected_prefix = "Modeled" if use_model_predictions else "Observed"
                    expected_suffix = 'relative to "status_quo"'
                    self.assertIn(expected_prefix, card.title)
                    self.assertIn("Arm Effects on branin", card.title)
                    self.assertIn(expected_suffix, card.title)
