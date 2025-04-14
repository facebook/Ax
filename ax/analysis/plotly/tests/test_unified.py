# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from ax.analysis.plotly.arm_effects.unified import ArmEffectsPlot
from ax.api.client import Client
from ax.api.configs import ExperimentConfig, ParameterType, RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_sobol_MBM_MTGP_gs
from pyre_extensions import assert_is_instance


class TestArmEffectsPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()

        self.client = Client()
        self.client.configure_experiment(
            experiment_config=ExperimentConfig(
                name="test_experiment",
                parameters=[
                    RangeParameterConfig(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                    RangeParameterConfig(
                        name="x2",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                ],
            )
        )
        self.client.configure_optimization(objective="foo, bar")

        # Get two trials and fail one, giving us a ragged structure
        self.client.get_next_trials(maximum_trials=2)
        self.client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 2.0})
        self.client.mark_trial_failed(trial_index=1)

        # Complete 5 trials successfully
        for _ in range(5):
            for trial_index, parameterization in self.client.get_next_trials().items():
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
