# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.constraint_feasibility import ConstraintFeasibilityPlot
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.arm import Arm
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_offline_experiments,
    get_online_experiments,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


class TestConstraintFeasibilityPlot(TestCase):
    def test_validate_applicable_state(self) -> None:
        self.assertIn(
            "Requires an Experiment",
            none_throws(ConstraintFeasibilityPlot().validate_applicable_state()),
        )

        self.assertIn(
            "must have at least one OutcomeConstraint",
            none_throws(
                ConstraintFeasibilityPlot().validate_applicable_state(
                    experiment=get_branin_experiment(
                        with_trial=True,
                        with_completed_trial=True,
                        has_optimization_config=True,
                    )
                )
            ),
        )

        self.assertIn(
            "Must provide either a GenerationStrategy or an Adapter",
            none_throws(
                ConstraintFeasibilityPlot(
                    use_model_predictions=True
                ).validate_applicable_state(
                    experiment=get_branin_experiment(
                        with_trial=True,
                        with_completed_trial=True,
                        has_optimization_config=True,
                        with_absolute_constraint=True,
                    )
                )
            ),
        )

        self.assertIn(
            "Cannot provide additional arms",
            none_throws(
                ConstraintFeasibilityPlot(
                    use_model_predictions=False,
                    additional_arms=[Arm(parameters={"foo": 1.0})],
                ).validate_applicable_state(
                    experiment=get_branin_experiment(
                        with_trial=True,
                        with_completed_trial=True,
                        has_optimization_config=True,
                        with_absolute_constraint=True,
                    )
                )
            ),
        )

    @mock_botorch_optimize
    def test_compute(self) -> None:
        client = Client()

        client.configure_experiment(
            name="foo",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    bounds=(-10.0, 10.0),
                    parameter_type="float",
                ),
                RangeParameterConfig(
                    name="x2",
                    bounds=(-10.0, 10.0),
                    parameter_type="float",
                ),
            ],
        )

        # Configure with multiple constraints to test per-constraint p_feasible
        client.configure_optimization(
            objective="foo",
            outcome_constraints=["x1_plus_x2 <= 1", "x1_squared <= 4"],
        )

        # Add a feasible trial
        client.complete_trial(
            trial_index=client.attach_trial(parameters={"x1": 0.0, "x2": 0.0}),
            raw_data={"foo": 0.0, "x1_plus_x2": 0, "x1_squared": 0},
        )

        # Add an infeasible trial (violates first constraint)
        client.complete_trial(
            trial_index=client.attach_trial(parameters={"x1": 1.0, "x2": 1.0}),
            raw_data={"foo": 0.0, "x1_plus_x2": 2, "x1_squared": 1},
        )

        # Add another trial (violates second constraint)
        client.complete_trial(
            trial_index=client.attach_trial(parameters={"x1": 3.0, "x2": -2.0}),
            raw_data={"foo": 0.0, "x1_plus_x2": 1, "x1_squared": 9},
        )

        # Run more trials to get out of Sobol node
        for _ in range(5):
            for trial_index, parameters in client.get_next_trials(max_trials=1).items():
                x1 = assert_is_instance(parameters["x1"], float)
                x2 = assert_is_instance(parameters["x2"], float)
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "foo": 0,
                        "x1_plus_x2": x1 + x2,
                        "x1_squared": x1 * x1,
                    },
                )

        cards = client.compute_analyses(
            [
                ConstraintFeasibilityPlot(use_model_predictions=False),
                ConstraintFeasibilityPlot(
                    use_model_predictions=True,
                    additional_arms=[Arm(parameters={"x1": -1.0, "x2": -1.0})],
                ),
            ]
        )
        observed_card = assert_is_instance(cards[0], PlotlyAnalysisCard)
        predicted_card = assert_is_instance(cards[1], PlotlyAnalysisCard)

        self.assertEqual(
            "Observed Constraint Feasibility by Constraint", observed_card.title
        )
        self.assertEqual(
            "Predicted Constraint Feasibility by Constraint", predicted_card.title
        )

        self.assertEqual(len(observed_card.df), len(client._experiment.trials))
        self.assertEqual(len(predicted_card.df), len(client._experiment.trials) + 1)

        # Check that we have per-constraint p_feasible columns
        self.assertIn("p_feasible_x1_plus_x2", observed_card.df.columns)
        self.assertIn("p_feasible_x1_squared", observed_card.df.columns)
        self.assertIn("p_feasible_x1_plus_x2", predicted_card.df.columns)
        self.assertIn("p_feasible_x1_squared", predicted_card.df.columns)

        expected_columns = {
            "trial_index",
            "arm_name",
            "trial_status",
            "generation_node",
            "overall p(feasible)",
            "p_feasible_x1_plus_x2",
            "p_feasible_x1_squared",
            "p_feasible_mean",
            "p_feasible_sem",
            "foo_mean",
            "foo_sem",
        }

        self.assertEqual({*observed_card.df.columns}, expected_columns)
        self.assertEqual({*predicted_card.df.columns}, expected_columns)

    @mock_botorch_optimize
    def test_offline(self) -> None:
        for experiment in get_offline_experiments():
            # Skip if no outcome constraints
            if (
                len(none_throws(experiment.optimization_config).outcome_constraints)
                == 0
            ):
                continue

            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        if use_model_predictions and with_additional_arms:
                            additional_arms = [
                                Arm(
                                    parameters=dict.fromkeys(
                                        experiment.search_space.parameters.keys(), 0
                                    )
                                )
                            ]
                        else:
                            additional_arms = None

                        generation_strategy = (
                            get_default_generation_strategy_at_MBM_node(
                                experiment=experiment
                            )
                        )

                        analysis = ConstraintFeasibilityPlot(
                            use_model_predictions=use_model_predictions,
                            trial_index=trial_index,
                            additional_arms=additional_arms,
                        )

                        _ = analysis.compute(
                            experiment=experiment,
                            generation_strategy=generation_strategy,
                        )

    def test_online(self) -> None:
        for experiment in get_online_experiments():
            # Skip if no outcome constraints
            if (
                len(none_throws(experiment.optimization_config).outcome_constraints)
                == 0
            ):
                continue

            for use_model_predictions in [True, False]:
                for trial_index in [None, 0]:
                    for with_additional_arms in [True, False]:
                        if use_model_predictions and with_additional_arms:
                            additional_arms = [
                                Arm(
                                    parameters=dict.fromkeys(
                                        experiment.search_space.parameters.keys(), 0
                                    )
                                )
                            ]
                        else:
                            additional_arms = None

                        generation_strategy = (
                            get_default_generation_strategy_at_MBM_node(
                                experiment=experiment
                            )
                        )

                        analysis = ConstraintFeasibilityPlot(
                            use_model_predictions=use_model_predictions,
                            trial_index=trial_index,
                            additional_arms=additional_arms,
                        )

                        _ = analysis.compute(
                            experiment=experiment,
                            generation_strategy=generation_strategy,
                        )
