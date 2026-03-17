# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Sequence
from unittest.mock import patch

import pandas as pd
from ax.adapter.registry import Generators
from ax.analysis.diagnostics import (
    DiagnosticAnalysis,
    DIAGNOSTICS_CARDGROUP_SUBTITLE,
    DIAGNOSTICS_CARDGROUP_TITLE,
)
from ax.analysis.plotly.cross_validation import CrossValidationPlot
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.analysis_card import ErrorAnalysisCard
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import none_throws


class DiagnosticAnalysisTest(TestCase):
    def test_validate_applicable_state(self) -> None:
        analysis = DiagnosticAnalysis()

        # Should return an error message when no experiment is provided
        result = analysis.validate_applicable_state()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

        # Should return None when an experiment is provided (no trials/data required)
        experiment = Experiment(
            name="test",
            search_space=SearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="x",
                        parameter_type=ParameterType.FLOAT,
                        values=[0.0, 1.0],
                    ),
                ]
            ),
        )
        result = analysis.validate_applicable_state(experiment=experiment)
        self.assertIsNone(result)

    @mock_botorch_optimize
    def test_compute(self) -> None:
        # Set up a basic optimization with the Client
        client = Client()
        client.configure_experiment(
            name="booth_function",
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
        client.configure_optimization(objective="-1 * booth")
        client.configure_tracking_metrics(["tracking_m"])

        # Iterate well into the BO phase
        for _ in range(10):
            for trial_index, parameters in client.get_next_trials(max_trials=1).items():
                raw_data = {
                    # pyre-ignore[58]
                    "booth": (parameters["x1"] + 2.0 * parameters["x2"] - 7) ** 2.0
                    # pyre-ignore[58]
                    + (2.0 * parameters["x1"] + parameters["x2"] - 5.0) ** 2.0,
                    "tracking_m": 0.0,
                }
                client.complete_trial(trial_index=trial_index, raw_data=raw_data)

        experiment = client._experiment
        generation_strategy = client._generation_strategy

        # Compute with include_tracking_metrics=False (default)
        card_group = DiagnosticAnalysis(include_tracking_metrics=False).compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )

        self.assertEqual(card_group.title, DIAGNOSTICS_CARDGROUP_TITLE)
        self.assertEqual(card_group.subtitle, DIAGNOSTICS_CARDGROUP_SUBTITLE)

        # Should have CrossValidationPlot and GenerationStrategyGraph children
        child_names = [child.name for child in card_group.children]
        self.assertIn("CrossValidationPlot", child_names)
        self.assertIn("GenerationStrategyGraph", child_names)

        # No error cards
        for card in card_group.flatten():
            self.assertNotIsInstance(card, ErrorAnalysisCard)

        # --- Verify metric_names via patching CrossValidationPlot ---
        original_cv_init: Callable[..., None] = CrossValidationPlot.__init__

        captured_metric_names: list[Sequence[str] | None] = []

        def capturing_init(self: CrossValidationPlot, **kwargs: object) -> None:
            # pyre-ignore[6]: metric_names is Sequence[str] | None
            captured_metric_names.append(kwargs.get("metric_names"))
            original_cv_init(self, **kwargs)

        # include_tracking_metrics=False should only use optimization config metrics
        with patch.object(CrossValidationPlot, "__init__", capturing_init):
            DiagnosticAnalysis(include_tracking_metrics=False).compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
            )
        self.assertEqual(len(captured_metric_names), 1)
        self.assertIn("booth", none_throws(captured_metric_names[0]))
        self.assertNotIn("tracking_m", none_throws(captured_metric_names[0]))

        # include_tracking_metrics=True should use all experiment metrics
        captured_metric_names.clear()
        with patch.object(CrossValidationPlot, "__init__", capturing_init):
            card_group_tracking = DiagnosticAnalysis(
                include_tracking_metrics=True
            ).compute(
                experiment=experiment,
                generation_strategy=generation_strategy,
            )
        self.assertEqual(len(captured_metric_names), 1)
        self.assertIn("booth", none_throws(captured_metric_names[0]))
        self.assertIn("tracking_m", none_throws(captured_metric_names[0]))

        # Verify the card group is still valid
        self.assertEqual(card_group_tracking.title, DIAGNOSTICS_CARDGROUP_TITLE)

        # --- Without generation_strategy: no GenerationStrategyGraph ---
        card_group_no_gs = DiagnosticAnalysis().compute(
            experiment=experiment,
            generation_strategy=None,
        )
        child_names_no_gs = [child.name for child in card_group_no_gs.children]
        self.assertIn("CrossValidationPlot", child_names_no_gs)
        self.assertNotIn("GenerationStrategyGraph", child_names_no_gs)

    def test_compute_bandit(self) -> None:
        experiment = Experiment(
            name="bandit_test",
            search_space=SearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        values=[-10.0, -5.0, 0.0, 5.0, 10.0],
                    ),
                    ChoiceParameter(
                        name="x2",
                        parameter_type=ParameterType.FLOAT,
                        values=[-10.0, -5.0, 0.0, 5.0, 10.0],
                    ),
                ]
            ),
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=Metric(name="booth"),
                    minimize=True,
                )
            ),
        )

        # Create batch trials with arms
        arm_configs = [
            [(-10.0, -10.0), (0.0, 0.0), (10.0, 10.0)],
            [(-10.0, 10.0), (10.0, -10.0), (5.0, 5.0)],
        ]

        data_rows = []
        for arm_coords in arm_configs:
            arms = [Arm(parameters={"x1": x1, "x2": x2}) for x1, x2 in arm_coords]
            trial = experiment.new_batch_trial()
            trial.add_arms_and_weights(arms=arms).mark_running(no_runner_required=True)

            for arm in trial.arms:
                x1, x2 = float(arm.parameters["x1"]), float(arm.parameters["x2"])
                data_rows.append(
                    {
                        "trial_index": trial.index,
                        "arm_name": arm.name,
                        "metric_name": "booth",
                        "metric_signature": "booth",
                        "mean": (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2,
                        "sem": 0.0,
                    }
                )

        experiment.attach_data(Data(df=pd.DataFrame(data_rows)))

        # Set up bandit generation strategy
        factorial_node = GenerationNode(
            name="FACTORIAL",
            generator_specs=[
                GeneratorSpec(generator_enum=Generators.FACTORIAL),
            ],
            transition_criteria=[
                MinTrials(
                    threshold=1,
                    transition_to="EMPIRICAL_BAYES_THOMPSON_SAMPLING",
                )
            ],
        )
        eb_ts_node = GenerationNode(
            name="EMPIRICAL_BAYES_THOMPSON_SAMPLING",
            generator_specs=[
                GeneratorSpec(generator_enum=Generators.EMPIRICAL_BAYES_THOMPSON),
            ],
            transition_criteria=None,
        )
        bandit_gs = GenerationStrategy(
            name=Keys.FACTORIAL_PLUS_EMPIRICAL_BAYES_THOMPSON_SAMPLING,
            nodes=[factorial_node, eb_ts_node],
        )

        card_group = DiagnosticAnalysis().compute(
            experiment=experiment,
            generation_strategy=bandit_gs,
        )

        child_names = [child.name for child in card_group.children]

        # Bandit experiment should NOT include CrossValidationPlot
        self.assertNotIn("CrossValidationPlot", child_names)

        # GenerationStrategyGraph should still be included (GS is provided)
        self.assertIn("GenerationStrategyGraph", child_names)

        # Verify card group metadata
        self.assertEqual(card_group.title, DIAGNOSTICS_CARDGROUP_TITLE)
        self.assertEqual(card_group.subtitle, DIAGNOSTICS_CARDGROUP_SUBTITLE)
