# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from datetime import datetime
from typing import Optional

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.registry import Generators

from ax.analysis.analysis_card import ErrorAnalysisCard
from ax.analysis.overview import OverviewAnalysis
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.plotly.scatter import ScatterPlot
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.optimization_config import Objective, OptimizationConfig
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
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node


class TestOverview(TestCase):
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
        # Iterate well into the BO phase
        for _ in range(10):
            for trial_index, parameters in client.get_next_trials(max_trials=1).items():
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        # pyre-ignore[58]
                        "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2
                        # pyre-ignore[58]
                        + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2
                    },
                )

        # Add a CANDIDATE batch trial to produce some trial analysis cards
        client._experiment.new_batch_trial()

        # Add metric fetching errors to produce some healthcheck analyses
        client._experiment._metric_fetching_errors = {
            (0, "booth"): {
                "trial_index": 0,
                "metric_name": "booth",
                "reason": "This is a test",
                "timestamp": datetime.now().isoformat(),
                "traceback": "Test traceback",
            }
        }

        card = OverviewAnalysis().compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        )

        children_names = [child.name for child in card.children]
        self.assertIn("ResultsAnalysis", children_names)
        self.assertIn("InsightsAnalysis", children_names)
        self.assertIn("DiagnosticAnalysis", children_names)
        self.assertIn("AllTrialsAnalysis", children_names)
        self.assertIn("HealthchecksAnalysis", children_names)

    @mock_botorch_optimize
    def test_online(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an online setting.
        try:
            import pymoo  # noqa: F401

            expect_error_for_obj_pfeasible = False
        except ImportError:
            expect_error_for_obj_pfeasible = True
        analysis = OverviewAnalysis()

        for experiment in get_online_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            card_group = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            # we expect the objective vs pfeasible plot to error out if it is an
            # experiment with one objective and constranits and pymoo is not
            # installed.
            total_errors = sum(
                isinstance(card, ErrorAnalysisCard) for card in card_group.flatten()
            )
            self.assertEqual(
                total_errors,
                1
                if (expect_error_for_obj_pfeasible and not experiment.is_moo_problem)
                else 0,
            )
            for card in card_group.flatten():
                # TODO: add more AnalysisCard types when they support relativization
                if isinstance(card, (ArmEffectsPlot, ScatterPlot)):
                    self.assertIn("Relativized", card.title)

    @mock_botorch_optimize
    def test_offline(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        analysis = OverviewAnalysis()

        for experiment in get_offline_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            card_group = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            for card in card_group.flatten():
                self.assertNotIsInstance(card, ErrorAnalysisCard)
                self.assertNotIn("Relativized", card.title)

    def test_bandit_experiment_dispatch(self) -> None:
        experiment = Experiment(
            name="bandit_test",
            search_space=SearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        values=[-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
                    ),
                    ChoiceParameter(
                        name="x2",
                        parameter_type=ParameterType.FLOAT,
                        values=[-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0],
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

        # Create multi-arm trials to enable bandit analysis
        arm_configs = [
            [(-10.0, -10.0), (0.0, 0.0), (10.0, 10.0)],
            [(-10.0, 10.0), (10.0, -10.0), (5.0, 5.0)],
        ]

        trials = []
        data_rows = []

        for arm_coords in arm_configs:
            # Create batch trials with arms
            arms = [Arm(parameters={"x1": x1, "x2": x2}) for x1, x2 in arm_coords]
            trial = experiment.new_batch_trial()
            trial.add_arms_and_weights(arms=arms)
            trials.append(trial)

            # Generate data rows in same loop
            for arm in trial.arms:
                x1, x2 = float(arm.parameters["x1"]), float(arm.parameters["x2"])
                data_rows.append(
                    {
                        "trial_index": trial.index,
                        "arm_name": arm.name,
                        "metric_name": "booth",
                        "mean": (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2,
                        "sem": 0.0,
                    }
                )

        data = Data(df=pd.DataFrame(data_rows))

        # Attach data to the experiment
        experiment.attach_data(data)

        factorial_gen_node = GenerationNode(
            node_name="FACTORIAL",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.FACTORIAL,
                )
            ],
            transition_criteria=[
                MinTrials(
                    threshold=1,
                    transition_to="EMPIRICAL_BAYES_THOMPSON_SAMPLING",
                )
            ],
        )

        eb_ts_gen_node = GenerationNode(
            node_name="EMPIRICAL_BAYES_THOMPSON_SAMPLING",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.EMPIRICAL_BAYES_THOMPSON,
                )
            ],
            transition_criteria=None,
        )

        # Setup Bandit Generation Strategy
        bandit_gs = GenerationStrategy(
            name=Keys.FACTORIAL_PLUS_EMPIRICAL_BAYES_THOMPSON_SAMPLING,
            nodes=[factorial_gen_node, eb_ts_gen_node],
        )

        # Set the current node to the empirical bayes thompson node as that is
        # what is used to generate the marginal effects plot
        bandit_gs._curr = bandit_gs._nodes[1]

        # Compute the overview card
        card = OverviewAnalysis().compute(
            experiment=experiment,
            generation_strategy=bandit_gs,
        )

        # Flatten all cards to check for bandit-specific analyses
        all_cards = card.flatten()
        card_names = [c.name for c in all_cards]

        # Check that bandit-specific analyses are included
        self.assertIn("BanditRollout", card_names)
        self.assertIn("MarginalEffectsPlot", card_names)

        # Ensure BanditRollout card is not an error card
        bandit_rollout_cards = [c for c in all_cards if c.name == "BanditRollout"]
        self.assertEqual(len(bandit_rollout_cards), 1)
        self.assertNotIsInstance(bandit_rollout_cards[0], ErrorAnalysisCard)

        marginal_effects_cards = [
            c for c in all_cards if c.name == "MarginalEffectsPlot"
        ]

        # Since there might be multiple MarginalEffectsPlot cards, check that at
        # least one exists and that none of them are error cards
        self.assertGreaterEqual(len(marginal_effects_cards), 1)

        # Check if adapter has predict properly implemented
        def has_predict_implemented(adapter: Optional[Adapter]) -> bool:
            """Check if adapter has predict method properly implemented."""
            if adapter is None or not hasattr(adapter, "predict"):
                return False
            try:
                adapter.predict(observation_features=[])
                return True
            except NotImplementedError:
                return False
            except Exception:
                # Other exceptions suggest method is implemented but needs proper setup
                return True

        predict_implemented = has_predict_implemented(bandit_gs.adapter)

        if predict_implemented:
            # Check that the marginal effects cards are not error cards
            for card in marginal_effects_cards:
                self.assertNotIsInstance(card, ErrorAnalysisCard)
