# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from datetime import datetime

from ax.analysis.analysis_card import ErrorAnalysisCard
from ax.analysis.overview import OverviewAnalysis
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.plotly.scatter import ScatterPlot
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
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

        analysis = OverviewAnalysis()

        for experiment in get_online_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            card_group = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
            for card in card_group.flatten():
                self.assertNotIsInstance(card, ErrorAnalysisCard)
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
