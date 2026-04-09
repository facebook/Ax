# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.analysis.plotly.utility_ranking import UtilityRankingPlot
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.metric import Metric
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance


class TestUtilityRankingPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()

        self.client = Client()
        self.client.configure_experiment(
            name="test_utility_ranking",
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
        self.pairwise_name = Keys.PAIRWISE_PREFERENCE_QUERY.value
        self.client.configure_optimization(objective="foo")

        # Generate 2 initial Sobol trials then 5 MBM trials (the latter
        # triggers botorch optimization, hence @mock_botorch_optimize on setUp).
        self.client.get_next_trials(max_trials=2)
        self.client.complete_trial(trial_index=0, raw_data={"foo": 1.0})
        self.client.complete_trial(trial_index=1, raw_data={"foo": 0.5})
        for _ in range(5):
            for trial_index, parameterization in self.client.get_next_trials(
                max_trials=1
            ).items():
                self.client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "foo": assert_is_instance(parameterization["x1"], float),
                    },
                )

    def test_validate_applicable_state(self) -> None:
        analysis = UtilityRankingPlot(metric_name="foo")

        # No experiment
        result = analysis.validate_applicable_state()
        self.assertIsNotNone(result)

        # No adapter available (GS hasn't been fit)
        result = analysis.validate_applicable_state(
            experiment=self.client._experiment,
        )
        self.assertIsNotNone(result)

    def test_compute_produces_card(self) -> None:
        """UtilityRankingPlot should produce a valid analysis card with a ranked
        bar chart of model-predicted utility."""
        experiment = self.client._experiment
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        card = UtilityRankingPlot(metric_name="foo").compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )

        self.assertIn("Utility Ranking", card.title)
        # DataFrame should have one row per arm with predictions
        self.assertGreater(len(card.df), 0)
        self.assertIn("foo_mean", card.df.columns)
        self.assertIn("foo_sem", card.df.columns)
        self.assertIn("arm_name", card.df.columns)
        self.assertIn("trial_index", card.df.columns)

    def test_compute_with_preference_metric_name(self) -> None:
        """Verify that UtilityRankingPlot correctly flags unmodeled preference
        metrics via validate_applicable_state."""
        experiment = self.client._experiment
        experiment.add_tracking_metric(Metric(name=self.pairwise_name))

        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        # This metric isn't modeled, so validate_applicable_state should flag it
        result = UtilityRankingPlot(
            metric_name=self.pairwise_name
        ).validate_applicable_state(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )
        self.assertIsNotNone(result)

    def test_ranking_order(self) -> None:
        """Arms should be ranked from highest to lowest predicted utility in
        the figure data (ascending in the bar chart y-axis for plotly)."""
        experiment = self.client._experiment
        generation_strategy = get_default_generation_strategy_at_MBM_node(
            experiment=experiment
        )

        card = UtilityRankingPlot(metric_name="foo").compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
        )

        # The figure's bar data x-values should be in ascending order
        # (plotly horizontal bars: ascending y = bottom to top, so highest
        # utility is at the top).
        fig = assert_is_instance(card, PlotlyAnalysisCard).get_figure()
        bar_data = fig.data[0]
        x_values = list(bar_data.x)
        self.assertEqual(x_values, sorted(x_values))
