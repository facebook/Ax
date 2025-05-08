# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import (
    AnalysisBlobAnnotation,
    AnalysisCardCategory,
    AnalysisCardLevel,
)
from ax.analysis.plotly.parallel_coordinates import (
    _get_parameter_dimension,
    ParallelCoordinatesPlot,
)
from ax.analysis.plotly.utils import select_metric
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_multi_objective,
    get_experiment_with_scalarized_objective_and_outcome_constraint,
    get_offline_experiments,
    get_online_experiments,
)
from pyre_extensions import none_throws


class TestParallelCoordinatesPlot(TestCase):
    def test_compute(self) -> None:
        analysis = ParallelCoordinatesPlot("branin")
        experiment = get_branin_experiment(with_completed_trial=True)

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        (card,) = analysis.compute(experiment=experiment)
        self.assertEqual(card.name, "ParallelCoordinatesPlot")
        self.assertEqual(card.title, "Parallel Coordinates for branin")
        self.assertEqual(
            card.subtitle,
            (
                "The parallel coordinates plot displays multi-dimensional "
                "data by representing each parameter as a parallel axis. This "
                "plot helps in assessing how thoroughly the search space has "
                "been explored and in identifying patterns or clusterings associated "
                "with high-performing (good) or low-performing (bad) arms. By "
                "tracing lines across the axes, one can observe correlations and "
                "interactions between parameters, gaining insights into the "
                "relationships that contribute to the success or failure of "
                "different configurations within the experiment."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.HIGH)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns}, {"trial_index", "arm_name", "branin", "x1", "x2"}
        )
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.PLOTLY)

        analysis_no_metric = ParallelCoordinatesPlot()
        _ = analysis_no_metric.compute(experiment=experiment)

    def test_select_metric(self) -> None:
        experiment = get_branin_experiment()
        experiment_no_optimization_config = get_branin_experiment(
            has_optimization_config=False
        )
        experiment_multi_objective = get_experiment_with_multi_objective()
        experiment_scalarized_objective = (
            get_experiment_with_scalarized_objective_and_outcome_constraint()
        )

        self.assertEqual(select_metric(experiment=experiment), "branin")

        with self.assertRaisesRegex(ValueError, "OptimizationConfig"):
            select_metric(experiment=experiment_no_optimization_config)

        with self.assertRaisesRegex(UnsupportedError, "MultiObjective"):
            select_metric(experiment=experiment_multi_objective)

        with self.assertRaisesRegex(UnsupportedError, "ScalarizedObjective"):
            select_metric(experiment=experiment_scalarized_objective)

    def test_get_parameter_dimension(self) -> None:
        range_series = pd.Series([0, 1, 2, 3], name="range")
        range_dimension = _get_parameter_dimension(series=range_series)
        self.assertEqual(
            range_dimension,
            {
                "tickvals": None,
                "ticktext": None,
                "label": "range",
                "values": range_series.tolist(),
            },
        )

        choice_series = pd.Series(["foo", "bar", "baz"], name="choice")
        choice_dimension = _get_parameter_dimension(series=choice_series)
        self.assertEqual(
            choice_dimension,
            {
                "tickvals": ["0", "1", "2"],
                "ticktext": ["bar", "baz", "foo"],
                "label": "choice",
                "values": [2, 0, 1],
            },
        )

    def test_online(self) -> None:
        # Test ParallelCoordinatesPlot can be computed for a variety of experiments
        # which resemble those we see in an online setting.

        for experiment in get_online_experiments():
            analysis = ParallelCoordinatesPlot(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0]
            )

            _ = analysis.compute(experiment=experiment)

    def test_offline(self) -> None:
        # Test ParallelCoordinatesPlot can be computed for a variety of experiments
        # which resemble those we see in an offline setting.

        for experiment in get_offline_experiments():
            analysis = ParallelCoordinatesPlot(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0]
            )

            _ = analysis.compute(experiment=experiment)
