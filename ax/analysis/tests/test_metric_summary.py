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
from ax.analysis.metric_summary import MetricSummary
from ax.api.client import Client
from ax.core.metric import Metric
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments


class TestMetricSummary(TestCase):
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            name="test_experiment",
            parameters=[],
        )
        client.configure_optimization(
            objective="foo, bar", outcome_constraints=["baz <= 0.0", "foo >= 1.0"]
        )
        # TODO: Debug error raised by
        # client.configure_metrics(metrics=[IMetric(name="qux")])

        client._experiment._tracking_metrics = {"qux": Metric(name="qux")}

        analysis = MetricSummary()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        (card,) = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "MetricSummary")
        self.assertEqual(card.title, "MetricSummary for `test_experiment`")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Metric`-s in this `Experiment`",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(card.category, AnalysisCardCategory.INFO)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.DATAFRAME)

        # Test dataframe for accuracy
        self.assertEqual(
            {*card.df.columns},
            {
                "Goal",
                "Name",
                "Type",
                "Lower is Better",
                "Bound",
            },
        )
        expected = pd.DataFrame(
            {
                "Name": ["bar", "foo", "baz", "qux"],
                "Type": ["MapMetric", "MapMetric", "MapMetric", "Metric"],
                "Goal": pd.Series(
                    ["maximize", "maximize", "constrain", "track"],
                    dtype=pd.CategoricalDtype(
                        categories=[
                            "minimize",
                            "maximize",
                            "constrain",
                            "track",
                            "None",
                        ],
                        ordered=True,
                    ),
                ),
                "Bound": ["None", ">= 1.0", "<= 0.0", "None"],
                "Lower is Better": [False, False, "None", "None"],
            }
        )
        pd.testing.assert_frame_equal(card.df, expected)

    def test_online(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        analysis = MetricSummary()
        for experiment in get_online_experiments():
            _ = analysis.compute(experiment=experiment)

    def test_offline(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        analysis = MetricSummary()
        for experiment in get_offline_experiments():
            _ = analysis.compute(experiment=experiment)
