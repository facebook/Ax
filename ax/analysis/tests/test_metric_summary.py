# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.metric_summary import MetricSummary
from ax.api.client import Client
from ax.core.metric import Metric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from pyre_extensions import none_throws


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

        client._experiment._metrics["qux"] = Metric(name="qux")

        analysis = MetricSummary()

        experiment = client._experiment
        card = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "MetricSummary")
        self.assertEqual(card.title, "MetricSummary for `test_experiment`")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Metric`-s in this `Experiment`",
        )
        self.assertIsNotNone(card.blob)

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

    def test_validate_applicable_state(self) -> None:
        self.assertIn(
            "Requires an Experiment",
            none_throws(MetricSummary().validate_applicable_state()),
        )

    def test_online_and_offline(self) -> None:
        # Verify MetricSummary computes for both online and offline experiment types.
        analysis = MetricSummary()
        for label, get_experiments in [
            ("online", get_online_experiments),
            ("offline", get_offline_experiments),
        ]:
            with self.subTest(setting=label):
                for experiment in get_experiments():
                    _ = analysis.compute(experiment=experiment)
