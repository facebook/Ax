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
from ax.analysis.search_space_summary import SearchSpaceSummary
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments


class TestSearchSpaceSummary(TestCase):
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0.1, 1),
                    scaling="log",
                ),
                ChoiceParameterConfig(
                    name="x2",
                    parameter_type="int",
                    values=[0, 1],
                ),
            ],
        )

        analysis = SearchSpaceSummary()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        (card,) = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "SearchSpaceSummary")
        self.assertEqual(card.title, "SearchSpaceSummary for `test_experiment`")
        self.assertEqual(
            card.subtitle,
            (
                "The search space summary provides an overview of all "
                "parameters, including their names, types, and ranges or "
                "categories. This holistic view provides quick understanding "
                "on the parameters being optimized, allowing one to verify "
                "and adjust the search space configuration for effective "
                "exploration."
            ),
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(card.category, AnalysisCardCategory.INFO)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.DATAFRAME)

        # Test dataframe for accuracy
        self.assertEqual(
            {*card.df.columns},
            {"Name", "Type", "Flags", "Datatype", "Domain"},
        )
        expected = pd.DataFrame(
            {
                "Name": ["x1", "x2"],
                "Type": ["Range", "Choice"],
                "Domain": ["range=[0.1, 1.0]", "values=[0, 1]"],
                "Datatype": ["float", "int"],
                "Flags": ["log_scale", "ordered, sorted"],
            }
        )
        pd.testing.assert_frame_equal(card.df, expected)

    def test_online(self) -> None:
        # Test SearchSpaceSummary can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        analysis = SearchSpaceSummary()
        for experiment in get_online_experiments():
            _ = analysis.compute(experiment=experiment)

    def test_offline(self) -> None:
        # Test SearchSpaceSummary can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        analysis = SearchSpaceSummary()
        for experiment in get_offline_experiments():
            _ = analysis.compute(experiment=experiment)
