# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.search_space_summary import SearchSpaceSummary
from ax.exceptions.core import UserInputError
from ax.preview.api.client import Client
from ax.preview.api.configs import (
    ChoiceParameterConfig,
    ExperimentConfig,
    ParameterScaling,
    ParameterType,
    RangeParameterConfig,
)
from ax.utils.common.testutils import TestCase


class TestSearchSpaceSummary(TestCase):
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            experiment_config=ExperimentConfig(
                name="test_experiment",
                parameters=[
                    RangeParameterConfig(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0.1, 1),
                        scaling=ParameterScaling.LOG,
                    ),
                    ChoiceParameterConfig(
                        name="x2",
                        parameter_type=ParameterType.INT,
                        values=[0, 1],
                    ),
                ],
            )
        )

        analysis = SearchSpaceSummary()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        card = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "SearchSpaceSummary")
        self.assertEqual(card.title, "SearchSpaceSummary for `test_experiment`")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Parameter`-s in this `Experiment`",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "dataframe")

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
