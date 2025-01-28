# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.summary import Summary
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.preview.api.client import Client
from ax.preview.api.configs import ExperimentConfig, ParameterType, RangeParameterConfig
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance, none_throws


class TestSummary(TestCase):
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            experiment_config=ExperimentConfig(
                name="test_experiment",
                parameters=[
                    RangeParameterConfig(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                    RangeParameterConfig(
                        name="x2",
                        parameter_type=ParameterType.FLOAT,
                        bounds=(0, 1),
                    ),
                ],
            )
        )
        client.configure_optimization(objective="foo, bar")

        # Get two trials and fail one, giving us a ragged structure
        client.get_next_trials(maximum_trials=2)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 2.0})
        client.mark_trial_failed(trial_index=1)

        analysis = Summary()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        card = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "Summary")
        self.assertEqual(card.title, "Summary for test_experiment")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Trial`-s in this `Experiment`",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "dataframe")

        # Test dataframe for accuracy
        self.assertEqual(
            {*card.df.columns},
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_method",
                "generation_node",
                "foo",
                "bar",
                "x1",
                "x2",
            },
        )

        trial_0_parameters = none_throws(
            assert_is_instance(experiment.trials[0], Trial).arm
        ).parameters
        trial_1_parameters = none_throws(
            assert_is_instance(experiment.trials[1], Trial).arm
        ).parameters
        expected = pd.DataFrame(
            {
                "trial_index": {0: 0, 1: 1},
                "arm_name": {0: "0_0", 1: "1_0"},
                "trial_status": {0: "COMPLETED", 1: "FAILED"},
                "generation_method": {0: "Sobol", 1: "Sobol"},
                "generation_node": {0: "Sobol", 1: "Sobol"},
                "foo": {0: 1.0, 1: np.nan},  # NaN because trial 1 failed
                "bar": {0: 2.0, 1: np.nan},
                "x1": {
                    0: trial_0_parameters["x1"],
                    1: trial_1_parameters["x1"],
                },
                "x2": {
                    0: trial_0_parameters["x2"],
                    1: trial_1_parameters["x2"],
                },
            }
        )
        self.assertTrue(card.df.equals(expected))

        # Test without omitting empty columns
        analysis_no_omit = Summary(omit_empty_columns=False)
        card_no_omit = analysis_no_omit.compute(experiment=experiment)
        self.assertEqual(
            {*card_no_omit.df.columns},
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "fail_reason",
                "generation_method",
                "generation_node",
                "foo",
                "bar",
                "x1",
                "x2",
            },
        )
        self.assertEqual(len(card_no_omit.df), len(experiment.arms_by_name))
