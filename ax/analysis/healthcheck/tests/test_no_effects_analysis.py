# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# pyre-strict

import pandas as pd
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.no_effects_analysis import TestOfNoEffectAnalysis
from ax.core.data import Data
from ax.exceptions.core import AxError, UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


class TestTestOfNoEffectAnalysis(TestCase):
    def setUp(self) -> None:
        self.experiment = get_branin_experiment(with_trial=True)
        self.moo_experiment = get_branin_experiment_with_multi_objective(
            with_trial=True
        )
        self.tone = TestOfNoEffectAnalysis()

    def test_effects_detected(self) -> None:
        # GIVEN an experiment with effects detected
        self.experiment.attach_data(
            data=Data(
                df=pd.DataFrame(
                    {
                        "arm_name": ["0_0", "0_1", "0_2"],
                        "metric_name": ["branin"] * 3,
                        "mean": [0.0, 1.0, 2.0],
                        "sem": [0.1] * 3,
                        "trial_index": [0] * 3,
                        "n": [1000] * 3,
                    }
                )
            )
        )
        # WHEN we compute the healthcheck
        card = self.tone.compute(experiment=self.experiment)[0]
        # THEN it is a PASS
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertEqual(card.name, "TestOfNoEffectAnalysis")
        self.assertEqual(card.title, "Ax Test of No Effect Success")
        self.assertEqual(
            "Effects are observed for all objective metrics.", card.subtitle
        )
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.DIAGNOSTIC)

    def test_no_effects_detected(self) -> None:
        # GIVEN an experiment with no effects detected
        self.experiment.attach_data(
            data=Data(
                df=pd.DataFrame(
                    {
                        "arm_name": ["0_0", "0_1", "0_2"],
                        "metric_name": ["branin"] * 3,
                        "mean": [1.0] * 3,  # same mean, so no effect detected
                        "sem": [0.1] * 3,
                        "trial_index": [0] * 3,
                        "n": [1000] * 3,
                    }
                )
            )
        )
        # WHEN we compute the healthcheck
        card = self.tone.compute(experiment=self.experiment)[0]
        # THEN it is a WARNING
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertEqual(card.name, "TestOfNoEffectAnalysis")
        self.assertEqual(card.title, "Ax Test of No Effect Warning")
        self.assertIn("no effects have been detected", card.subtitle)
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.category, AnalysisCardCategory.DIAGNOSTIC)

    def test_raises_error_with_no_experiment(self) -> None:
        # GIVEN no experiment is provided
        # WHEN we compute the healthcheck
        # THEN it raises a UserInputError
        with self.assertRaises(UserInputError):
            self.tone.compute(experiment=None)

    def test_raises_error_without_optimization_confi(self) -> None:
        # GIVEN we have an experiment with no opt conifg
        self.experiment._optimization_config = None
        # WHEN we compute the healthcheck
        # THEN it raises a UserInputError
        with self.assertRaises(UserInputError):
            self.tone.compute(experiment=self.experiment)

    def test_raises_error_without_data(self) -> None:
        # GIVEN we have an experiment with no data
        # WHEN we compute the healthcheck
        # THEN it raises an AxError
        with self.assertRaises(AxError):
            self.tone.compute(experiment=self.experiment)

    def test_multi_objective_partial_no_effects(self) -> None:
        # GIVEN we have a multi-objective experiment with one metric with no effects
        self.moo_experiment.attach_data(
            data=Data(
                df=pd.DataFrame(
                    {
                        "arm_name": ["status_quo", "0_0", "status_quo", "0_0"],
                        "metric_name": ["branin_a", "branin_a", "branin_b", "branin_b"],
                        "mean": [1.0, 1.0, 2.0, 4.0],
                        "sem": [0.1] * 4,
                        "trial_index": [0] * 4,
                        "n": [1000] * 4,
                    }
                )
            )
        )
        # WHEN we compute the healthcheck
        card = self.tone.compute(experiment=self.moo_experiment)[0]
        # THEN it warns about one of the metrics
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertIn("branin_a", card.subtitle)
        self.assertNotIn("branin_b", card.subtitle)
