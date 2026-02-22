# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.analysis.healthcheck.transfer_learning_analysis import TransferLearningAnalysis
from ax.core.auxiliary import TransferLearningMetadata
from ax.core.experiment import Experiment
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


def _make_experiment(
    param_names: list[str],
    experiment_type: str | None = None,
) -> Experiment:
    """Create a simple experiment with the given parameter names."""
    return Experiment(
        search_space=SearchSpace(
            parameters=[
                RangeParameter(
                    name=name,
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                )
                for name in param_names
            ]
        ),
        name="test_experiment",
        experiment_type=experiment_type,
    )


_MOCK_TARGET = "ax.storage.sqa_store.load.identify_transferable_experiments"


class TestTransferLearningAnalysis(TestCase):
    def test_no_experiment_type_returns_pass(self) -> None:
        """When no experiment_type is set and no experiment_types provided,
        return PASS."""
        experiment = _make_experiment(["x1", "x2"], experiment_type=None)
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertTrue(card.is_passing())
        self.assertIn("No experiment type set", card.subtitle)

    @patch(_MOCK_TARGET, return_value={})
    def test_no_candidates_returns_pass(self, mock_identify: object) -> None:
        experiment = _make_experiment(["x1", "x2"], experiment_type="my_type")
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
        self.assertTrue(card.is_passing())
        self.assertTrue(card.df.empty)

    @patch(_MOCK_TARGET)
    def test_single_candidate_returns_warning(self, mock_identify: object) -> None:
        experiment = _make_experiment(
            ["x1", "x2", "x3", "x4", "x5"], experiment_type="my_type"
        )
        mock_identify.return_value = {  # pyre-ignore[16]
            "source_exp": TransferLearningMetadata(
                overlap_parameters=["x1", "x2", "x3", "x4"],
            ),
        }
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertFalse(card.is_passing())
        self.assertIn("source_exp", card.subtitle)
        self.assertIn("80.0%", card.subtitle)
        self.assertEqual(len(card.df), 1)
        self.assertEqual(card.df.iloc[0]["Experiment"], "source_exp")
        self.assertEqual(card.df.iloc[0]["Overlapping Parameters"], 4)
        self.assertEqual(card.df.iloc[0]["Overlap (%)"], 80.0)

    @patch(_MOCK_TARGET)
    def test_multiple_candidates_sorted_by_count(self, mock_identify: object) -> None:
        experiment = _make_experiment(
            ["x1", "x2", "x3", "x4", "x5"], experiment_type="my_type"
        )
        mock_identify.return_value = {  # pyre-ignore[16]
            "exp_low": TransferLearningMetadata(
                overlap_parameters=["x1"],
            ),
            "exp_high": TransferLearningMetadata(
                overlap_parameters=["x1", "x2", "x3", "x4"],
            ),
            "exp_mid": TransferLearningMetadata(
                overlap_parameters=["x1", "x2", "x3"],
            ),
        }
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)

        # Verify sorted descending by overlap count
        self.assertEqual(card.df.iloc[0]["Experiment"], "exp_high")
        self.assertEqual(card.df.iloc[0]["Overlapping Parameters"], 4)
        self.assertEqual(card.df.iloc[1]["Experiment"], "exp_mid")
        self.assertEqual(card.df.iloc[1]["Overlapping Parameters"], 3)
        self.assertEqual(card.df.iloc[2]["Experiment"], "exp_low")
        self.assertEqual(card.df.iloc[2]["Overlapping Parameters"], 1)

        # All experiments listed in subtitle
        self.assertIn("exp_high", card.subtitle)
        self.assertIn("exp_mid", card.subtitle)
        self.assertIn("exp_low", card.subtitle)
        self.assertIn("We found **3 eligible source experiment(s)**", card.subtitle)

    @patch(_MOCK_TARGET)
    def test_percentage_calculation(self, mock_identify: object) -> None:
        experiment = _make_experiment(["x1", "x2", "x3"], experiment_type="my_type")
        mock_identify.return_value = {  # pyre-ignore[16]
            "exp_a": TransferLearningMetadata(
                overlap_parameters=["x1"],
            ),
        }
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.df.iloc[0]["Overlap (%)"], 33.3)

    @patch(_MOCK_TARGET)
    def test_parameters_listed_alphabetically(self, mock_identify: object) -> None:
        experiment = _make_experiment(
            ["alpha", "beta", "gamma", "delta"], experiment_type="my_type"
        )
        mock_identify.return_value = {  # pyre-ignore[16]
            "exp_a": TransferLearningMetadata(
                overlap_parameters=["gamma", "alpha", "delta"],
            ),
        }
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.df.iloc[0]["Parameters"], "alpha, delta, gamma")

    def test_requires_experiment(self) -> None:
        analysis = TransferLearningAnalysis()
        with self.assertRaises(UserInputError):
            analysis.compute(experiment=None)

    @patch(_MOCK_TARGET)
    def test_target_experiment_filtered_out(self, mock_identify: object) -> None:
        """The target experiment should be excluded from the results."""
        experiment = _make_experiment(["x1", "x2", "x3"], experiment_type="my_type")
        mock_identify.return_value = {  # pyre-ignore[16]
            "test_experiment": TransferLearningMetadata(
                overlap_parameters=["x1", "x2", "x3"],
            ),
            "other_exp": TransferLearningMetadata(
                overlap_parameters=["x1"],
            ),
        }
        analysis = TransferLearningAnalysis()
        card = analysis.compute(experiment=experiment)
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)
        self.assertEqual(len(card.df), 1)
        self.assertEqual(card.df.iloc[0]["Experiment"], "other_exp")
