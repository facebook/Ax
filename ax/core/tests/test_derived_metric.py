#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.derived_metric import DerivedMetric
from ax.core.experiment import Experiment
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.result import Err, Ok
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space
from pyre_extensions import none_throws


def _make_trial_data(
    trial_index: int,
    arm_metrics: dict[str, dict[str, float]],
) -> Data:
    """Build Data with explicit per-arm, per-metric values.

    Args:
        trial_index: Trial index for the data rows.
        arm_metrics: ``{arm_name: {metric_name: mean_value}}``.
    """
    rows: list[dict[str, Any]] = []
    for arm_name, metrics in arm_metrics.items():
        for metric_name, mean in metrics.items():
            rows.append(
                {
                    "trial_index": trial_index,
                    "arm_name": arm_name,
                    "metric_name": metric_name,
                    "metric_signature": metric_name,
                    "mean": mean,
                    "sem": 0.1,
                }
            )
    return Data(df=pd.DataFrame(rows))


class _SumDerivedMetric(DerivedMetric):
    """Trivial concrete subclass for testing: sums its input metrics."""

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            cached_data = trial.lookup_data()
        except Exception as e:
            return Err(MetricFetchE(message=f"Lookup failed: {e}", exception=e))

        if cached_data.empty:
            return Err(
                MetricFetchE(
                    message=f"No cached data for trial {trial.index}.",
                    exception=None,
                )
            )

        df = cached_data.df
        result_rows: list[dict[str, Any]] = []

        for arm_name in df["arm_name"].unique():
            arm_df = df[df["arm_name"] == arm_name]
            total = 0.0
            for metric_name in self.input_metric_names:
                rows = self._lookup_metric_values_for_arm(arm_df, metric_name)
                if rows.empty:
                    return Err(
                        MetricFetchE(
                            message=(
                                f"Missing '{metric_name}' for arm "
                                f"'{arm_name}' in trial {trial.index}."
                            ),
                            exception=None,
                        )
                    )
                total += float(rows["mean"].iloc[-1])

            result_rows.append(
                {
                    "trial_index": trial.index,
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "metric_signature": self.signature,
                    "mean": total,
                    "sem": float("nan"),
                }
            )

        return Ok(value=Data(df=pd.DataFrame(result_rows)))


class DerivedMetricTest(TestCase):
    """Tests for the DerivedMetric base class."""

    def test_init_and_properties(self) -> None:
        """Construction, attributes, and clone round-trip."""
        metric = _SumDerivedMetric(
            name="total",
            input_metric_names=["a", "b"],
            lower_is_better=True,
            properties={"key": "value"},
        )
        self.assertIsInstance(metric, DerivedMetric)
        self.assertIsInstance(metric, Metric)
        self.assertEqual(metric.input_metric_names, ["a", "b"])
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(metric.properties, {"key": "value"})

        # summary_dict includes input_metric_names
        self.assertIn("input_metric_names", metric.summary_dict)
        self.assertEqual(metric.summary_dict["input_metric_names"], ["a", "b"])

    def test_empty_input_metric_names_raises(self) -> None:
        with self.assertRaises(UserInputError):
            _SumDerivedMetric(name="bad", input_metric_names=[])

    def test_clone(self) -> None:
        metric = _SumDerivedMetric(
            name="total",
            input_metric_names=["a", "b"],
            lower_is_better=False,
            properties={"p": 1},
        )
        cloned = metric.clone()
        self.assertIsInstance(cloned, _SumDerivedMetric)
        assert isinstance(cloned, _SumDerivedMetric)
        self.assertEqual(cloned.input_metric_names, metric.input_metric_names)
        self.assertEqual(cloned.lower_is_better, metric.lower_is_better)
        self.assertEqual(cloned.properties, metric.properties)


class DerivedMetricExperimentIntegrationTest(TestCase):
    """Two-phase fetching: base metrics first, then derived metrics."""

    def test_two_phase_fetch(self) -> None:
        """Experiment.fetch_data fetches base metrics, attaches them, then
        fetches derived metrics that read from the cache."""
        derived = _SumDerivedMetric(
            name="total",
            input_metric_names=["base_a", "base_b"],
        )
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="obj"), minimize=True),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=derived,
                        op=ComparisonOp.LEQ,
                        bound=100.0,
                        relative=False,
                    )
                ],
            ),
            tracking_metrics=[Metric(name="base_a"), Metric(name="base_b")],
        )
        self.assertIsInstance(experiment.metrics["total"], DerivedMetric)

        # Create 2 trials with known base metric values.
        for i in range(2):
            trial = experiment.new_trial()
            trial.add_arm(
                Arm(name=f"{i}_0", parameters={"x1": float(i), "x2": float(i)})
            )
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            experiment.attach_data(
                _make_trial_data(
                    i,
                    {
                        f"{i}_0": {
                            "obj": float(i + 1),
                            "base_a": float(i + 2),
                            "base_b": float(i + 3),
                        }
                    },
                )
            )

        data = experiment.fetch_data()
        metric_names = set(data.df["metric_name"].unique())
        self.assertIn("total", metric_names)

        # Verify derived values: total = base_a + base_b
        derived_df = data.df[data.df["metric_name"] == "total"]
        for i in range(2):
            row = derived_df[derived_df["trial_index"] == i]
            expected = (i + 2) + (i + 3)
            self.assertAlmostEqual(row["mean"].iloc[0], expected, places=10)

    def test_fetch_without_derived_metrics(self) -> None:
        """When no derived metrics exist, fetch_data works as before."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            tracking_metrics=[Metric(name="m1")],
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()
        experiment.attach_data(_make_trial_data(0, {"0_0": {"m1": 42.0}}))

        data = experiment.fetch_data()
        self.assertEqual(len(data.df), 1)
        self.assertEqual(data.df["mean"].iloc[0], 42.0)

    def test_derived_metric_missing_input_returns_err(self) -> None:
        """Derived metric returns Err when an input metric is missing."""
        derived = _SumDerivedMetric(
            name="total",
            input_metric_names=["a", "b"],
        )
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            tracking_metrics=[Metric(name="a"), derived],
        )
        trial = experiment.new_trial()
        trial.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()
        # Only attach "a", not "b"
        experiment.attach_data(_make_trial_data(0, {"0_0": {"a": 1.0}}))

        results = experiment.fetch_trials_data_results(trial_indices=[0])
        result = results[0]["total"]
        self.assertIsInstance(result, Err)
        self.assertIn("b", none_throws(result.err).message)
