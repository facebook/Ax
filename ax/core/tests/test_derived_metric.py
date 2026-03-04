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
from ax.core.metric import Metric, MetricFetchResult
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
    """Trivial concrete subclass for testing: sums its input metrics.

    Optionally accepts ``source_trial_map`` to simulate cross-trial data
    lookup (used by LILOPairwiseMetric-like scenarios).
    """

    def __init__(
        self,
        name: str,
        input_metric_names: list[str],
        source_trial_map: dict[str, tuple[int, str]] | None = None,
        relativize_inputs: bool = False,
        as_percent: bool = True,
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_metric_names=input_metric_names,
            relativize_inputs=relativize_inputs,
            as_percent=as_percent,
            lower_is_better=lower_is_better,
            properties=properties,
        )
        self._source_trial_map = source_trial_map

    @property
    def source_trial_map(self) -> dict[str, tuple[int, str]] | None:
        return self._source_trial_map

    def _resolve_source_trial_indices(
        self,
        trial: BaseTrial,
    ) -> dict[str, tuple[int, str]] | None:
        return self._source_trial_map

    def _compute_derived_values(
        self,
        trial: BaseTrial,
        arm_data: dict[str, pd.DataFrame],
    ) -> MetricFetchResult:
        result_rows: list[dict[str, Any]] = []
        for arm_name, arm_df in arm_data.items():
            total = 0.0
            for m in self.input_metric_names:
                rows = self._lookup_metric_values_for_arm(arm_df, m)
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


def _make_experiment_with_trial(
    arm_metrics: dict[str, dict[str, float]],
    batch: bool = False,
) -> Experiment:
    """Create an experiment with one completed trial and attached data.

    Args:
        arm_metrics: ``{arm_name: {metric_name: mean_value}}``.
        batch: If True, create a BatchTrial; otherwise a Trial.
    """
    exp = Experiment(name="test", search_space=get_branin_search_space())
    trial = exp.new_batch_trial() if batch else exp.new_trial()
    for i, arm_name in enumerate(arm_metrics):
        trial.add_arm(
            Arm(
                name=arm_name,
                parameters={"x1": float(i), "x2": float(i)},
            )
        )
    trial.mark_running(no_runner_required=True)
    trial.mark_completed()
    exp.attach_data(_make_trial_data(0, arm_metrics))
    return exp


class DerivedMetricTest(TestCase):
    """Tests for the DerivedMetric base class."""

    def test_init_validation_and_clone(self) -> None:
        """Construction, attributes, summary_dict, empty-input rejection,
        and clone round-trip."""
        metric = _SumDerivedMetric(
            name="total",
            input_metric_names=["a", "b"],
            relativize_inputs=True,
            lower_is_better=True,
            properties={"key": "value"},
        )
        self.assertIsInstance(metric, DerivedMetric)
        self.assertIsInstance(metric, Metric)
        self.assertEqual(metric.input_metric_names, ["a", "b"])
        self.assertTrue(metric.relativize_inputs)
        self.assertTrue(metric.lower_is_better)
        self.assertEqual(metric.properties, {"key": "value"})
        self.assertIn("input_metric_names", metric.summary_dict)
        self.assertIn("relativize_inputs", metric.summary_dict)
        self.assertTrue(metric.summary_dict["relativize_inputs"])
        self.assertIn("as_percent", metric.summary_dict)
        self.assertTrue(metric.summary_dict["as_percent"])

        with self.subTest("default_relativize_inputs"):
            m2 = _SumDerivedMetric(name="m2", input_metric_names=["a"])
            self.assertFalse(m2.relativize_inputs)
            self.assertFalse(m2.summary_dict["relativize_inputs"])

        with self.subTest("empty_input_metric_names"):
            with self.assertRaises(UserInputError):
                _SumDerivedMetric(name="bad", input_metric_names=[])

        with self.subTest("clone"):
            cloned = metric.clone()
            assert isinstance(cloned, _SumDerivedMetric)
            self.assertEqual(cloned.input_metric_names, metric.input_metric_names)
            self.assertEqual(cloned.relativize_inputs, metric.relativize_inputs)
            self.assertEqual(cloned.as_percent, metric.as_percent)
            self.assertEqual(cloned.lower_is_better, metric.lower_is_better)
            self.assertEqual(cloned.properties, metric.properties)

    def test_fetch_trial_data(self) -> None:
        """Happy path: correct derived value from same-trial data."""
        metric = _SumDerivedMetric(name="total", input_metric_names=["a", "b"])
        exp = _make_experiment_with_trial({"0_0": {"a": 3.0, "b": 4.0}})

        result = metric.fetch_trial_data(exp.trials[0])
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["mean"].iloc[0], 7.0)

    def test_fetch_trial_data_errors(self) -> None:
        """Err results for various missing-data scenarios."""
        with self.subTest("empty_data"):
            metric = _SumDerivedMetric(name="total", input_metric_names=["a"])
            exp = Experiment(name="test", search_space=get_branin_search_space())
            trial = exp.new_trial()
            trial.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            # No data attached.
            result = metric.fetch_trial_data(trial)
            self.assertIsInstance(result, Err)
            self.assertIn("no cached data", none_throws(result.err).message.lower())

        with self.subTest("missing_global_metric"):
            metric = _SumDerivedMetric(name="total", input_metric_names=["a", "b"])
            exp = _make_experiment_with_trial({"0_0": {"a": 1.0}})
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            self.assertIn("b", none_throws(result.err).message)

        with self.subTest("per_arm_missing_metric"):
            metric = _SumDerivedMetric(name="total", input_metric_names=["a", "b"])
            exp = _make_experiment_with_trial(
                {"arm1": {"a": 1.0, "b": 2.0}, "arm2": {"a": 3.0}},
                batch=True,
            )
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            msg = none_throws(result.err).message
            self.assertIn("arm2", msg)
            self.assertIn("b", msg)

    def test_relativize_arm_data(self) -> None:
        """Relativization of input metrics w.r.t. status quo arm."""
        metric = _SumDerivedMetric(
            name="total",
            input_metric_names=["a", "b"],
            relativize_inputs=True,
        )

        # Build experiment with SQ arm + one treatment arm.
        exp = Experiment(name="test", search_space=get_branin_search_space())
        sq_arm = Arm(name="sq", parameters={"x1": 0.0, "x2": 0.0})
        exp.status_quo = sq_arm
        trial = exp.new_batch_trial()
        trial.add_arm(sq_arm)
        trial.add_arm(Arm(name="arm1", parameters={"x1": 1.0, "x2": 1.0}))
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()
        # SQ: a=10, b=20.  arm1: a=15, b=30.
        exp.attach_data(
            _make_trial_data(
                0,
                {
                    "sq": {"a": 10.0, "b": 20.0},
                    "arm1": {"a": 15.0, "b": 30.0},
                },
            )
        )

        result = metric.fetch_trial_data(exp.trials[0])
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df
        # SQ arm should be excluded from output.
        self.assertEqual(set(df["arm_name"].unique()), {"arm1"})
        # arm1 relativized (as_percent=True):
        # a=(15-10)/10=50%, b=(30-20)/20=50%; sum=100.0
        self.assertAlmostEqual(df["mean"].iloc[0], 100.0)

        with self.subTest("no_status_quo"):
            exp_no_sq = Experiment(name="no_sq", search_space=get_branin_search_space())
            trial2 = exp_no_sq.new_trial()
            trial2.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
            trial2.mark_running(no_runner_required=True)
            trial2.mark_completed()
            exp_no_sq.attach_data(_make_trial_data(0, {"0_0": {"a": 1.0, "b": 2.0}}))
            result = metric.fetch_trial_data(trial2)
            self.assertIsInstance(result, Err)
            self.assertIn("no status quo", none_throws(result.err).message.lower())

        with self.subTest("sq_not_in_trial_data"):
            # SQ arm set on experiment but not present in trial.
            exp3 = Experiment(name="sq_missing", search_space=get_branin_search_space())
            exp3.status_quo = Arm(name="sq_arm", parameters={"x1": 0.0, "x2": 0.0})
            trial3 = exp3.new_trial()
            trial3.add_arm(Arm(name="other", parameters={"x1": 1.0, "x2": 1.0}))
            trial3.mark_running(no_runner_required=True)
            trial3.mark_completed()
            exp3.attach_data(_make_trial_data(0, {"other": {"a": 1.0, "b": 2.0}}))
            result = metric.fetch_trial_data(trial3)
            self.assertIsInstance(result, Err)
            self.assertIn("sq_arm", none_throws(result.err).message)

        with self.subTest("sq_near_zero"):
            exp4 = Experiment(name="sq_zero", search_space=get_branin_search_space())
            sq4 = Arm(name="sq", parameters={"x1": 0.0, "x2": 0.0})
            exp4.status_quo = sq4
            trial4 = exp4.new_batch_trial()
            trial4.add_arm(sq4)
            trial4.add_arm(Arm(name="arm1", parameters={"x1": 1.0, "x2": 1.0}))
            trial4.mark_running(no_runner_required=True)
            trial4.mark_completed()
            # SQ has a=0 (too close to zero for relativization).
            exp4.attach_data(
                _make_trial_data(
                    0,
                    {
                        "sq": {"a": 0.0, "b": 5.0},
                        "arm1": {"a": 1.0, "b": 10.0},
                    },
                )
            )
            result = metric.fetch_trial_data(trial4)
            self.assertIsInstance(result, Err)
            self.assertIn(
                "too small to reliably", none_throws(result.err).message.lower()
            )

        with self.subTest("relativize_false_passthrough"):
            # When relativize_inputs=False (default), SQ is not excluded
            # and values are raw.
            metric_no_rel = _SumDerivedMetric(
                name="total", input_metric_names=["a", "b"]
            )
            result = metric_no_rel.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Ok)
            df_no_rel = none_throws(result.ok).df
            # Both arms should be present.
            self.assertEqual(set(df_no_rel["arm_name"].unique()), {"sq", "arm1"})
            sq_row = df_no_rel[df_no_rel["arm_name"] == "sq"]
            # SQ raw sum: 10 + 20 = 30
            self.assertAlmostEqual(sq_row["mean"].iloc[0], 30.0)
            arm1_row = df_no_rel[df_no_rel["arm_name"] == "arm1"]
            # arm1 raw sum: 15 + 30 = 45
            self.assertAlmostEqual(arm1_row["mean"].iloc[0], 45.0)

        with self.subTest("sq_in_different_trial"):
            # SQ data lives in a baseline trial, treatment arms are in
            # separate trials.  The SQ fallback should find it.
            exp5 = Experiment(name="sq_sep", search_space=get_branin_search_space())
            sq5 = Arm(name="sq", parameters={"x1": 0.0, "x2": 0.0})
            exp5.status_quo = sq5
            # Trial 0: SQ baseline only.
            t0 = exp5.new_batch_trial()
            t0.add_arm(sq5)
            t0.mark_running(no_runner_required=True)
            t0.mark_completed()
            exp5.attach_data(_make_trial_data(0, {"sq": {"a": 10.0, "b": 20.0}}))
            # Trial 1: treatment arm only (no SQ data in this trial).
            t1 = exp5.new_batch_trial()
            t1.add_arm(Arm(name="arm1", parameters={"x1": 1.0, "x2": 1.0}))
            t1.mark_running(no_runner_required=True)
            t1.mark_completed()
            exp5.attach_data(_make_trial_data(1, {"arm1": {"a": 15.0, "b": 30.0}}))
            result = metric.fetch_trial_data(t1)
            self.assertIsInstance(result, Ok)
            df5 = none_throws(result.ok).df
            self.assertEqual(set(df5["arm_name"].unique()), {"arm1"})
            # arm1: a=(15-10)/10*100=50%, b=(30-20)/20*100=50%; sum=100.0
            self.assertAlmostEqual(df5["mean"].iloc[0], 100.0)

    def test_cross_trial_relativization(self) -> None:
        """Cross-trial relativization handles non-stationarity correctly.

        When arms come from different trials, each arm must be relativized
        against the SQ from its own source trial, not a single global SQ.
        """
        exp = Experiment(name="test", search_space=get_branin_search_space())
        sq_arm = Arm(name="sq", parameters={"x1": 0.0, "x2": 0.0})
        exp.status_quo = sq_arm

        # Trial 0: SQ a=10, arm_a a=15.  (arm_a: 50% improvement)
        t0 = exp.new_batch_trial()
        t0.add_arm(sq_arm)
        t0.add_arm(Arm(name="arm_a", parameters={"x1": 1.0, "x2": 1.0}))
        t0.mark_running(no_runner_required=True)
        t0.mark_completed()
        exp.attach_data(_make_trial_data(0, {"sq": {"a": 10.0}, "arm_a": {"a": 15.0}}))

        # Trial 1: SQ a=20 (non-stationarity!), arm_b a=30.
        # arm_b: (30-20)/20 = 50% improvement (same %).
        t1 = exp.new_batch_trial()
        t1.add_arm(sq_arm)
        t1.add_arm(Arm(name="arm_b", parameters={"x1": 2.0, "x2": 2.0}))
        t1.mark_running(no_runner_required=True)
        t1.mark_completed()
        exp.attach_data(_make_trial_data(1, {"sq": {"a": 20.0}, "arm_b": {"a": 30.0}}))

        # Trial 2 (LILO-like): arm_a + arm_b, no own data.
        t2 = exp.new_batch_trial()
        t2.add_arm(Arm(name="arm_a", parameters={"x1": 1.0, "x2": 1.0}))
        t2.add_arm(Arm(name="arm_b", parameters={"x1": 2.0, "x2": 2.0}))
        t2.mark_running(no_runner_required=True)
        t2.mark_completed()

        # source_trial_map: arm_a from trial 0, arm_b from trial 1.
        metric = _SumDerivedMetric(
            name="total",
            input_metric_names=["a"],
            relativize_inputs=True,
            source_trial_map={
                "arm_a": (0, "arm_a"),
                "arm_b": (1, "arm_b"),
            },
        )
        result = metric.fetch_trial_data(t2)
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df

        # Both arms should be relativized against their own trial's SQ.
        self.assertEqual(set(df["arm_name"].unique()), {"arm_a", "arm_b"})
        arm_a_val = df[df["arm_name"] == "arm_a"]["mean"].iloc[0]
        arm_b_val = df[df["arm_name"] == "arm_b"]["mean"].iloc[0]
        # arm_a: (15-10)/10 * 100 = 50%
        self.assertAlmostEqual(arm_a_val, 50.0)
        # arm_b: (30-20)/20 * 100 = 50%
        self.assertAlmostEqual(arm_b_val, 50.0)

        with self.subTest("non_stationary_sq_difference"):
            # If we had incorrectly used a single SQ value, arm_b would
            # show (30-10)/10=200% (wrong) instead of 50% (correct).
            # Verify by making SQ values clearly different.
            metric2 = _SumDerivedMetric(
                name="total",
                input_metric_names=["a"],
                relativize_inputs=True,
                source_trial_map={
                    "arm_a": (0, "arm_a"),
                    "arm_b": (1, "arm_b"),
                },
            )
            result2 = metric2.fetch_trial_data(t2)
            self.assertIsInstance(result2, Ok)
            df2 = none_throws(result2.ok).df
            # Both should be 50%, proving per-trial SQ is used.
            for arm in ["arm_a", "arm_b"]:
                self.assertAlmostEqual(
                    df2[df2["arm_name"] == arm]["mean"].iloc[0], 50.0
                )

    def test_two_phase_experiment_fetch(self) -> None:
        """Experiment.fetch_data runs base metrics first, then derived metrics.

        Also verifies that missing input metrics propagate as Err through
        the experiment-level fetch path.
        """
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
        self.assertIn("total", set(data.df["metric_name"].unique()))

        derived_df = data.df[data.df["metric_name"] == "total"]
        for i in range(2):
            row = derived_df[derived_df["trial_index"] == i]
            expected = (i + 2) + (i + 3)
            self.assertAlmostEqual(row["mean"].iloc[0], expected, places=10)

        with self.subTest("missing_input_via_experiment"):
            incomplete = _SumDerivedMetric(name="inc", input_metric_names=["a", "b"])
            exp2 = Experiment(
                name="test2",
                search_space=get_branin_search_space(),
                tracking_metrics=[Metric(name="a"), incomplete],
            )
            trial = exp2.new_trial()
            trial.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            exp2.attach_data(_make_trial_data(0, {"0_0": {"a": 1.0}}))

            results = exp2.fetch_trials_data_results(trial_indices=[0])
            result = results[0]["inc"]
            self.assertIsInstance(result, Err)
            self.assertIn("b", none_throws(result.err).message)
