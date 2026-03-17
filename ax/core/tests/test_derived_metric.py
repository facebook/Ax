#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Any, cast

import pandas as pd
from ax.adapter.torch import TorchAdapter
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.derived_metric import DerivedMetric, ExpressionDerivedMetric
from ax.core.experiment import Experiment
from ax.core.metric import Metric, MetricFetchResult
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.result import Err, Ok
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space
from ax.utils.testing.mock import mock_botorch_optimize
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
        # SQ arm should be included with zero-valued inputs (sum=0).
        self.assertEqual(set(df["arm_name"].unique()), {"sq", "arm1"})
        sq_row = df[df["arm_name"] == "sq"]
        # SQ: inputs are zero after relativization, sum(0,0) = 0.
        self.assertAlmostEqual(sq_row["mean"].iloc[0], 0.0)
        arm1_row = df[df["arm_name"] == "arm1"]
        # arm1 relativized (as_percent=True):
        # a=(15-10)/10=50%, b=(30-20)/20=50%; sum=100.0
        self.assertAlmostEqual(arm1_row["mean"].iloc[0], 100.0)

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
            tracking_metrics=[Metric(name="base_a"), Metric(name="base_b"), derived],
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

    def test_fetch_trial_data_skips_abandoned_arms(self) -> None:
        """Abandoned arms are skipped in _collect_arm_data.

        When a BatchTrial has an abandoned arm, that arm has no base metric
        data (the metric system correctly does not fetch data for abandoned
        arms).  _collect_arm_data must skip abandoned arms rather than
        returning a MetricFetchE for missing input data.
        """
        metric = _SumDerivedMetric(name="total", input_metric_names=["a", "b"])
        exp = Experiment(name="test", search_space=get_branin_search_space())
        trial = exp.new_batch_trial()
        arm_ok = Arm(name="arm_ok", parameters={"x1": 1.0, "x2": 1.0})
        arm_abandoned = Arm(name="arm_abandoned", parameters={"x1": 2.0, "x2": 2.0})
        trial.add_arm(arm_ok)
        trial.add_arm(arm_abandoned)
        trial.mark_running(no_runner_required=True)
        trial.mark_arm_abandoned(arm_name="arm_abandoned")
        trial.mark_completed()

        # Only attach data for the non-abandoned arm.
        exp.attach_data(_make_trial_data(0, {"arm_ok": {"a": 3.0, "b": 4.0}}))

        result = metric.fetch_trial_data(trial)
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df
        self.assertEqual(list(df["arm_name"]), ["arm_ok"])
        self.assertAlmostEqual(df["mean"].iloc[0], 7.0)


class ExpressionDerivedMetricTest(TestCase):
    """Tests for ExpressionDerivedMetric."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _batch_experiment_with_sq(
        self,
        sq_values: dict[str, float],
        arm_values: dict[str, dict[str, float]],
    ) -> Experiment:
        """Experiment with a batch trial containing a status quo and treatment arms."""
        sq_arm = Arm(name="status_quo", parameters={"x1": 0.0, "x2": 0.0})
        exp = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            status_quo=sq_arm,
        )
        trial = exp.new_batch_trial()
        trial.add_arm(sq_arm)
        for i, arm_name in enumerate(arm_values, start=1):
            trial.add_arm(
                Arm(name=arm_name, parameters={"x1": float(i), "x2": float(i)})
            )
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()
        all_arm_metrics = {"status_quo": sq_values, **arm_values}
        exp.attach_data(_make_trial_data(0, all_arm_metrics))
        return exp

    # ------------------------------------------------------------------
    # Construction & validation
    # ------------------------------------------------------------------

    def test_init_and_clone(self) -> None:
        """Construction, inheritance, and clone round-trip."""
        metric = ExpressionDerivedMetric(
            name="ratio",
            input_metric_names=["a", "b"],
            expression_str="a / b",
            lower_is_better=True,
            properties={"key": "value"},
        )
        self.assertIsInstance(metric, DerivedMetric)
        self.assertIsInstance(metric, Metric)
        self.assertEqual(metric.input_metric_names, ["a", "b"])
        self.assertEqual(metric.expression_str, "a / b")
        self.assertTrue(metric.lower_is_better)
        self.assertFalse(metric.relativize_inputs)

        # Clone round-trip
        cloned = cast(ExpressionDerivedMetric, metric.clone())
        self.assertIsInstance(cloned, ExpressionDerivedMetric)
        self.assertEqual(cloned.input_metric_names, metric.input_metric_names)
        self.assertEqual(cloned.expression_str, metric.expression_str)
        self.assertEqual(cloned.lower_is_better, metric.lower_is_better)
        self.assertEqual(cloned.properties, metric.properties)
        self.assertFalse(cloned.relativize_inputs)

        # Clone preserves relativize_inputs=True
        rel_metric = ExpressionDerivedMetric(
            name="rel",
            input_metric_names=["a", "b"],
            expression_str="a / b",
            relativize_inputs=True,
        )
        rel_cloned = cast(ExpressionDerivedMetric, rel_metric.clone())
        self.assertTrue(rel_cloned.relativize_inputs)

    def test_validation_rejects_invalid_expressions(self) -> None:
        """Various invalid expressions must raise ``UserInputError``."""
        cases: list[tuple[str, list[str], str]] = [
            ("empty_inputs", [], "42"),
            ("undeclared_variable", ["a"], "a + b"),
            ("bad_syntax", ["a"], "a +"),
            ("list_comprehension", ["a"], "[x for x in a]"),
            ("comparison", ["a", "b"], "a > b"),
        ]
        for label, inputs, expr in cases:
            with self.subTest(label=label):
                with self.assertRaises(UserInputError):
                    ExpressionDerivedMetric(
                        name="t",
                        input_metric_names=inputs,
                        expression_str=expr,
                    )

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def test_expression_evaluation(self) -> None:
        """Operators, functions, constants, nested calls, and special-char
        metric names (dots, slashes, colons) all evaluate correctly."""
        cases: list[tuple[str, dict[str, float], float]] = [
            ("a + b", {"a": 3.0, "b": 4.0}, 7.0),
            ("a * 2.5 + 10", {"a": 4.0}, 20.0),
            ("-a", {"a": 5.0}, -5.0),
            ("a ** 2", {"a": 3.0}, 9.0),
            ("log(a) - log(b)", {"a": math.e, "b": 1.0}, 1.0),
            ("sqrt(abs(x))", {"x": -16.0}, 4.0),
            ("min(a, b)", {"a": 3.0, "b": 5.0}, 3.0),
            ("max(a, b)", {"a": 3.0, "b": 5.0}, 5.0),
            # Special-char metric names (sanitize_name / unsanitize_name).
            (
                "model.loss - baseline.loss",
                {"model.loss": 5.0, "baseline.loss": 3.0},
                2.0,
            ),
            ("train/acc + val/acc", {"train/acc": 0.8, "val/acc": 0.7}, 1.5),
            ("scope:metric * 2", {"scope:metric": 4.0}, 8.0),
        ]
        for expr, values, expected in cases:
            with self.subTest(expr=expr):
                metric = ExpressionDerivedMetric(
                    name="test",
                    input_metric_names=list(values.keys()),
                    expression_str=expr,
                )
                self.assertAlmostEqual(
                    metric._evaluate_expression(values),
                    expected,
                    places=10,
                )

    # ------------------------------------------------------------------
    # fetch_trial_data
    # ------------------------------------------------------------------

    def test_fetch_trial_data(self) -> None:
        """Basic fetch: correct value, NaN SEM, per-arm differentiation."""
        exp = _make_experiment_with_trial(
            {"0_0": {"a": 10.0, "b": 2.0}, "0_1": {"a": 6.0, "b": 3.0}},
            batch=True,
        )
        metric = ExpressionDerivedMetric(
            name="ratio", input_metric_names=["a", "b"], expression_str="a / b"
        )
        result = metric.fetch_trial_data(exp.trials[0])
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df.sort_values("arm_name").reset_index(drop=True)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.loc[0, "mean"], 5.0)  # 10/2
        self.assertEqual(df.loc[1, "mean"], 2.0)  # 6/3
        self.assertTrue(df["sem"].isna().all())

    def test_expression_evaluation_errors(self) -> None:
        """Err results for expression-specific math errors."""
        with self.subTest("division_by_zero"):
            exp = _make_experiment_with_trial({"0_0": {"a": 10.0, "b": 0.0}})
            metric = ExpressionDerivedMetric(
                name="ratio", input_metric_names=["a", "b"], expression_str="a / b"
            )
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            self.assertIn("division by zero", none_throws(result.err).message.lower())

        with self.subTest("log_of_negative"):
            exp = _make_experiment_with_trial({"0_0": {"a": -1.0}})
            metric = ExpressionDerivedMetric(
                name="log_a", input_metric_names=["a"], expression_str="log(a)"
            )
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            err_msg = none_throws(result.err).message.lower()
            # Sympy versions differ: older raises "math domain error" from
            # math.log; newer generates a guard with "expected a positive
            # input".  Accept either.
            self.assertTrue(
                "math domain error" in err_msg
                or "expected a positive input" in err_msg,
                f"Unexpected error message: {err_msg}",
            )

    # ------------------------------------------------------------------
    # Relativization
    # ------------------------------------------------------------------

    def test_relativize_inputs(self) -> None:
        """Relativized fetch: correct computation, SQ included, multi-arm.
        Also verifies that relativize_inputs=False (default) includes SQ
        and uses raw values."""
        # SQ: a=10, b=4.
        # arm_1: a=15, b=8 → a_rel=50%, b_rel=100% → a+b = 150
        # arm_2: a=20, b=6 → a_rel=100%, b_rel=50% → a+b = 150
        # SQ: a_rel=0, b_rel=0 → a+b = 0
        exp = self._batch_experiment_with_sq(
            sq_values={"a": 10.0, "b": 4.0},
            arm_values={
                "arm_1": {"a": 15.0, "b": 8.0},
                "arm_2": {"a": 20.0, "b": 6.0},
            },
        )
        metric = ExpressionDerivedMetric(
            name="sum_rel",
            input_metric_names=["a", "b"],
            expression_str="a + b",
            relativize_inputs=True,
        )
        result = metric.fetch_trial_data(exp.trials[0])
        self.assertIsInstance(result, Ok)
        df = none_throws(result.ok).df.sort_values("arm_name").reset_index(drop=True)
        # SQ is included: 3 rows (arm_1, arm_2, status_quo).
        self.assertEqual(len(df), 3)
        self.assertIn("status_quo", df["arm_name"].values)
        arm1_row = df[df["arm_name"] == "arm_1"]
        arm2_row = df[df["arm_name"] == "arm_2"]
        sq_row = df[df["arm_name"] == "status_quo"]
        self.assertAlmostEqual(arm1_row["mean"].iloc[0], 150.0, places=10)
        self.assertAlmostEqual(arm2_row["mean"].iloc[0], 150.0, places=10)
        # SQ: zero-valued inputs → a+b = 0.
        self.assertAlmostEqual(sq_row["mean"].iloc[0], 0.0, places=10)
        self.assertTrue(df["sem"].isna().all())

        with self.subTest("sq_evaluates_expression_on_zeros"):
            # exp(0) = 1, verifying the expression is evaluated (not
            # hardcoded to 0) on the SQ arm's zero-valued inputs.
            exp2 = self._batch_experiment_with_sq(
                sq_values={"a": 10.0},
                arm_values={"arm_1": {"a": 15.0}},
            )
            metric2 = ExpressionDerivedMetric(
                name="exp_a",
                input_metric_names=["a"],
                expression_str="exp(a)",
                relativize_inputs=True,
            )
            result2 = metric2.fetch_trial_data(exp2.trials[0])
            self.assertIsInstance(result2, Ok)
            df2 = none_throws(result2.ok).df
            sq_row2 = df2[df2["arm_name"] == "status_quo"]
            self.assertEqual(len(sq_row2), 1)
            # exp(0) = 1.0
            self.assertAlmostEqual(sq_row2["mean"].iloc[0], 1.0, places=10)

        with self.subTest("not_applied_by_default"):
            exp = self._batch_experiment_with_sq(
                sq_values={"a": 10.0, "b": 5.0},
                arm_values={"treat": {"a": 12.0, "b": 10.0}},
            )
            metric = ExpressionDerivedMetric(
                name="sum",
                input_metric_names=["a", "b"],
                expression_str="a + b",
            )
            self.assertFalse(metric.relativize_inputs)
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Ok)
            df = (
                none_throws(result.ok).df.sort_values("arm_name").reset_index(drop=True)
            )
            self.assertEqual(len(df), 2)
            sq_row = df[df["arm_name"] == "status_quo"]
            treat_row = df[df["arm_name"] == "treat"]
            self.assertAlmostEqual(sq_row["mean"].iloc[0], 15.0, places=10)
            self.assertAlmostEqual(treat_row["mean"].iloc[0], 22.0, places=10)

    def test_relativize_errors(self) -> None:
        """Relativization error modes: no SQ, SQ missing from data, SQ near zero."""
        with self.subTest("no_status_quo"):
            exp = _make_experiment_with_trial({"0_0": {"a": 1.0}})
            metric = ExpressionDerivedMetric(
                name="r",
                input_metric_names=["a"],
                expression_str="a",
                relativize_inputs=True,
            )
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            self.assertIn("no status quo", none_throws(result.err).message.lower())

        with self.subTest("sq_not_in_data"):
            sq_arm = Arm(name="sq", parameters={"x1": 0.0, "x2": 0.0})
            exp = Experiment(
                name="test",
                search_space=get_branin_search_space(),
                status_quo=sq_arm,
            )
            trial = exp.new_batch_trial()
            trial.add_arm(sq_arm)
            trial.add_arm(Arm(name="treat", parameters={"x1": 1.0, "x2": 1.0}))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            # Only attach data for "treat", not "sq".
            exp.attach_data(_make_trial_data(0, {"treat": {"a": 5.0}}))
            metric = ExpressionDerivedMetric(
                name="r",
                input_metric_names=["a"],
                expression_str="a",
                relativize_inputs=True,
            )
            result = metric.fetch_trial_data(trial)
            self.assertIsInstance(result, Err)
            # Base class catches missing data for the SQ arm before
            # ExpressionDerivedMetric's relativization check.
            self.assertIn("sq", none_throws(result.err).message)

        with self.subTest("sq_value_near_zero"):
            exp = self._batch_experiment_with_sq(
                sq_values={"a": 0.0},
                arm_values={"treat": {"a": 5.0}},
            )
            metric = ExpressionDerivedMetric(
                name="r",
                input_metric_names=["a"],
                expression_str="a",
                relativize_inputs=True,
            )
            result = metric.fetch_trial_data(exp.trials[0])
            self.assertIsInstance(result, Err)
            self.assertIn(
                "too small to reliably", none_throws(result.err).message.lower()
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def test_serialization_roundtrip(self) -> None:
        """JSON and SQA round-trip, with and without relativize_inputs."""
        for relativize in (False, True):
            original = ExpressionDerivedMetric(
                name="log_ratio",
                input_metric_names=["a", "b"],
                expression_str="log(a) - log(b)",
                relativize_inputs=relativize,
                lower_is_better=True,
                properties={"key": "value"},
            )
            with self.subTest(backend="json", relativize=relativize):
                json_dict = object_to_json(original)
                self.assertEqual(json_dict["__type"], "ExpressionDerivedMetric")
                restored = cast(ExpressionDerivedMetric, object_from_json(json_dict))
                self._assert_metric_equal(restored, original)

            with self.subTest(backend="sqa", relativize=relativize):
                config = SQAConfig()
                sqa = Encoder(config=config).metric_to_sqa(original)
                restored = cast(
                    ExpressionDerivedMetric,
                    Decoder(config=config).metric_from_sqa(sqa),
                )
                self._assert_metric_equal(restored, original)

        with self.subTest("backward_compat_missing_key"):
            original = ExpressionDerivedMetric(
                name="log_ratio",
                input_metric_names=["a", "b"],
                expression_str="log(a) - log(b)",
                lower_is_better=True,
                properties={"key": "value"},
            )
            json_dict = object_to_json(original)
            assert isinstance(json_dict, dict)
            json_dict.pop("relativize_inputs", None)
            json_dict.pop("as_percent", None)
            restored = cast(ExpressionDerivedMetric, object_from_json(json_dict))
            self.assertFalse(restored.relativize_inputs)
            self.assertTrue(restored.as_percent)

    def _assert_metric_equal(
        self,
        restored: ExpressionDerivedMetric,
        original: ExpressionDerivedMetric,
    ) -> None:
        self.assertEqual(restored.name, original.name)
        self.assertEqual(restored.input_metric_names, original.input_metric_names)
        self.assertEqual(restored.expression_str, original.expression_str)
        self.assertEqual(restored.lower_is_better, original.lower_is_better)
        self.assertEqual(restored.properties, original.properties)
        self.assertEqual(restored.relativize_inputs, original.relativize_inputs)
        self.assertEqual(restored.as_percent, original.as_percent)

    # ------------------------------------------------------------------
    # End-to-end integration
    # ------------------------------------------------------------------

    @mock_botorch_optimize
    def test_experiment_integration(self) -> None:
        """Two-phase fetch + TorchAdapter candidate generation with a derived
        constraint that has NaN SEM."""
        derived_ratio = ExpressionDerivedMetric(
            name="ratio",
            input_metric_names=["base_a", "base_b"],
            expression_str="base_a / base_b",
        )
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="obj"), minimize=True),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=derived_ratio,
                        op=ComparisonOp.GEQ,
                        bound=1.0,
                        relative=False,
                    )
                ],
            ),
            tracking_metrics=[
                Metric(name="base_a"),
                Metric(name="base_b"),
                derived_ratio,
            ],
        )

        for i in range(3):
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
                            "base_b": float(i + 1),
                        }
                    },
                )
            )

        data = experiment.fetch_data()
        self.assertIn("ratio", set(data.df["metric_name"].unique()))

        derived_df = data.df[data.df["metric_name"] == "ratio"]
        self.assertTrue(derived_df["sem"].isna().all())
        for i in range(3):
            row = derived_df[derived_df["trial_index"] == i]
            self.assertAlmostEqual(row["mean"].iloc[0], (i + 2) / (i + 1), places=10)

        gr = TorchAdapter(experiment=experiment, generator=BoTorchGenerator()).gen(n=1)
        self.assertEqual(len(gr.arms), 1)
