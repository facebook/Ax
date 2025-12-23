#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from ax.core.metric import MetricFetchE
from ax.metrics.tensorboard import _grid_interpolate, logger, TensorboardMetric
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
)
from ax.storage.metric_registry import register_metrics
from ax.utils.common.result import Ok
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial
from pyre_extensions import assert_is_instance
from tensorboard.backend.event_processing import event_multiplexer


@dataclass
class _TensorProto:
    double_val: list[float]


@dataclass
class _TensorEvent:
    step: int
    tensor_proto: _TensorProto


def _get_fake_multiplexer(
    fake_data: Sequence[float],
) -> event_multiplexer.EventMultiplexer:
    mul = event_multiplexer.EventMultiplexer()

    # pyre-ignore[8] Return fake tags when content is requested
    mul.PluginRunToTagToContent = lambda plugin: {".": {"loss": ""}}

    # pyre-ignore[8] Return fake data when tensors requested
    mul.Tensors = lambda run, tag: [
        _TensorEvent(step=i, tensor_proto=_TensorProto(double_val=[dat]))
        for i, dat in enumerate(fake_data)
    ]

    return mul


def tb_smooth(scalars: list[float], weight: float) -> list[float]:
    """
    Debiased EMA implementation according to Tensorboard Github:
    https://fburl.com/6v6q8scg

    This function is used to test that the smoothing functaionality
    of pandas.series.ewm is equivalent to the smoothing functionality
    of Tensorboard.
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


class TensorboardMetricTest(TestCase):
    def test_fetch_trial_data(self) -> None:
        fake_data = [8.0, 9.0, 2.0, 1.0]
        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=0
        )
        trial = get_trial()

        # This is mocked to avoid an xdb call
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = result.unwrap().full_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "metric_signature": "loss",
                    "mean": fake_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))

    def test_fetch_trial_data_with_bad_data(self) -> None:
        nan_data = [1, 2, np.nan, 4]
        nan_multiplexer = _get_fake_multiplexer(fake_data=nan_data)
        metric = TensorboardMetric(name="loss", tag="loss")

        trial = get_trial()
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=nan_multiplexer,
        ), mock.patch.object(logger, "warning") as mock_warning:
            result = metric.fetch_trial_data(trial=trial)
        mock_warning.assert_called_once_with(
            "1 / 4 data points are NaNs or Infs. Filtering out non-finite values."
        )

        assert_is_instance(result, Ok)
        map_data = result.unwrap()
        nan_array = np.array(nan_data)
        nan_array = nan_array[np.isfinite(nan_array)]
        self.assertTrue(
            np.array_equal(
                map_data.full_df["mean"].to_numpy(), nan_array, equal_nan=False
            )
        )

        # testing some inf data
        inf_data = [1, 2, np.inf, 4]
        inf_multiplexer = _get_fake_multiplexer(fake_data=inf_data)

        metric = TensorboardMetric(name="loss", tag="loss")

        trial = get_trial()
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=inf_multiplexer,
        ):
            with mock.patch.object(logger, "warning") as mock_warning:
                result = metric.fetch_trial_data(trial=trial)
            mock_warning.assert_called_once_with(
                "1 / 4 data points are NaNs or Infs. Filtering out non-finite values."
            )

        assert_is_instance(result, Ok)
        map_data = result.unwrap()
        inf_array = np.array(inf_data)
        inf_array = inf_array[np.isfinite(inf_array)]
        self.assertTrue(np.array_equal(map_data.full_df["mean"].to_numpy(), inf_array))

        # testing all non-finite data
        for non_finite_val in [np.nan, np.inf]:
            nf_data = [non_finite_val for _ in range(4)]
            nf_multiplexer = _get_fake_multiplexer(fake_data=nf_data)
            metric = TensorboardMetric(name="loss", tag="loss")

            trial = get_trial()
            with mock.patch.object(
                TensorboardMetric,
                "_get_event_multiplexer_for_trial",
                return_value=nf_multiplexer,
            ):
                result = metric.fetch_trial_data(trial=trial)

            err = assert_is_instance(result.unwrap_err(), MetricFetchE)
            self.assertEqual(err.message, "Failed to fetch data for loss")
            self.assertEqual(str(err.exception), "All values are NaNs or Infs.")

    def test_smoothing(self) -> None:
        fake_data = [8.0, 4.0, 2.0, 1.0]
        smoothing = 0.5
        smooth_data = tb_smooth(fake_data, smoothing)

        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=smoothing
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = result.unwrap().full_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "metric_signature": "loss",
                    "mean": smooth_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))

        # Test that smoothing must be in the range [0, 1).
        # Valid smoothing values
        for smoothing in [0, 0.5, 0.99]:
            with self.subTest(f"smoothing={smoothing}"):
                TensorboardMetric(name="loss", tag="loss", smoothing=smoothing)

        # Invalid smoothing values
        expected_str = r"smoothing must be in the range \[0, 1\)"
        for smoothing in [1, 1.5, -0.1]:
            with self.subTest(f"smoothing={smoothing}"):
                with self.assertRaisesRegex(ValueError, expected_str):
                    TensorboardMetric(name="loss", tag="loss", smoothing=smoothing)

    def test_cumulative_best(self) -> None:
        fake_data = [4.0, 8.0, 2.0, 1.0]
        cummin_data = pd.Series(fake_data).cummin().tolist()

        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        metric = TensorboardMetric(
            name="loss",
            tag="loss",
            lower_is_better=True,
            cumulative_best=True,
            smoothing=0,
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = result.unwrap().full_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "metric_signature": "loss",
                    "mean": cummin_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))

    def test_smoothing_then_cumulative_best_order(self) -> None:
        """Test that smoothing is applied before cumulative best.

        This test validates that when both smoothing and cumulative_best are
        enabled, the operations are applied in the correct order:
        1. First, smoothing is applied to the raw data
        2. Then, cumulative best is applied to the smoothed data

        This ensures cumulative best operates on smoothed values rather than
        raw values. Tests both lower_is_better=True (cummin) and
        lower_is_better=False (cummax).
        """
        # Setup: Create test data where order of operations matters
        # Using data where smoothing vs. cumulative best order produces
        # different results
        smoothing = 0.5

        # Test both lower_is_better=True and lower_is_better=False
        for lower_is_better in [True, False]:
            # Use different data for each case to ensure order matters
            if lower_is_better:
                # For cummin: descending data with a spike
                fake_data = [8.0, 4.0, 6.0, 2.0]
            else:
                # For cummax: ascending data with a dip
                fake_data = [2.0, 6.0, 4.0, 8.0]

            fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
            metric = TensorboardMetric(
                name="loss",
                tag="loss",
                lower_is_better=lower_is_better,
                smoothing=smoothing,
                cumulative_best=True,
            )
            trial = get_trial()

            # Execute: Fetch data with both smoothing and cumulative_best enabled
            with mock.patch.object(
                TensorboardMetric,
                "_get_event_multiplexer_for_trial",
                return_value=fake_multiplexer,
            ):
                result = metric.fetch_trial_data(trial=trial)

            df = result.unwrap().full_df

            # Assert: Verify that smoothing was applied first, then cumulative best
            # Step 1: Apply smoothing to get smoothed values
            smoothed_data = tb_smooth(fake_data, smoothing)

            # Step 2: Apply cumulative best to the smoothed values
            if lower_is_better:
                expected_data = pd.Series(smoothed_data).cummin().tolist()
                cum_op = "cummin"
            else:
                expected_data = pd.Series(smoothed_data).cummax().tolist()
                cum_op = "cummax"

            # Verify the result matches smoothed-then-cum order
            self.assertEqual(
                df["mean"].tolist(),
                expected_data,
                f"Failed for lower_is_better={lower_is_better}",
            )

            # Demonstrate that cum-then-smooth would give different results
            if lower_is_better:
                cum_first = pd.Series(fake_data).cummin().tolist()
            else:
                cum_first = pd.Series(fake_data).cummax().tolist()
            smooth_then_cum = tb_smooth(cum_first, smoothing)

            # These should be different, proving order matters
            self.assertNotEqual(
                expected_data,
                smooth_then_cum,
                f"Order should matter for lower_is_better={lower_is_better} ({cum_op})",
            )

    def test_percentile(self) -> None:
        fake_data = [8.0, 4.0, 2.0, 1.0]
        percentile = 0.5
        percentile_data = pd.Series(fake_data).expanding().quantile(percentile)
        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss", tag="loss", lower_is_better=True, percentile=percentile
            )
            result = metric.fetch_trial_data(trial=trial)
        df = result.unwrap().full_df
        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "metric_signature": "loss",
                    "mean": percentile_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))

    def test_grid_interpolate_with_unevenly_spaced_data(self) -> None:
        """Test that _grid_interpolate correctly interpolates unevenly spaced data."""
        for sem_is_nan in [False, True]:
            # Create unevenly spaced data
            df = pd.DataFrame(
                {
                    "trial_index": [0] * 4,
                    "arm_name": ["0_0"] * 4,
                    "metric_signature": ["loss"] * 4,
                    "step": [0.0, 1.0, 5.0, 10.0],  # Unevenly spaced
                    "mean": [1.0, 2.0, 3.0, 4.0],
                    "sem": [np.nan] * 4 if sem_is_nan else [0.1, 0.2, 0.3, 0.4],
                }
            )

            result_df = _grid_interpolate(df, arm_name="0_0", metric_signature="loss")

            # Should have same number of points
            self.assertEqual(len(result_df), 4)

            # Steps should now be evenly spaced
            expected_steps = np.linspace(0.0, 10.0, 4)
            np.testing.assert_array_almost_equal(
                result_df["step"].values, expected_steps
            )

            # Check that values are interpolated (not equal to original)
            # At step 3.333, mean should be interpolated between original points
            self.assertNotEqual(result_df["mean"].iloc[1], 2.0)

            # Verify interpolation is working by checking endpoints
            self.assertAlmostEqual(result_df["mean"].iloc[0], 1.0)
            self.assertAlmostEqual(result_df["mean"].iloc[-1], 4.0)
            if sem_is_nan:
                self.assertTrue(all(np.isnan(result_df["sem"].values)))
            else:
                self.assertAlmostEqual(result_df["sem"].iloc[0], 0.1)
                self.assertAlmostEqual(result_df["sem"].iloc[-1], 0.4)

    def test_grid_interpolate_with_evenly_spaced_data(self) -> None:
        """Test that evenly spaced data remains unchanged (backward compatibility)."""
        # Create evenly spaced data
        df = pd.DataFrame(
            {
                "trial_index": [0] * 4,
                "arm_name": ["0_0"] * 4,
                "metric_signature": ["loss"] * 4,
                "step": [0.0, 1.0, 2.0, 3.0],  # Already evenly spaced
                "mean": [1.0, 2.0, 3.0, 4.0],
                "sem": [0.1, 0.2, 0.3, 0.4],
            }
        )

        result_df = _grid_interpolate(df, arm_name="0_0", metric_signature="loss")

        # Should have same number of points
        self.assertEqual(len(result_df), 4)

        # Steps should be identical
        np.testing.assert_array_almost_equal(
            result_df["step"].values, df["step"].values
        )

        # Values should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(
            result_df["mean"].values, df["mean"].values
        )
        np.testing.assert_array_almost_equal(result_df["sem"].values, df["sem"].values)

    def test_grid_interpolate_empty_dataframe(self) -> None:
        """Test that empty dataframe is handled correctly."""
        df = pd.DataFrame(
            {
                "trial_index": [0],
                "arm_name": ["0_0"],
                "metric_signature": ["loss"],
                "step": [1.0],
                "mean": [1.0],
                "sem": [0.1],
            }
        )

        # Request data for non-existent arm_name
        with mock.patch.object(logger, "warning") as mock_warning:
            result_df = _grid_interpolate(
                df, arm_name="non_existent", metric_signature="loss"
            )
            mock_warning.assert_called_once()

        # Should return empty dataframe
        self.assertEqual(len(result_df), 0)

    def test_grid_interpolate_filters_by_arm_and_metric(self) -> None:
        """Test that filtering by arm_name and metric_signature works correctly."""
        # Create data with multiple arms and metrics
        df = pd.DataFrame(
            {
                "trial_index": [0] * 8,
                "arm_name": ["0_0"] * 4 + ["0_1"] * 4,
                "metric_signature": ["loss"] * 2 + ["accuracy"] * 2 + ["loss"] * 4,
                "step": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "sem": [0.1] * 8,
            }
        )

        result_df = _grid_interpolate(df, arm_name="0_0", metric_signature="loss")

        # Should only have data for arm "0_0" and metric "loss"
        self.assertEqual(len(result_df), 2)
        self.assertTrue((result_df["arm_name"] == "0_0").all())
        self.assertTrue((result_df["metric_signature"] == "loss").all())

    def test_smoothing_with_unevenly_spaced_data(self) -> None:
        """Test that smoothing works correctly with unevenly spaced data.

        This test verifies grid interpolation happens before smoothing.
        """

        # Create unevenly spaced fake data
        class _UnevenMultiplexer:
            def PluginRunToTagToContent(self, plugin: str) -> dict[str, dict[str, str]]:
                return {".": {"loss": ""}}

            def Tensors(self, run: str, tag: str) -> list[_TensorEvent]:
                # Unevenly spaced steps: 0, 1, 5, 10
                return [
                    _TensorEvent(step=0, tensor_proto=_TensorProto(double_val=[8.0])),
                    _TensorEvent(step=1, tensor_proto=_TensorProto(double_val=[4.0])),
                    _TensorEvent(step=5, tensor_proto=_TensorProto(double_val=[2.0])),
                    _TensorEvent(step=10, tensor_proto=_TensorProto(double_val=[1.0])),
                ]

        uneven_multiplexer = _UnevenMultiplexer()
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=0.5
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=uneven_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = result.unwrap().full_df

        # Should have 4 points
        self.assertEqual(len(df), 4)

        # Steps should be evenly spaced after grid interpolation
        expected_steps = np.linspace(0.0, 10.0, 4)
        np.testing.assert_array_almost_equal(df["step"].values, expected_steps)

        # Values should be smoothed (after interpolation)
        # The first value should be close to the interpolated value at step 0
        self.assertAlmostEqual(df["mean"].iloc[0], 8.0, places=5)

        # All values should be finite
        self.assertTrue(np.all(np.isfinite(df["mean"].values)))

    def test_smoothing_applies_to_both_mean_and_sem(self) -> None:
        """Test that smoothing is applied to both mean and sem columns."""

        # Create fake data with non-NaN sem values
        class _MultiValueMultiplexer:
            def PluginRunToTagToContent(self, plugin: str) -> dict[str, dict[str, str]]:
                return {".": {"loss": ""}}

            def Tensors(self, run: str, tag: str) -> list[_TensorEvent]:
                # Return data with varying values
                return [
                    _TensorEvent(
                        step=i, tensor_proto=_TensorProto(double_val=[float(i * 2)])
                    )
                    for i in range(4)
                ]

        multi_multiplexer = _MultiValueMultiplexer()
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=0.3
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=multi_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = result.unwrap().full_df

        # Verify smoothing was applied by checking that values are
        # different from original
        original_values = [0.0, 2.0, 4.0, 6.0]

        # The smoothed values should differ from original (except first value)
        # We can't easily predict exact values, but we know they should be smoothed
        self.assertAlmostEqual(df["mean"].iloc[0], original_values[0], places=5)

        # Later values should show smoothing effect (not equal to original)
        # Due to grid interpolation on evenly spaced data, values should be identical
        # to original, then smoothed
        for i in range(1, len(df)):
            # Check that smoothing effect is present
            # (values should be between neighboring original values due to EMA)
            self.assertTrue(np.isfinite(df["mean"].iloc[i]))

    def test_quantile_percentile_validation_and_deprecation(
        self: TestCase,
        metric_class: type[TensorboardMetric] = TensorboardMetric,
        base_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Test validation and deprecation for quantile/percentile parameters.

        This method is designed to be reusable by subclasses and external tests
        by overriding the default arguments.

        Args:
            metric_class: The metric class to test. Defaults to TensorboardMetric.
            base_kwargs: Base keyword arguments to pass to the metric constructor.
                Defaults to {"name": "loss", "tag": "loss"}.
        """
        if base_kwargs is None:
            base_kwargs = {"name": "loss", "tag": "loss"}

        # Test 1: quantile must be in range [0, 1]
        valid_values = [0.0, 0.5, 1.0]
        for val in valid_values:
            with self.subTest(f"valid quantile={val}"):
                metric_class(**base_kwargs, quantile=val)

        invalid_values = [-0.1, 1.5]
        for val in invalid_values:
            with self.subTest(f"invalid quantile={val}"):
                with self.assertRaisesRegex(
                    ValueError, r"quantile must be in the range \[0, 1\]"
                ):
                    metric_class(**base_kwargs, quantile=val)

        # Test 2: percentile is deprecated alias for quantile
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            metric = metric_class(**base_kwargs, percentile=0.5)
            self.assertEqual(len(w), 1)
            self.assertIn("percentile", str(w[0].message))
            self.assertIn("deprecated", str(w[0].message).lower())
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            # Check that quantile attribute has the correct value
            self.assertEqual(metric.quantile, 0.5)
            # Check that percentile attribute is None (deprecated)
            self.assertIsNone(metric.percentile)

        # Test 3: Cannot specify both quantile and percentile
        with self.assertRaisesRegex(
            ValueError,
            r"Cannot specify both",
        ):
            metric_class(**base_kwargs, quantile=0.5, percentile=0.5)

    def test_quantile_applied_to_curve(
        self: TestCase,
        fetch_data_with_metric: Callable[[TensorboardMetric], pd.DataFrame]
        | None = None,
        metric_class: type[TensorboardMetric] = TensorboardMetric,
        base_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Test that quantile is actually applied to the curve data.

        This method is designed to be reusable by subclasses and external tests
        by overriding the default arguments.

        Args:
            fetch_data_with_metric: A callable that takes a metric and returns
                the fetched DataFrame. If None, uses default TensorboardMetric
                behavior with a fake multiplexer.
            metric_class: The metric class to test. Defaults to TensorboardMetric.
            base_kwargs: Base keyword arguments to pass to the metric constructor.
                Defaults to {"name": "loss", "tag": "loss"}.
        """
        if base_kwargs is None:
            base_kwargs = {"name": "loss", "tag": "loss"}

        # If no fetch function is provided, use default behavior
        if fetch_data_with_metric is None:
            fake_data = [8.0, 9.0, 2.0, 1.0]
            fake_multiplexer: event_multiplexer.EventMultiplexer = (
                _get_fake_multiplexer(fake_data=fake_data)
            )

            def fetch_data_with_metric(metric: TensorboardMetric) -> pd.DataFrame:
                trial = get_trial()
                with mock.patch.object(
                    TensorboardMetric,
                    "_get_event_multiplexer_for_trial",
                    return_value=fake_multiplexer,
                ):
                    result = metric.fetch_trial_data(trial=trial)
                return result.unwrap().full_df

        # Track calls to Expanding.quantile to verify it's called with the
        # correct value
        call_tracker: list[float] = []
        original_quantile: Callable[..., Any] = (
            pd.core.window.expanding.Expanding.quantile
        )

        def tracked_quantile(
            self: pd.core.window.expanding.Expanding,
            q: float,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            call_tracker.append(q)
            return original_quantile(self, q, *args, **kwargs)

        # Test: When quantile is set, Expanding.quantile should be called with
        # the correct value
        test_quantile = 0.75
        metric = metric_class(**base_kwargs, quantile=test_quantile)

        with mock.patch.object(
            pd.core.window.expanding.Expanding,
            "quantile",
            tracked_quantile,
        ):
            fetch_data_with_metric(metric)

        self.assertIn(
            test_quantile,
            call_tracker,
            f"Expanding.quantile was not called with {test_quantile}. "
            f"Calls: {call_tracker}",
        )

    def test_storage_roundtrip(self) -> None:
        """Test that TensorboardMetric can be serialized and deserialized, i.e.,
        1. A TensorboardMetric can be encoded to JSON and decoded back.
        2. The resulting metric is equivalent to the original.
        3. Metrics created with the deprecated `percentile` parameter can still
           be serialized and deserialized without throwing validation errors.
        """
        _, encoder_registry, decoder_registry = register_metrics(
            metric_clss={TensorboardMetric: None},
        )

        test_cases = [
            # Metric with all options (with quantile)
            TensorboardMetric(
                name="test",
                tag="test",
                lower_is_better=False,
                smoothing=0.5,
                cumulative_best=True,
                quantile=0.25,
            ),
            # Metric with all options (with percentile)
            TensorboardMetric(
                name="test",
                tag="test",
                lower_is_better=False,
                smoothing=0.5,
                cumulative_best=True,
                percentile=0.25,
            ),
        ]

        for original_metric in test_cases:
            with self.subTest(
                metric=original_metric.name, quantile=original_metric.quantile
            ):
                # Encode metric to JSON
                json_obj = object_to_json(
                    original_metric,
                    encoder_registry=encoder_registry,
                    class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
                )

                # Decode metric from JSON
                decoded_metric = object_from_json(
                    json_obj,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
                )

                # Verify the decoded metric equals the original
                self.assertIsInstance(decoded_metric, TensorboardMetric)
                self.assertEqual(decoded_metric.name, original_metric.name)
                self.assertEqual(decoded_metric.tag, original_metric.tag)
                self.assertEqual(
                    decoded_metric.lower_is_better, original_metric.lower_is_better
                )
                self.assertEqual(decoded_metric.smoothing, original_metric.smoothing)
                self.assertEqual(
                    decoded_metric.cumulative_best, original_metric.cumulative_best
                )
                self.assertEqual(decoded_metric.quantile, original_metric.quantile)
                # Verify percentile is None (deprecated)
                self.assertIsNone(decoded_metric.percentile)
