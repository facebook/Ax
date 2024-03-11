# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod

from ax.core.metric import Metric


class BenchmarkMetricBase(Metric, ABC):
    """A generic metric used for Ax Benchmarks.

    Attributes:
        has_ground_truth: Whether or not there exists a ground truth for this
            metric, i.e. whether each observation has an associated ground
            truth value. This is trivially true for deterministic metrics, and
            is also true for metrics where synthetic observation noise is added
            to its (deterministic) values. This is not true for metrics that
            are inherently noisy.
    """

    has_ground_truth: bool

    @abstractmethod
    def make_ground_truth_metric(self) -> BenchmarkMetricBase:
        """Create a ground truth version of this metric. If metric observations
        are noisy, the ground truth would be the underlying noiseless values."""


class GroundTruthMetricMixin(ABC):
    """A mixin for metrics that defines a naming convention and associated helper
    methods that allow mapping from a ground truth metric to its original metric
    and vice versa."""

    is_ground_truth: bool = True
    _GROUND_TRUTH_SUFFIX = "__GROUND_TRUTH"

    @classmethod
    def get_ground_truth_name(cls, metric: Metric) -> str:
        return f"{metric.name}{cls._GROUND_TRUTH_SUFFIX}"

    @classmethod
    def get_original_name(cls, full_name: str) -> str:
        if not full_name.endswith(cls._GROUND_TRUTH_SUFFIX):
            raise ValueError("full_name does not end with ground truth suffix.")
        return full_name.replace(cls._GROUND_TRUTH_SUFFIX, "")
