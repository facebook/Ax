# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module containing the metric base classes for benchmarks. The key property of
a benchmark metric is whether it has a ground truth or not, which is indicated
by a `has_ground_truth` attribute of `BenchmarkMetricBase`. All mnetrics used
in Ax bechmarks need to be subclassed from `BenchmarkMetricBase`.

For metrics that do have a ground truth, we can compute the performance of the
optimization directly in terms of the ground truth observations (or the ground
truth of the out-of-sample model-suggested best point). For metrics that do not
have a ground truth, this is not possible.

The benchmarks are designed in a way so that (unless the metric is noiseless)
no ground truth observations are available to the optimziation algorithm.
Instead, we use separate "ground truth metrics" attached as tracking metrics
to the experiment that are used to evaluate the performance after the
optimization is complete. `GroundTruthMetricMixin` can be used to construct
such ground truth metrics (with the `is_ground_truth` property indicating
that the metric provides the ground truth) and implements naming conventions
and helpers for associating a the ground truth metric to the respective metric
used during the optimization.
"""

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
