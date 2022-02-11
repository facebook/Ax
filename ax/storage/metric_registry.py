#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Type

from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.metrics.branin import BraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.chemistry import ChemistryMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.metrics.sklearn import SklearnMetric


"""
Mapping of Metric classes to ints.

All metrics will be stored in the same table in the database. When
saving, we look up the metric subclass in METRIC_REGISTRY, and store
the corresponding type field in the database. When loading, we look
up the type field in REVERSE_METRIC_REGISTRY, and initialize the
corresponding metric subclass.
"""
DEPRECATED_METRIC_REGISTRY: Dict[Type[Metric], int] = {
    Metric: 0,
    FactorialMetric: 1,
    BraninMetric: 2,
    NoisyFunctionMetric: 3,
    Hartmann6Metric: 4,
    SklearnMetric: 5,
    ChemistryMetric: 7,
    MapMetric: 8,
    BraninTimestampMapMetric: 9,
}
