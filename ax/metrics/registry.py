#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Dict, Optional, Type

from ax.core.metric import Metric
from ax.metrics.branin import BraninMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.noisy_function import NoisyFunctionMetric


"""
Mapping of Metric classes to ints.

All metrics will be stored in the same table in the database. When
saving, we look up the metric subclass in METRIC_REGISTRY, and store
the corresponding type field in the database. When loading, we look
up the type field in REVERSE_METRIC_REGISTRY, and initialize the
corresponding metric subclass.
"""
METRIC_REGISTRY: Dict[Type[Metric], int] = {
    Metric: 0,
    FactorialMetric: 1,
    BraninMetric: 2,
    NoisyFunctionMetric: 3,
    Hartmann6Metric: 4,
}

REVERSE_METRIC_REGISTRY: Dict[int, Type[Metric]] = {
    v: k for k, v in METRIC_REGISTRY.items()
}


def register_metric(metric_cls: Type[Metric], val: Optional[int] = None) -> None:
    """Add a custom metric class to the registries.
    If no int is specified, use a hash of the class name.
    """
    registered_val = val or abs(hash(metric_cls.__name__)) % (10 ** 5)
    METRIC_REGISTRY[metric_cls] = registered_val
    REVERSE_METRIC_REGISTRY[registered_val] = metric_cls
