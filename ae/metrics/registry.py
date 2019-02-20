#!/usr/bin/env python3

from typing import Dict, Type

from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.metrics.factorial import FactorialMetric


class MetricRegistry:
    """Class that contains dictionaries mapping metric classes to ints.

    All metrics will be stored in the same table in the database. When
    saving, we look up the metric subclass in TYPE_TO_CLASS, and store
    the corresponding type field in the database. When loading, we look
    up the type field in CLASS_TO_TYPE, and initialize the corresponding
    metric subclass.

    Create a subclass that inherits from MetricRegistry if you want
    to add support for additional custom metric subclasses.
    """

    CLASS_TO_TYPE: Dict[Type, int] = {Metric: 0, FactorialMetric: 1, BraninMetric: 2}

    TYPE_TO_CLASS: Dict[int, Type] = {v: k for k, v in CLASS_TO_TYPE.items()}
