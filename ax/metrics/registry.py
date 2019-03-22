#!/usr/bin/env python3

from typing import Dict, Optional, Type

from ax.core.metric import Metric
from ax.metrics.branin import BraninMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.noisy_function import NoisyFunctionMetric


class MetricRegistry:
    """Class that contains dictionaries mapping metric classes to ints.

    All metrics will be stored in the same table in the database. When
    saving, we look up the metric subclass in `type_to_class`, and store
    the corresponding type field in the database. When loading, we look
    up the type field in `class_to_type`, and initialize the corresponding
    metric subclass.
    """

    def __init__(self, class_to_type: Optional[Dict[Type, int]] = None):
        self.class_to_type = class_to_type or {
            Metric: 0,
            FactorialMetric: 1,
            BraninMetric: 2,
            NoisyFunctionMetric: 3,
            Hartmann6Metric: 4,
        }
        self.type_to_class = {v: k for k, v in self.class_to_type.items()}
