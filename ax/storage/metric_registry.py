#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional, Any, Callable, Dict, Type

from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.metrics.branin import BraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.chemistry import ChemistryMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.metrics.sklearn import SklearnMetric
from ax.storage.json_store.encoders import metric_to_dict
from ax.storage.json_store.registry import CORE_ENCODER_REGISTRY, CORE_DECODER_REGISTRY
from ax.utils.common.logger import get_logger

# TODO[T113829027] Remove in a few months
logger = get_logger(__name__)
WARNING_MSG = (
    "There have been some recent changes to `register_metric`. Please see "
    "https://ax.dev/tutorials/gpei_hartmann_developer.html#9.-Save-to-JSON-or-SQL "
    "for the most up-to-date information on saving custom metrics."
)

"""
Mapping of Metric classes to ints.

All metrics will be stored in the same table in the database. When
saving, we look up the metric subclass in METRIC_REGISTRY, and store
the corresponding type field in the database.
"""
CORE_METRIC_REGISTRY: Dict[Type[Metric], int] = {
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


def register_metric(
    metric_cls: Type[Metric],
    metric_registry: Optional[Dict[Type[Metric], int]] = None,
    encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_ENCODER_REGISTRY,
    decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
    val: Optional[int] = None,
) -> Tuple[
    Dict[Type[Metric], int],
    Dict[Type, Callable[[Any], Dict[str, Any]]],
    Dict[str, Type],
]:
    """Add a custom metric class to the SQA and JSON registries.
    For the SQA registry, if no int is specified, use a hash of the class name.
    """
    logger.warn(WARNING_MSG)

    metric_registry = metric_registry or {Metric: 0}

    registered_val = val or abs(hash(metric_cls.__name__)) % (10 ** 5)

    new_metric_registry = {metric_cls: registered_val, **metric_registry}
    new_encoder_registry = {metric_cls: metric_to_dict, **encoder_registry}
    new_decoder_registry = {metric_cls.__name__: metric_cls, **decoder_registry}

    return new_metric_registry, new_encoder_registry, new_decoder_registry


def register_metrics(
    metric_clss: Dict[Type[Metric], Optional[int]],
    metric_registry: Optional[Dict[Type[Metric], int]] = None,
    encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_ENCODER_REGISTRY,
    decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
) -> Tuple[
    Dict[Type[Metric], int],
    Dict[Type, Callable[[Any], Dict[str, Any]]],
    Dict[str, Type],
]:
    """Add custom metric classes to the SQA and JSON registries.
    For the SQA registry, if no int is specified, use a hash of the class name.
    """
    logger.warn(WARNING_MSG)

    metric_registry = metric_registry or {Metric: 1}

    new_metric_registry = {
        **{
            metric_cls: val if val else abs(hash(metric_cls.__name__)) % (10 ** 5)
            for metric_cls, val in metric_clss.items()
        },
        **metric_registry,
    }
    new_encoder_registry = {
        **{metric_cls: metric_to_dict for metric_cls in metric_clss},
        **encoder_registry,
    }
    new_decoder_registry = {
        **{metric_cls.__name__: metric_cls for metric_cls in metric_clss},
        **decoder_registry,
    }

    return new_metric_registry, new_encoder_registry, new_decoder_registry
