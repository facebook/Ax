#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Any, Callable, Optional, Dict, Type

from ax.core.runner import Runner
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.encoders import runner_to_dict
from ax.storage.json_store.registry import CORE_ENCODER_REGISTRY, CORE_DECODER_REGISTRY
from ax.utils.common.logger import get_logger

# TODO[T113829027] Remove in a few months
logger = get_logger(__name__)
WARNING_MSG = (
    "There have been some recent changes to `register_metric`. Please see "
    "https://ax.dev/tutorials/gpei_hartmann_developer.html#9.-Save-to-JSON-or-SQL "
    "for the most up-to-date information on saving custom runners."
)

# """
# Mapping of Runner classes to ints.

# All runners will be stored in the same table in the database. When
# saving, we look up the runner subclass in RUNNER_REGISTRY, and store
# the corresponding type field in the database. When loading, we look
# up the type field in REVERSE_RUNNER_REGISTRY, and initialize the
# corresponding runner subclass.
# """
CORE_RUNNER_REGISTRY: Dict[Type[Runner], int] = {SyntheticRunner: 0}


def register_runner(
    runner_cls: Type[Runner],
    runner_registry: Dict[Type[Runner], int] = CORE_RUNNER_REGISTRY,
    encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_ENCODER_REGISTRY,
    decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
    val: Optional[int] = None,
) -> Tuple[
    Dict[Type[Runner], int],
    Dict[Type, Callable[[Any], Dict[str, Any]]],
    Dict[str, Type],
]:
    """Add a custom runner class to the SQA and JSON registries.
    For the SQA registry, if no int is specified, use a hash of the class name.
    """
    logger.warn(WARNING_MSG)

    registered_val = val or abs(hash(runner_cls.__name__)) % (10 ** 5)

    new_runner_registry = {runner_cls: registered_val, **runner_registry}
    new_encoder_registry = {runner_cls: runner_to_dict, **encoder_registry}
    new_decoder_registry = {runner_cls.__name__: runner_cls, **decoder_registry}

    return new_runner_registry, new_encoder_registry, new_decoder_registry


def register_runners(
    runner_clss: Dict[Type[Runner], Optional[int]],
    runner_registry: Dict[Type[Runner], int] = CORE_RUNNER_REGISTRY,
    encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_ENCODER_REGISTRY,
    decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
) -> Tuple[
    Dict[Type[Runner], int],
    Dict[Type, Callable[[Any], Dict[str, Any]]],
    Dict[str, Type],
]:
    """Add custom runner classes to the SQA and JSON registries.
    For the SQA registry, if no int is specified, use a hash of the class name.
    """
    logger.warn(WARNING_MSG)

    new_runner_registry = {
        **{
            runner_cls: val if val else abs(hash(runner_cls.__name__)) % (10 ** 5)
            for runner_cls, val in runner_clss.items()
        },
        **runner_registry,
    }
    new_encoder_registry = {
        **{runner_cls: runner_to_dict for runner_cls in runner_clss},
        **encoder_registry,
    }
    new_decoder_registry = {
        **{runner_cls.__name__: runner_cls for runner_cls in runner_clss},
        **decoder_registry,
    }

    return new_runner_registry, new_encoder_registry, new_decoder_registry
