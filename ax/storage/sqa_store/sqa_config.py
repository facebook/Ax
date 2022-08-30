#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

from ax.core.arm import Arm
from ax.core.batch_trial import AbandonedArm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.metric import Metric
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
)
from ax.storage.metric_registry import CORE_METRIC_REGISTRY
from ax.storage.runner_registry import CORE_RUNNER_REGISTRY
from ax.storage.sqa_store.db import SQABase
from ax.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAArm,
    SQAData,
    SQAExperiment,
    SQAGenerationStrategy,
    SQAGeneratorRun,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
    SQATrial,
)
from ax.utils.common.base import Base


@dataclass
class SQAConfig:
    """Metadata needed to save and load an experiment to SQLAlchemy.

    Attributes:
        class_to_sqa_class: Mapping of user-facing class to SQLAlchemy class
            that it will be encoded to. This allows overwriting of the default
            classes to provide custom save functionality.
        experiment_type_enum: Enum containing valid Experiment types.
        generator_run_type_enum: Enum containing valid Generator Run types.
        json_encoder_registry: Mapping from user-facing types to their json
            serialization function.
    """

    def _default_class_to_sqa_class(self=None) -> Dict[Type[Base], Type[SQABase]]:
        # pyre-ignore [7]
        return {
            AbandonedArm: SQAAbandonedArm,
            Arm: SQAArm,
            Data: SQAData,
            Experiment: SQAExperiment,
            GenerationStrategy: SQAGenerationStrategy,
            GeneratorRun: SQAGeneratorRun,
            Parameter: SQAParameter,
            ParameterConstraint: SQAParameterConstraint,
            Metric: SQAMetric,
            Runner: SQARunner,
            Trial: SQATrial,
        }

    class_to_sqa_class: Dict[Type[Base], Type[SQABase]] = field(
        default_factory=_default_class_to_sqa_class
    )
    experiment_type_enum: Optional[Enum] = None
    generator_run_type_enum: Optional[Enum] = GeneratorRunType  # pyre-ignore [8]

    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    json_encoder_registry: Dict[Type, Callable[[Any], Dict[str, Any]]] = field(
        default_factory=lambda: CORE_ENCODER_REGISTRY
    )
    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    json_class_encoder_registry: Dict[Type, Callable[[Any], Dict[str, Any]]] = field(
        default_factory=lambda: CORE_CLASS_ENCODER_REGISTRY
    )
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    json_decoder_registry: Dict[str, Type] = field(
        default_factory=lambda: CORE_DECODER_REGISTRY
    )
    # pyre-fixme[4]: Attribute annotation cannot contain `Any`.
    json_class_decoder_registry: Dict[str, Callable[[Dict[str, Any]], Any]] = field(
        default_factory=lambda: CORE_CLASS_DECODER_REGISTRY
    )

    metric_registry: Dict[Type[Metric], int] = field(
        default_factory=lambda: CORE_METRIC_REGISTRY
    )
    runner_registry: Dict[Type[Runner], int] = field(
        default_factory=lambda: CORE_RUNNER_REGISTRY
    )

    @property
    def reverse_metric_registry(self) -> Dict[int, Type[Metric]]:
        return {v: k for k, v in self.metric_registry.items()}

    @property
    def reverse_runner_registry(self) -> Dict[int, Type[Runner]]:
        return {v: k for k, v in self.runner_registry.items()}
