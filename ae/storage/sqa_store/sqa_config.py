#!/usr/bin/env python3

from enum import Enum
from typing import Dict, NamedTuple, Optional, Type

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.base import Base
from ae.lazarus.ae.core.batch_trial import AbandonedArm
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun, GeneratorRunType
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.parameter import Parameter
from ae.lazarus.ae.core.parameter_constraint import ParameterConstraint
from ae.lazarus.ae.core.runner import Runner
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.metrics.registry import MetricRegistry
from ae.lazarus.ae.runners.registry import RunnerRegistry
from ae.lazarus.ae.storage.sqa_store.db import SQABase
from ae.lazarus.ae.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAArm,
    SQAExperiment,
    SQAGeneratorRun,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
    SQATrial,
)


class SQAConfig(NamedTuple):
    """Metadata needed to save and load an experiment to SQLAlchemy.

    Attributes:
        class_to_sqa_class: Mapping of user-facing class to SQLAlchemy class
            that it will be encoded to. This allows overwriting of the default
            classes to provide custom save functionality.
        metric_registry: Maps Metric classes to an int constant representing
            their type. Ensures that when we store metric types, they will
            correspond to an existing Metric class.
        runner_registry: Maps Runner classes to an int constaint representing
            their type. Ensures that when we store runner types, they will
            correspond to an existing Runner class.
        experiment_type_enum: Enum containing valid Experiment types.
        generator_run_type_enum: Enum containing valid Generator Run types.
    """

    class_to_sqa_class: Dict[Type[Base], Type[SQABase]] = {
        AbandonedArm: SQAAbandonedArm,
        Arm: SQAArm,
        Experiment: SQAExperiment,
        GeneratorRun: SQAGeneratorRun,
        Parameter: SQAParameter,
        ParameterConstraint: SQAParameterConstraint,
        Metric: SQAMetric,
        Runner: SQARunner,
        Trial: SQATrial,
    }
    metric_registry: MetricRegistry = MetricRegistry()
    runner_registry: RunnerRegistry = RunnerRegistry()
    experiment_type_enum: Optional[Enum] = None
    generator_run_type_enum: Optional[Enum] = GeneratorRunType
