#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from enum import Enum
from typing import Dict, NamedTuple, Optional, Type

from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.batch_trial import AbandonedArm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.metric import Metric
from ax.core.parameter import Parameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.storage.sqa_store.db import SQABase
from ax.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAArm,
    SQAData,
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
        experiment_type_enum: Enum containing valid Experiment types.
        generator_run_type_enum: Enum containing valid Generator Run types.
    """

    class_to_sqa_class: Dict[Type[Base], Type[SQABase]] = {
        AbandonedArm: SQAAbandonedArm,
        Arm: SQAArm,
        Data: SQAData,
        Experiment: SQAExperiment,
        GeneratorRun: SQAGeneratorRun,
        Parameter: SQAParameter,
        ParameterConstraint: SQAParameterConstraint,
        Metric: SQAMetric,
        Runner: SQARunner,
        Trial: SQATrial,
    }
    experiment_type_enum: Optional[Enum] = None
    generator_run_type_enum: Optional[Enum] = GeneratorRunType
