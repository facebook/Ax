#!/usr/bin/env python3

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.base_trial import TrialStatus
from ae.lazarus.ae.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ae.lazarus.ae.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.core.types.types import ComparisonOp
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.runners.synthetic import SyntheticRunner
from ae.lazarus.ae.storage.json_store.encoders import (
    arm_to_dict,
    batch_to_dict,
    branin_metric_to_dict,
    choice_parameter_to_dict,
    experiment_to_dict,
    fixed_parameter_to_dict,
    generator_run_to_dict,
    metric_to_dict,
    objective_to_dict,
    optimization_config_to_dict,
    order_parameter_constraint_to_dict,
    outcome_constraint_to_dict,
    parameter_constraint_to_dict,
    range_parameter_to_dict,
    search_space_to_dict,
    sum_parameter_constraint_to_dict,
    synthetic_runner_to_dict,
    trial_to_dict,
)
from ae.lazarus.ae.storage.utils import DomainType, ParameterConstraintType


ENCODER_REGISTRY = {
    BatchTrial: batch_to_dict,
    BraninMetric: branin_metric_to_dict,
    ChoiceParameter: choice_parameter_to_dict,
    Arm: arm_to_dict,
    Experiment: experiment_to_dict,
    FixedParameter: fixed_parameter_to_dict,
    GeneratorRun: generator_run_to_dict,
    Metric: metric_to_dict,
    Objective: objective_to_dict,
    OptimizationConfig: optimization_config_to_dict,
    OutcomeConstraint: outcome_constraint_to_dict,
    OrderConstraint: order_parameter_constraint_to_dict,
    ParameterConstraint: parameter_constraint_to_dict,
    RangeParameter: range_parameter_to_dict,
    SearchSpace: search_space_to_dict,
    SumConstraint: sum_parameter_constraint_to_dict,
    SyntheticRunner: synthetic_runner_to_dict,
    Trial: trial_to_dict,
}

DECODER_REGISTRY = {
    "AbandonedArm": AbandonedArm,
    "BatchTrial": BatchTrial,
    "TrialStatus": TrialStatus,
    "BraninMetric": BraninMetric,
    "ChoiceParameter": ChoiceParameter,
    "ComparisonOp": ComparisonOp,
    "Arm": Arm,
    "DomainType": DomainType,
    "Experiment": Experiment,
    "FixedParameter": FixedParameter,
    "GeneratorRun": GeneratorRun,
    "GeneratorRunStruct": GeneratorRunStruct,
    "Metric": Metric,
    "Objective": Objective,
    "OptimizationConfig": OptimizationConfig,
    "OutcomeConstraint": OutcomeConstraint,
    "OrderConstraint": OrderConstraint,
    "ParameterConstraint": ParameterConstraint,
    "ParameterConstraintType": ParameterConstraintType,
    "ParameterType": ParameterType,
    "RangeParameter": RangeParameter,
    "SearchSpace": SearchSpace,
    "SumConstraint": SumConstraint,
    "SyntheticRunner": SyntheticRunner,
    "Trial": Trial,
}
