#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Type

from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.benchmark.benchmark_result import BenchmarkResult
from ax.core import ObservationFeatures
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.encoders import (
    arm_to_dict,
    batch_to_dict,
    benchmark_problem_to_dict,
    choice_parameter_to_dict,
    data_to_dict,
    experiment_to_dict,
    fixed_parameter_to_dict,
    generation_strategy_to_dict,
    generator_run_to_dict,
    metric_to_dict,
    multi_objective_to_dict,
    objective_to_dict,
    observation_features_to_dict,
    optimization_config_to_dict,
    order_parameter_constraint_to_dict,
    outcome_constraint_to_dict,
    parameter_constraint_to_dict,
    range_parameter_to_dict,
    runner_to_dict,
    scalarized_objective_to_dict,
    search_space_to_dict,
    simple_experiment_to_dict,
    sum_parameter_constraint_to_dict,
    transform_type_to_dict,
    trial_to_dict,
)
from ax.storage.utils import DomainType, ParameterConstraintType


ENCODER_REGISTRY: Dict[Type, Callable[[Any], Dict[str, Any]]] = {
    Arm: arm_to_dict,
    BatchTrial: batch_to_dict,
    BenchmarkProblem: benchmark_problem_to_dict,
    BraninMetric: metric_to_dict,
    ChoiceParameter: choice_parameter_to_dict,
    Data: data_to_dict,
    Experiment: experiment_to_dict,
    FactorialMetric: metric_to_dict,
    FixedParameter: fixed_parameter_to_dict,
    GenerationStrategy: generation_strategy_to_dict,
    GeneratorRun: generator_run_to_dict,
    Hartmann6Metric: metric_to_dict,
    L2NormMetric: metric_to_dict,
    Metric: metric_to_dict,
    MultiObjective: multi_objective_to_dict,
    NegativeBraninMetric: metric_to_dict,
    NoisyFunctionMetric: metric_to_dict,
    Objective: objective_to_dict,
    OptimizationConfig: optimization_config_to_dict,
    OrderConstraint: order_parameter_constraint_to_dict,
    OutcomeConstraint: outcome_constraint_to_dict,
    ParameterConstraint: parameter_constraint_to_dict,
    RangeParameter: range_parameter_to_dict,
    ScalarizedObjective: scalarized_objective_to_dict,
    SearchSpace: search_space_to_dict,
    SimpleBenchmarkProblem: benchmark_problem_to_dict,
    SimpleExperiment: simple_experiment_to_dict,
    SumConstraint: sum_parameter_constraint_to_dict,
    SyntheticRunner: runner_to_dict,
    Trial: trial_to_dict,
    Type[Transform]: transform_type_to_dict,
    ObservationFeatures: observation_features_to_dict,
}

DECODER_REGISTRY: Dict[str, Type] = {
    "AbandonedArm": AbandonedArm,
    "Arm": Arm,
    "BatchTrial": BatchTrial,
    "BenchmarkProblem": BenchmarkProblem,
    "BenchmarkResult": BenchmarkResult,
    "BraninMetric": BraninMetric,
    "ChoiceParameter": ChoiceParameter,
    "ComparisonOp": ComparisonOp,
    "Data": Data,
    "DomainType": DomainType,
    "Experiment": Experiment,
    "FactorialMetric": FactorialMetric,
    "FixedParameter": FixedParameter,
    "GenerationStrategy": GenerationStrategy,
    "GenerationStep": GenerationStep,
    "GeneratorRun": GeneratorRun,
    "GeneratorRunStruct": GeneratorRunStruct,
    "Hartmann6Metric": Hartmann6Metric,
    "L2NormMetric": L2NormMetric,
    "Metric": Metric,
    "Models": Models,
    "MultiObjective": MultiObjective,
    "NegativeBraninMetric": NegativeBraninMetric,
    "NoisyFunctionMetric": NoisyFunctionMetric,
    "Objective": Objective,
    "OptimizationConfig": OptimizationConfig,
    "OrderConstraint": OrderConstraint,
    "OutcomeConstraint": OutcomeConstraint,
    "ParameterConstraint": ParameterConstraint,
    "ParameterConstraintType": ParameterConstraintType,
    "ParameterType": ParameterType,
    "RangeParameter": RangeParameter,
    "ScalarizedObjective": ScalarizedObjective,
    "SearchSpace": SearchSpace,
    "SimpleBenchmarkProblem": SimpleBenchmarkProblem,
    "SimpleExperiment": SimpleExperiment,
    "SumConstraint": SumConstraint,
    "SyntheticRunner": SyntheticRunner,
    "Trial": Trial,
    "TrialStatus": TrialStatus,
    "Type[Transform]": Type[Transform],
    "ObservationFeatures": ObservationFeatures,
}
