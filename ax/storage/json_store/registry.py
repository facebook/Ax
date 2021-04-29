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
from ax.core.experiment import DataType, Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
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
from ax.metrics.branin import AugmentedBraninMetric, BraninMetric, NegativeBraninMetric
from ax.metrics.chemistry import ChemistryProblemType, ChemistryMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import AugmentedHartmann6Metric, Hartmann6Metric
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.metrics.sklearn import SklearnMetric, SklearnDataset, SklearnModelType
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.runners.synthetic import SyntheticRunner
from ax.storage.json_store.decoders import class_from_json, transform_type_from_json
from ax.storage.json_store.encoders import (
    arm_to_dict,
    batch_to_dict,
    benchmark_problem_to_dict,
    botorch_model_to_dict,
    botorch_modular_to_dict,
    choice_parameter_to_dict,
    data_to_dict,
    experiment_to_dict,
    fixed_parameter_to_dict,
    generation_step_to_dict,
    generation_strategy_to_dict,
    generator_run_to_dict,
    map_data_to_dict,
    metric_to_dict,
    multi_objective_optimization_config_to_dict,
    multi_objective_to_dict,
    multi_type_experiment_to_dict,
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
    surrogate_to_dict,
    transform_type_to_dict,
    trial_to_dict,
)
from ax.storage.utils import DomainType, ParameterConstraintType
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


ENCODER_REGISTRY: Dict[Type, Callable[[Any], Dict[str, Any]]] = {
    Arm: arm_to_dict,
    AugmentedBraninMetric: metric_to_dict,
    AugmentedHartmann6Metric: metric_to_dict,
    BatchTrial: batch_to_dict,
    BenchmarkProblem: benchmark_problem_to_dict,
    BoTorchModel: botorch_model_to_dict,
    BraninMetric: metric_to_dict,
    ChoiceParameter: choice_parameter_to_dict,
    Data: data_to_dict,
    Experiment: experiment_to_dict,
    FactorialMetric: metric_to_dict,
    FixedParameter: fixed_parameter_to_dict,
    GenerationStep: generation_step_to_dict,
    GenerationStrategy: generation_strategy_to_dict,
    GeneratorRun: generator_run_to_dict,
    Hartmann6Metric: metric_to_dict,
    ListSurrogate: surrogate_to_dict,
    L2NormMetric: metric_to_dict,
    MapData: map_data_to_dict,
    MapMetric: metric_to_dict,
    Metric: metric_to_dict,
    MultiObjective: multi_objective_to_dict,
    MultiObjectiveOptimizationConfig: multi_objective_optimization_config_to_dict,
    MultiTypeExperiment: multi_type_experiment_to_dict,
    SklearnMetric: metric_to_dict,
    ChemistryMetric: metric_to_dict,
    NegativeBraninMetric: metric_to_dict,
    NoisyFunctionMetric: metric_to_dict,
    Objective: objective_to_dict,
    ObjectiveThreshold: outcome_constraint_to_dict,
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
    Surrogate: surrogate_to_dict,
    SyntheticRunner: runner_to_dict,
    Trial: trial_to_dict,
    ObservationFeatures: observation_features_to_dict,
}

# Registry for class types, not instances.
# NOTE: Avoid putting a class along with its subclass in `CLASS_ENCODER_REGISTRY`.
# The encoder iterates through this dictionary and uses the first superclass that
# it finds, which might not be the intended superclass.
CLASS_ENCODER_REGISTRY: Dict[Type, Callable[[Any], Dict[str, Any]]] = {
    Acquisition: botorch_modular_to_dict,
    AcquisitionFunction: botorch_modular_to_dict,
    MarginalLogLikelihood: botorch_modular_to_dict,
    Model: botorch_modular_to_dict,
    Transform: transform_type_to_dict,
}

DECODER_REGISTRY: Dict[str, Type] = {
    "AbandonedArm": AbandonedArm,
    "AugmentedBraninMetric": AugmentedBraninMetric,
    "AugmentedHartmann6Metric": AugmentedHartmann6Metric,
    "Arm": Arm,
    "BatchTrial": BatchTrial,
    "BenchmarkProblem": BenchmarkProblem,
    "BenchmarkResult": BenchmarkResult,
    "BoTorchModel": BoTorchModel,
    "BraninMetric": BraninMetric,
    "ChemistryMetric": ChemistryMetric,
    "ChemistryProblemType": ChemistryProblemType,
    "ChoiceParameter": ChoiceParameter,
    "ComparisonOp": ComparisonOp,
    "Data": Data,
    "DataType": DataType,
    "DomainType": DomainType,
    "Experiment": Experiment,
    "FactorialMetric": FactorialMetric,
    "FixedParameter": FixedParameter,
    "GenerationStrategy": GenerationStrategy,
    "GenerationStep": GenerationStep,
    "GeneratorRun": GeneratorRun,
    "GeneratorRunStruct": GeneratorRunStruct,
    "Hartmann6Metric": Hartmann6Metric,
    "ListSurrogate": ListSurrogate,
    "L2NormMetric": L2NormMetric,
    "MapData": MapData,
    "MapMetric": MapMetric,
    "Metric": Metric,
    "Models": Models,
    "MultiObjective": MultiObjective,
    "MultiObjectiveOptimizationConfig": MultiObjectiveOptimizationConfig,
    "MultiTypeExperiment": MultiTypeExperiment,
    "NegativeBraninMetric": NegativeBraninMetric,
    "NoisyFunctionMetric": NoisyFunctionMetric,
    "Objective": Objective,
    "ObjectiveThreshold": ObjectiveThreshold,
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
    "SklearnDataset": SklearnDataset,
    "SklearnMetric": SklearnMetric,
    "SklearnModelType": SklearnModelType,
    "SumConstraint": SumConstraint,
    "Surrogate": Surrogate,
    "SyntheticRunner": SyntheticRunner,
    "Trial": Trial,
    "TrialStatus": TrialStatus,
    "ObservationFeatures": ObservationFeatures,
}


# Registry for class types, not instances.
CLASS_DECODER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "Type[Acquisition]": class_from_json,
    "Type[AcquisitionFunction]": class_from_json,
    "Type[MarginalLogLikelihood]": class_from_json,
    "Type[Model]": class_from_json,
    "Type[Transform]": transform_type_from_json,
}
