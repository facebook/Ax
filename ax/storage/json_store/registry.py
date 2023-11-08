#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
from typing import Any, Callable, Dict, Type

import torch

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    MultiObjectiveBenchmarkProblem,
    SingleObjectiveBenchmarkProblem,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.problems.hpo.pytorch_cnn import PyTorchCNNMetric
from ax.benchmark.problems.hpo.torchvision import (
    PyTorchCNNTorchvisionBenchmarkProblem,
    PyTorchCNNTorchvisionRunner,
)
from ax.benchmark.problems.surrogate import SurrogateMetric, SurrogateRunner
from ax.core import ObservationFeatures
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ax.core.data import Data
from ax.core.experiment import DataType, Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData, MapKeyInfo
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
from ax.core.parameter_distribution import ParameterDistribution
from ax.core.risk_measures import RiskMeasure
from ax.core.search_space import HierarchicalSearchSpace, RobustSearchSpace, SearchSpace
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.early_stopping.strategies import (
    PercentileEarlyStoppingStrategy,
    ThresholdEarlyStoppingStrategy,
)
from ax.early_stopping.strategies.logical import (
    AndEarlyStoppingStrategy,
    OrEarlyStoppingStrategy,
)
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.metrics.botorch_test_problem import BotorchTestProblemMetric
from ax.metrics.branin import AugmentedBraninMetric, BraninMetric, NegativeBraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.chemistry import ChemistryMetric, ChemistryProblemType
from ax.metrics.dict_lookup import DictLookupMetric
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import AugmentedHartmann6Metric, Hartmann6Metric
from ax.metrics.jenatton import JenattonMetric
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.metrics.sklearn import SklearnDataset, SklearnMetric, SklearnModelType
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transition_criterion import (
    MaxGenerationParallelism,
    MaxTrials,
    MinimumPreferenceOccurances,
    MinimumTrialsInStatus,
    MinTrials,
)
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.model import BoTorchModel, SurrogateSpec
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.winsorization_config import WinsorizationConfig
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.storage.json_store.decoders import (
    class_from_json,
    input_transform_type_from_json,
    outcome_transform_type_from_json,
    pathlib_from_json,
    transform_type_from_json,
)
from ax.storage.json_store.encoders import (
    arm_to_dict,
    batch_to_dict,
    benchmark_problem_to_dict,
    botorch_component_to_dict,
    botorch_model_to_dict,
    botorch_modular_to_dict,
    choice_parameter_to_dict,
    data_to_dict,
    experiment_to_dict,
    fixed_parameter_to_dict,
    generation_step_to_dict,
    generation_strategy_to_dict,
    generator_run_to_dict,
    improvement_global_stopping_strategy_to_dict,
    logical_early_stopping_strategy_to_dict,
    map_data_to_dict,
    map_key_info_to_dict,
    metric_to_dict,
    multi_objective_benchmark_problem_to_dict,
    multi_objective_optimization_config_to_dict,
    multi_objective_to_dict,
    multi_type_experiment_to_dict,
    objective_to_dict,
    observation_features_to_dict,
    optimization_config_to_dict,
    order_parameter_constraint_to_dict,
    outcome_constraint_to_dict,
    parameter_constraint_to_dict,
    parameter_distribution_to_dict,
    pathlib_to_dict,
    percentile_early_stopping_strategy_to_dict,
    pytorch_cnn_torchvision_benchmark_problem_to_dict,
    range_parameter_to_dict,
    risk_measure_to_dict,
    robust_search_space_to_dict,
    runner_to_dict,
    scalarized_objective_to_dict,
    search_space_to_dict,
    single_objective_benchmark_problem_to_dict,
    sum_parameter_constraint_to_dict,
    surrogate_to_dict,
    threshold_early_stopping_strategy_to_dict,
    transform_type_to_dict,
    transition_criterion_to_dict,
    trial_to_dict,
    winsorization_config_to_dict,
)
from ax.storage.utils import DomainType, ParameterConstraintType
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Round
from gpytorch.constraints import Interval
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior


# pyre-fixme[5]: Global annotation cannot contain `Any`.
# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
CORE_ENCODER_REGISTRY: Dict[Type, Callable[[Any], Dict[str, Any]]] = {
    Arm: arm_to_dict,
    AndEarlyStoppingStrategy: logical_early_stopping_strategy_to_dict,
    AugmentedBraninMetric: metric_to_dict,
    AugmentedHartmann6Metric: metric_to_dict,
    BatchTrial: batch_to_dict,
    BenchmarkProblem: benchmark_problem_to_dict,
    BoTorchModel: botorch_model_to_dict,
    BotorchTestProblemMetric: metric_to_dict,
    BotorchTestProblemRunner: runner_to_dict,
    BraninMetric: metric_to_dict,
    BraninTimestampMapMetric: metric_to_dict,
    ChainedInputTransform: botorch_component_to_dict,
    ChoiceParameter: choice_parameter_to_dict,
    Data: data_to_dict,
    DictLookupMetric: metric_to_dict,
    Experiment: experiment_to_dict,
    FactorialMetric: metric_to_dict,
    FixedParameter: fixed_parameter_to_dict,
    GammaPrior: botorch_component_to_dict,
    GenerationStep: generation_step_to_dict,
    GenerationStrategy: generation_strategy_to_dict,
    GeneratorRun: generator_run_to_dict,
    Hartmann6Metric: metric_to_dict,
    ImprovementGlobalStoppingStrategy: improvement_global_stopping_strategy_to_dict,
    Interval: botorch_component_to_dict,
    JenattonMetric: metric_to_dict,
    L2NormMetric: metric_to_dict,
    MapData: map_data_to_dict,
    MapKeyInfo: map_key_info_to_dict,
    MapMetric: metric_to_dict,
    MaxGenerationParallelism: transition_criterion_to_dict,
    MaxTrials: transition_criterion_to_dict,
    Metric: metric_to_dict,
    MinTrials: transition_criterion_to_dict,
    MinimumTrialsInStatus: transition_criterion_to_dict,
    MinimumPreferenceOccurances: transition_criterion_to_dict,
    MultiObjective: multi_objective_to_dict,
    MultiObjectiveBenchmarkProblem: multi_objective_benchmark_problem_to_dict,
    MultiObjectiveOptimizationConfig: multi_objective_optimization_config_to_dict,
    MultiTypeExperiment: multi_type_experiment_to_dict,
    Normalize: botorch_component_to_dict,
    PercentileEarlyStoppingStrategy: percentile_early_stopping_strategy_to_dict,
    SklearnMetric: metric_to_dict,
    ChemistryMetric: metric_to_dict,
    NegativeBraninMetric: metric_to_dict,
    NoisyFunctionMetric: metric_to_dict,
    Objective: objective_to_dict,
    ObjectiveThreshold: outcome_constraint_to_dict,
    OptimizationConfig: optimization_config_to_dict,
    OrEarlyStoppingStrategy: logical_early_stopping_strategy_to_dict,
    OrderConstraint: order_parameter_constraint_to_dict,
    OutcomeConstraint: outcome_constraint_to_dict,
    ParameterConstraint: parameter_constraint_to_dict,
    ParameterDistribution: parameter_distribution_to_dict,
    pathlib.Path: pathlib_to_dict,
    pathlib.PurePath: pathlib_to_dict,
    pathlib.PosixPath: pathlib_to_dict,
    pathlib.WindowsPath: pathlib_to_dict,
    pathlib.PurePosixPath: pathlib_to_dict,
    pathlib.PureWindowsPath: pathlib_to_dict,
    PyTorchCNNTorchvisionBenchmarkProblem: pytorch_cnn_torchvision_benchmark_problem_to_dict,  # noqa
    PyTorchCNNMetric: metric_to_dict,
    PyTorchCNNTorchvisionRunner: runner_to_dict,
    RangeParameter: range_parameter_to_dict,
    RiskMeasure: risk_measure_to_dict,
    RobustSearchSpace: robust_search_space_to_dict,
    Round: botorch_component_to_dict,
    ScalarizedObjective: scalarized_objective_to_dict,
    SearchSpace: search_space_to_dict,
    SingleObjectiveBenchmarkProblem: single_objective_benchmark_problem_to_dict,
    HierarchicalSearchSpace: search_space_to_dict,
    SumConstraint: sum_parameter_constraint_to_dict,
    Surrogate: surrogate_to_dict,
    SurrogateMetric: metric_to_dict,
    SurrogateRunner: runner_to_dict,
    SyntheticRunner: runner_to_dict,
    ThresholdEarlyStoppingStrategy: threshold_early_stopping_strategy_to_dict,
    Trial: trial_to_dict,
    ObservationFeatures: observation_features_to_dict,
    WinsorizationConfig: winsorization_config_to_dict,
}

# Registry for class types, not instances.
# NOTE: Avoid putting a class along with its subclass in `CLASS_ENCODER_REGISTRY`.
# The encoder iterates through this dictionary and uses the first superclass that
# it finds, which might not be the intended superclass.
# pyre-fixme[5]: Global annotation cannot contain `Any`.
# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
CORE_CLASS_ENCODER_REGISTRY: Dict[Type, Callable[[Any], Dict[str, Any]]] = {
    Acquisition: botorch_modular_to_dict,  # Ax MBM component
    AcquisitionFunction: botorch_modular_to_dict,  # BoTorch component
    Likelihood: botorch_modular_to_dict,  # BoTorch component
    torch.nn.Module: botorch_modular_to_dict,  # BoTorch module
    MarginalLogLikelihood: botorch_modular_to_dict,  # BoTorch component
    Model: botorch_modular_to_dict,  # BoTorch component
    Transform: transform_type_to_dict,  # Ax general (not just MBM) component
}

# TODO Clean up type signature. Decoders should be allowed to be any method from some
# splattable inputs to the resultant class, not just Types with kwarg inits.
# pyre-fixme[9, 24]
CORE_DECODER_REGISTRY: Dict[str, Type] = {
    "AbandonedArm": AbandonedArm,
    "AndEarlyStoppingStrategy": AndEarlyStoppingStrategy,
    "AugmentedBraninMetric": AugmentedBraninMetric,
    "AugmentedHartmann6Metric": AugmentedHartmann6Metric,
    "Arm": Arm,
    "BatchTrial": BatchTrial,
    "AggregatedBenchmarkResult": AggregatedBenchmarkResult,
    "BenchmarkMethod": BenchmarkMethod,
    "BenchmarkProblem": BenchmarkProblem,
    "BenchmarkResult": BenchmarkResult,
    "BoTorchModel": BoTorchModel,
    "BotorchTestProblemMetric": BotorchTestProblemMetric,
    "BotorchTestProblemRunner": BotorchTestProblemRunner,
    "BraninMetric": BraninMetric,
    "BraninTimestampMapMetric": BraninTimestampMapMetric,
    "ChainedInputTransform": ChainedInputTransform,
    "ChemistryMetric": ChemistryMetric,
    "ChemistryProblemType": ChemistryProblemType,
    "ChoiceParameter": ChoiceParameter,
    "ComparisonOp": ComparisonOp,
    "Data": Data,
    "DataType": DataType,
    "DictLookupMetric": DictLookupMetric,
    "DomainType": DomainType,
    "Experiment": Experiment,
    "FactorialMetric": FactorialMetric,
    "FixedParameter": FixedParameter,
    "GammaPrior": GammaPrior,
    "GenerationStrategy": GenerationStrategy,
    "GenerationStep": GenerationStep,
    "GeneratorRun": GeneratorRun,
    "GeneratorRunStruct": GeneratorRunStruct,
    "Hartmann6Metric": Hartmann6Metric,
    "HierarchicalSearchSpace": HierarchicalSearchSpace,
    "ImprovementGlobalStoppingStrategy": ImprovementGlobalStoppingStrategy,
    "Interval": Interval,
    "JenattonMetric": JenattonMetric,
    "ListSurrogate": Surrogate,  # For backwards compatibility
    "L2NormMetric": L2NormMetric,
    "MapData": MapData,
    "MapMetric": MapMetric,
    "MapKeyInfo": MapKeyInfo,
    "MaxTrials": MaxTrials,
    "Metric": Metric,
    "MinTrials": MinTrials,
    "MinimumTrialsInStatus": MinimumTrialsInStatus,
    "MinimumPreferenceOccurances": MinimumPreferenceOccurances,
    "Models": Models,
    "MultiObjective": MultiObjective,
    "MultiObjectiveBenchmarkProblem": MultiObjectiveBenchmarkProblem,
    "MultiObjectiveOptimizationConfig": MultiObjectiveOptimizationConfig,
    "MultiTypeExperiment": MultiTypeExperiment,
    "NegativeBraninMetric": NegativeBraninMetric,
    "NoisyFunctionMetric": NoisyFunctionMetric,
    "Normalize": Normalize,
    "Objective": Objective,
    "ObjectiveThreshold": ObjectiveThreshold,
    "OptimizationConfig": OptimizationConfig,
    "OrEarlyStoppingStrategy": OrEarlyStoppingStrategy,
    "OrderConstraint": OrderConstraint,
    "OutcomeConstraint": OutcomeConstraint,
    "ParameterConstraint": ParameterConstraint,
    "ParameterConstraintType": ParameterConstraintType,
    "ParameterDistribution": ParameterDistribution,
    "ParameterType": ParameterType,
    "Path": pathlib_from_json,
    "PurePath": pathlib_from_json,
    "PosixPath": pathlib_from_json,
    "WindowsPath": pathlib_from_json,
    "PurePosixPath": pathlib_from_json,
    "PureWindowsPath": pathlib_from_json,
    "PercentileEarlyStoppingStrategy": PercentileEarlyStoppingStrategy,
    "PyTorchCNNTorchvisionBenchmarkProblem": PyTorchCNNTorchvisionBenchmarkProblem,
    "PyTorchCNNMetric": PyTorchCNNMetric,
    "PyTorchCNNTorchvisionRunner": PyTorchCNNTorchvisionRunner,
    "RangeParameter": RangeParameter,
    "RiskMeasure": RiskMeasure,
    "RobustSearchSpace": RobustSearchSpace,
    "Round": Round,
    "ScalarizedObjective": ScalarizedObjective,
    "SchedulerOptions": SchedulerOptions,
    "SearchSpace": SearchSpace,
    "SingleObjectiveBenchmarkProblem": SingleObjectiveBenchmarkProblem,
    "SklearnDataset": SklearnDataset,
    "SklearnMetric": SklearnMetric,
    "SklearnModelType": SklearnModelType,
    "SumConstraint": SumConstraint,
    "Surrogate": Surrogate,
    "SurrogateMetric": SurrogateMetric,
    # NOTE: SurrogateRunners -> SyntheticRunner on load due to complications
    "SurrogateRunner": SyntheticRunner,
    "SyntheticRunner": SyntheticRunner,
    "SurrogateSpec": SurrogateSpec,
    "Trial": Trial,
    "TrialType": TrialType,
    "TrialStatus": TrialStatus,
    "ThresholdEarlyStoppingStrategy": ThresholdEarlyStoppingStrategy,
    "ObservationFeatures": ObservationFeatures,
    "WinsorizationConfig": WinsorizationConfig,
}


# Registry for class types, not instances.
# pyre-fixme[5]: Global annotation cannot contain `Any`.
CORE_CLASS_DECODER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "Type[Acquisition]": class_from_json,
    "Type[AcquisitionFunction]": class_from_json,
    "Type[Likelihood]": class_from_json,
    "Type[torch.nn.Module]": class_from_json,
    "Type[MarginalLogLikelihood]": class_from_json,
    "Type[Model]": class_from_json,
    "Type[Transform]": transform_type_from_json,
    "Type[InputTransform]": input_transform_type_from_json,
    "Type[OutcomeTransform]": outcome_transform_type_from_json,
}
