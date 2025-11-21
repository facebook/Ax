#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pathlib
from collections.abc import Callable
from typing import Any

import torch
from ax.adapter.base import DataLoaderConfig
from ax.adapter.registry import GeneratorRegistryBase, Generators
from ax.adapter.transforms.base import Transform
from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_metric import (
    BenchmarkMapMetric,
    BenchmarkMapUnavailableWhileRunningMetric,
    BenchmarkMetric,
    BenchmarkTimeVaryingMetric,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core import Experiment, ObservationFeatures
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import DataType
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.core.types import ComparisonOp
from ax.early_stopping.strategies import (
    PercentileEarlyStoppingStrategy,
    ThresholdEarlyStoppingStrategy,
)
from ax.early_stopping.strategies.logical import (
    AndEarlyStoppingStrategy,
    OrEarlyStoppingStrategy,
)
from ax.generation_strategy.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode, GenerationStep
from ax.generation_strategy.generation_node_input_constructors import (
    InputConstructorPurpose,
    NodeInputConstructors,
)
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGen,
    AuxiliaryExperimentCheck,
    IsSingleObjective,
    MaxGenerationParallelism,
    MinimumPreferenceOccurances,
    MinimumTrialsInStatus,
    MinTrials,
    TransitionCriterion,
)
from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.generators.winsorization_config import WinsorizationConfig
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.metrics.branin_map import BraninTimestampMapMetric
from ax.metrics.chemistry import ChemistryMetric, ChemistryProblemType
from ax.metrics.factorial import FactorialMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.noisy_function import NoisyFunctionMetric
from ax.metrics.sklearn import SklearnDataset, SklearnMetric, SklearnModelType
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.orchestrator_options import OrchestratorOptions, TrialType
from ax.storage.json_store.decoders import (
    choice_parameter_from_json,
    class_from_json,
    default_from_json,
    fixed_parameter_from_json,
    input_transform_type_from_json,
    multi_objective_from_json,
    observation_features_from_json,
    outcome_transform_type_from_json,
    pathlib_from_json,
    transform_type_from_json,
)
from ax.storage.json_store.encoders import (
    arm_to_dict,
    auxiliary_experiment_to_dict,
    backend_simulator_to_dict,
    batch_to_dict,
    best_model_selector_to_dict,
    botorch_component_to_dict,
    botorch_model_to_dict,
    botorch_modular_to_dict,
    choice_parameter_to_dict,
    data_to_dict,
    default_to_dict,
    derived_parameter_to_dict,
    experiment_to_dict,
    fixed_parameter_to_dict,
    generation_node_to_dict,
    generation_step_to_dict,
    generation_strategy_to_dict,
    generator_run_to_dict,
    generator_spec_to_dict,
    improvement_global_stopping_strategy_to_dict,
    logical_early_stopping_strategy_to_dict,
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
    pathlib_to_dict,
    percentile_early_stopping_strategy_to_dict,
    preference_optimization_config_to_dict,
    range_parameter_to_dict,
    runner_to_dict,
    scalarized_objective_to_dict,
    search_space_to_dict,
    sum_parameter_constraint_to_dict,
    surrogate_to_dict,
    threshold_early_stopping_strategy_to_dict,
    transform_type_to_dict,
    transition_criterion_to_dict,
    trial_to_dict,
    winsorization_config_to_dict,
)
from ax.storage.utils import DomainType, ParameterConstraintType
from ax.utils.common.constants import Keys
from ax.utils.common.serialization import TDecoderRegistry
from ax.utils.testing.backend_simulator import (
    BackendSimulator,
    BackendSimulatorOptions,
    SimTrial,
)
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputTransform,
    Normalize,
    Round,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.types import DEFAULT
from gpytorch.constraints import Interval
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior


# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
CORE_ENCODER_REGISTRY: dict[type, Callable[[Any], dict[str, Any]]] = {
    Arm: arm_to_dict,
    AuxiliaryExperiment: auxiliary_experiment_to_dict,
    AndEarlyStoppingStrategy: logical_early_stopping_strategy_to_dict,
    AutoTransitionAfterGen: transition_criterion_to_dict,
    BackendSimulator: backend_simulator_to_dict,
    BatchTrial: batch_to_dict,
    BenchmarkMetric: metric_to_dict,
    BenchmarkMapMetric: metric_to_dict,
    BenchmarkTimeVaryingMetric: metric_to_dict,
    BenchmarkMapUnavailableWhileRunningMetric: metric_to_dict,
    BoTorchGenerator: botorch_model_to_dict,
    BraninMetric: metric_to_dict,
    BraninTimestampMapMetric: metric_to_dict,
    ChainedInputTransform: botorch_component_to_dict,
    ChoiceParameter: choice_parameter_to_dict,
    Data: data_to_dict,
    DerivedParameter: derived_parameter_to_dict,
    Experiment: experiment_to_dict,
    FactorialMetric: metric_to_dict,
    FixedParameter: fixed_parameter_to_dict,
    GammaPrior: botorch_component_to_dict,
    GenerationStep: generation_step_to_dict,
    GenerationNode: generation_node_to_dict,
    GenerationStrategy: generation_strategy_to_dict,
    GeneratorRun: generator_run_to_dict,
    Hartmann6Metric: metric_to_dict,
    ImprovementGlobalStoppingStrategy: improvement_global_stopping_strategy_to_dict,
    Interval: botorch_component_to_dict,
    IsSingleObjective: transition_criterion_to_dict,
    L2NormMetric: metric_to_dict,
    LogNormalPrior: botorch_component_to_dict,
    MapData: map_data_to_dict,
    MapMetric: metric_to_dict,
    MaxGenerationParallelism: transition_criterion_to_dict,
    Metric: metric_to_dict,
    MinTrials: transition_criterion_to_dict,
    MinimumTrialsInStatus: transition_criterion_to_dict,
    MinimumPreferenceOccurances: transition_criterion_to_dict,
    AuxiliaryExperimentCheck: transition_criterion_to_dict,
    GeneratorSpec: generator_spec_to_dict,
    MultiObjective: multi_objective_to_dict,
    MultiObjectiveOptimizationConfig: multi_objective_optimization_config_to_dict,
    MultiTypeExperiment: multi_type_experiment_to_dict,
    Normalize: botorch_component_to_dict,
    FilterFeatures: botorch_component_to_dict,
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
    pathlib.Path: pathlib_to_dict,
    pathlib.PurePath: pathlib_to_dict,
    pathlib.PosixPath: pathlib_to_dict,
    pathlib.WindowsPath: pathlib_to_dict,
    pathlib.PurePosixPath: pathlib_to_dict,
    pathlib.PureWindowsPath: pathlib_to_dict,
    PreferenceOptimizationConfig: preference_optimization_config_to_dict,
    RangeParameter: range_parameter_to_dict,
    Round: botorch_component_to_dict,
    TransitionCriterion: transition_criterion_to_dict,
    ScalarizedObjective: scalarized_objective_to_dict,
    SearchSpace: search_space_to_dict,
    SingleDiagnosticBestModelSelector: best_model_selector_to_dict,
    HierarchicalSearchSpace: search_space_to_dict,
    SobolQMCNormalSampler: botorch_component_to_dict,
    SumConstraint: sum_parameter_constraint_to_dict,
    Surrogate: surrogate_to_dict,
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
# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
CORE_CLASS_ENCODER_REGISTRY: dict[type, Callable[[Any], dict[str, Any]]] = {
    Acquisition: botorch_modular_to_dict,  # Ax MBM component
    AcquisitionFunction: botorch_modular_to_dict,  # BoTorch component
    InputTransform: botorch_modular_to_dict,  # BoTorch input transform component
    Likelihood: botorch_modular_to_dict,  # BoTorch component
    torch.nn.Module: botorch_modular_to_dict,  # BoTorch module
    MarginalLogLikelihood: botorch_modular_to_dict,  # BoTorch component
    Model: botorch_modular_to_dict,  # BoTorch component
    Transform: transform_type_to_dict,  # Ax general (not just MBM) component
    DEFAULT: default_to_dict,  # BoTorch DEFAULT, used in MBM.
}

# TODO Clean up type signature. Decoders should be allowed to be any method from some
# splattable inputs to the resultant class, not just Types with kwarg inits.
CORE_DECODER_REGISTRY: TDecoderRegistry = {
    "AbandonedArm": AbandonedArm,
    "AndEarlyStoppingStrategy": AndEarlyStoppingStrategy,
    "AutoTransitionAfterGen": AutoTransitionAfterGen,
    "AuxiliaryExperiment": AuxiliaryExperiment,
    "AuxiliaryExperimentCheck": AuxiliaryExperimentCheck,
    "AuxiliaryExperimentPurpose": AuxiliaryExperimentPurpose,
    "Arm": Arm,
    "AggregatedBenchmarkResult": AggregatedBenchmarkResult,
    "BackendSimulator": BackendSimulator,
    "BackendSimulatorOptions": BackendSimulatorOptions,
    "BatchTrial": BatchTrial,
    "BenchmarkMethod": BenchmarkMethod,
    "BenchmarkMetric": BenchmarkMetric,
    "BenchmarkMapMetric": BenchmarkMapMetric,
    "BenchmarkTimeVaryingMetric": BenchmarkTimeVaryingMetric,
    "BenchmarkMapUnavailableWhileRunningMetric": (
        BenchmarkMapUnavailableWhileRunningMetric
    ),
    "BenchmarkResult": BenchmarkResult,
    "BenchmarkTrialMetadata": BenchmarkTrialMetadata,
    "BoTorchGenerator": BoTorchGenerator,
    "BoTorchModel": BoTorchGenerator,
    "BraninMetric": BraninMetric,
    "BraninTimestampMapMetric": BraninTimestampMapMetric,
    "CenterGenerationNode": CenterGenerationNode,
    "ChainedInputTransform": ChainedInputTransform,
    "ChemistryMetric": ChemistryMetric,
    "ChemistryProblemType": ChemistryProblemType,
    "ChoiceParameter": choice_parameter_from_json,
    "ComparisonOp": ComparisonOp,
    "Data": Data,
    "DataLoaderConfig": DataLoaderConfig,
    "DataType": DataType,
    "DerivedParameter": DerivedParameter,
    "DomainType": DomainType,
    "Experiment": Experiment,
    "FactorialMetric": FactorialMetric,
    "FilterFeatures": FilterFeatures,
    "FixedParameter": fixed_parameter_from_json,
    "GammaPrior": GammaPrior,
    "GenerationNode": GenerationNode,
    "GenerationStrategy": GenerationStrategy,
    "GenerationStep": GenerationStep,
    "GeneratorRun": GeneratorRun,
    "Generators": Generators,
    "GeneratorSpec": GeneratorSpec,
    "Hartmann6Metric": Hartmann6Metric,
    "HierarchicalSearchSpace": HierarchicalSearchSpace,
    "ImprovementGlobalStoppingStrategy": ImprovementGlobalStoppingStrategy,
    "InputConstructorPurpose": InputConstructorPurpose,
    "Interval": Interval,
    "IsSingleObjective": IsSingleObjective,
    "Keys": Keys,
    # DEPRECATED; remains here backward compatibility, with old class
    # name linked to the new corresponding class
    "ListSurrogate": Surrogate,
    "L2NormMetric": L2NormMetric,
    "LogNormalPrior": LogNormalPrior,
    "MapData": MapData,
    "MapMetric": MapMetric,
    "MaxTrials": MinTrials,
    "MaxGenerationParallelism": MaxGenerationParallelism,
    "Metric": Metric,
    "MinTrials": MinTrials,
    "MinimumTrialsInStatus": MinimumTrialsInStatus,
    "MinimumPreferenceOccurances": MinimumPreferenceOccurances,
    "GeneratorRegistryBase": GeneratorRegistryBase,
    "ModelRegistryBase": GeneratorRegistryBase,
    "ModelConfig": ModelConfig,
    "Models": Generators,
    "ModelSpec": GeneratorSpec,
    "MultiObjective": multi_objective_from_json,
    "MultiObjectiveOptimizationConfig": MultiObjectiveOptimizationConfig,
    "MultiTypeExperiment": MultiTypeExperiment,
    "NegativeBraninMetric": NegativeBraninMetric,
    "NodeInputConstructors": NodeInputConstructors,
    "NoisyFunctionMetric": NoisyFunctionMetric,
    "Normalize": Normalize,
    "Objective": Objective,
    "ObjectiveThreshold": ObjectiveThreshold,
    "OptimizationConfig": OptimizationConfig,
    "OrchestratorOptions": OrchestratorOptions,
    "OrEarlyStoppingStrategy": OrEarlyStoppingStrategy,
    "OrderConstraint": OrderConstraint,
    "OutcomeConstraint": OutcomeConstraint,
    "ParameterConstraint": ParameterConstraint,
    "ParameterConstraintType": ParameterConstraintType,
    "ParameterType": ParameterType,
    "Path": pathlib_from_json,
    "PurePath": pathlib_from_json,
    "PosixPath": pathlib_from_json,
    "WindowsPath": pathlib_from_json,
    "PreferenceOptimizationConfig": PreferenceOptimizationConfig,
    "PurePosixPath": pathlib_from_json,
    "PureWindowsPath": pathlib_from_json,
    "PercentileEarlyStoppingStrategy": PercentileEarlyStoppingStrategy,
    "RangeParameter": RangeParameter,
    "ReductionCriterion": ReductionCriterion,
    "Round": Round,
    "ScalarizedObjective": ScalarizedObjective,
    "SchedulerOptions": OrchestratorOptions,  # DEPRECATED; backward compatibility
    "SearchSpace": SearchSpace,
    "SimTrial": SimTrial,
    "SingleDiagnosticBestModelSelector": SingleDiagnosticBestModelSelector,
    "SklearnDataset": SklearnDataset,
    "SklearnMetric": SklearnMetric,
    "SklearnModelType": SklearnModelType,
    "SumConstraint": SumConstraint,
    "Surrogate": Surrogate,
    "SurrogateMetric": BenchmarkMetric,  # DEPRECATED; backward compatibility
    "SobolQMCNormalSampler": SobolQMCNormalSampler,
    "SyntheticRunner": SyntheticRunner,
    "SurrogateSpec": SurrogateSpec,
    "Trial": Trial,
    "TrialType": TrialType,
    "TrialStatus": TrialStatus,
    "ThresholdEarlyStoppingStrategy": ThresholdEarlyStoppingStrategy,
    "ObservationFeatures": observation_features_from_json,
    "WinsorizationConfig": WinsorizationConfig,
}


# Registry for class types, not instances.
CORE_CLASS_DECODER_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "Type[Acquisition]": class_from_json,
    "Type[AcquisitionFunction]": class_from_json,
    "Type[Kernel]": class_from_json,
    "Type[Likelihood]": class_from_json,
    "Type[torch.nn.Module]": class_from_json,
    "Type[MarginalLogLikelihood]": class_from_json,
    "Type[Model]": class_from_json,
    "Type[Transform]": transform_type_from_json,
    "Type[InputTransform]": input_transform_type_from_json,
    "Type[OutcomeTransform]": outcome_transform_type_from_json,
    "_DefaultType": default_from_json,
}
