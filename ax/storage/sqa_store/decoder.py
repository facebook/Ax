#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from ax.core.arm import Arm
from ax.core.base import Base
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, FixedParameter, Parameter, RangeParameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.exceptions.storage import SQADecodeError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.json_store.decoder import object_from_json
from ax.storage.metric_registry import REVERSE_METRIC_REGISTRY
from ax.storage.runner_registry import REVERSE_RUNNER_REGISTRY
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
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.utils import DomainType, MetricIntent, ParameterConstraintType
from ax.utils.common.typeutils import not_none


class Decoder:
    """Class that contains methods for loading an Ax experiment from SQLAlchemy.

    Instantiate with an instance of Config to customize the functionality.
    For even more flexibility, create a subclass.

    Attributes:
        config: Metadata needed to save and load an experiment to SQLAlchemy.
    """

    def __init__(self, config: SQAConfig) -> None:
        self.config = config

    def get_enum_name(
        self, value: Optional[int], enum: Optional[Enum]
    ) -> Optional[str]:
        """Given an enum value (int) and an enum (of ints), return the
        corresponding enum name. If the value is not present in the enum,
        throw an error.
        """
        if value is None or enum is None:
            return None

        try:
            return enum(value).name  # pyre-ignore T29651755
        except ValueError:
            raise SQADecodeError(f"Value {value} is invalid for enum {enum}.")

    def _init_experiment_from_sqa(self, experiment_sqa: SQAExperiment) -> Experiment:
        """First step of conversion within experiment_from_sqa."""
        opt_config, tracking_metrics = self.opt_config_and_tracking_metrics_from_sqa(
            metrics_sqa=experiment_sqa.metrics
        )
        search_space = self.search_space_from_sqa(
            parameters_sqa=experiment_sqa.parameters,
            parameter_constraints_sqa=experiment_sqa.parameter_constraints,
        )
        if search_space is None:
            raise SQADecodeError(  # pragma: no cover
                "Experiment SearchSpace cannot be None."
            )
        status_quo = (
            Arm(
                # pyre-fixme[6]: Expected `Dict[str, Optional[Union[bool, float,
                #  int, str]]]` for 1st param but got `Optional[Dict[str,
                #  Optional[Union[bool, float, int, str]]]]`.
                parameters=experiment_sqa.status_quo_parameters,
                name=experiment_sqa.status_quo_name,
            )
            if experiment_sqa.status_quo_parameters is not None
            else None
        )
        if len(experiment_sqa.runners) == 0:
            runner = None
        elif len(experiment_sqa.runners) == 1:
            runner = self.runner_from_sqa(experiment_sqa.runners[0])
        else:
            raise ValueError(  # pragma: no cover
                "Multiple runners on experiment "
                "only supported for MultiTypeExperiment."
            )

        subclass = (experiment_sqa.properties or {}).get("subclass")
        if subclass == "SimpleExperiment":
            if opt_config is None:
                raise SQADecodeError(  # pragma: no cover
                    "SimpleExperiment must have an optimization config."
                )
            experiment = SimpleExperiment(
                name=experiment_sqa.name,
                search_space=search_space,
                objective_name=opt_config.objective.metric.name,
                minimize=opt_config.objective.minimize,
                outcome_constraints=opt_config.outcome_constraints,
                status_quo=status_quo,
            )
            experiment.description = experiment_sqa.description
            experiment.is_test = experiment_sqa.is_test
        else:
            experiment = Experiment(
                name=experiment_sqa.name,
                description=experiment_sqa.description,
                search_space=search_space,
                optimization_config=opt_config,
                tracking_metrics=tracking_metrics,
                runner=runner,
                status_quo=status_quo,
                is_test=experiment_sqa.is_test,
            )
        return experiment

    def _init_mt_experiment_from_sqa(
        self, experiment_sqa: SQAExperiment
    ) -> MultiTypeExperiment:
        """First step of conversion within experiment_from_sqa."""
        opt_config, tracking_metrics = self.opt_config_and_tracking_metrics_from_sqa(
            metrics_sqa=experiment_sqa.metrics
        )
        search_space = self.search_space_from_sqa(
            parameters_sqa=experiment_sqa.parameters,
            parameter_constraints_sqa=experiment_sqa.parameter_constraints,
        )
        if search_space is None:
            raise SQADecodeError(  # pragma: no cover
                "Experiment SearchSpace cannot be None."
            )
        status_quo = (
            Arm(
                # pyre-fixme[6]: Expected `Dict[str, Optional[Union[bool, float,
                #  int, str]]]` for 1st param but got `Optional[Dict[str,
                #  Optional[Union[bool, float, int, str]]]]`.
                parameters=experiment_sqa.status_quo_parameters,
                name=experiment_sqa.status_quo_name,
            )
            if experiment_sqa.status_quo_parameters is not None
            else None
        )
        trial_type_to_runner = {
            not_none(sqa_runner.trial_type): self.runner_from_sqa(sqa_runner)
            for sqa_runner in experiment_sqa.runners
        }
        default_trial_type = not_none(experiment_sqa.default_trial_type)
        experiment = MultiTypeExperiment(
            name=experiment_sqa.name,
            search_space=search_space,
            default_trial_type=default_trial_type,
            default_runner=trial_type_to_runner[default_trial_type],
            optimization_config=opt_config,
            status_quo=status_quo,
        )
        experiment._trial_type_to_runner = trial_type_to_runner
        sqa_metric_dict = {metric.name: metric for metric in experiment_sqa.metrics}
        for tracking_metric in tracking_metrics:
            sqa_metric = sqa_metric_dict[tracking_metric.name]
            experiment.add_tracking_metric(
                tracking_metric,
                trial_type=not_none(sqa_metric.trial_type),
                canonical_name=sqa_metric.canonical_name,
            )
        return experiment

    def experiment_from_sqa(self, experiment_sqa: SQAExperiment) -> Experiment:
        """Convert SQLAlchemy Experiment to Ax Experiment."""
        subclass = (experiment_sqa.properties or {}).get("subclass")
        if subclass == "MultiTypeExperiment":
            experiment = self._init_mt_experiment_from_sqa(experiment_sqa)
        else:
            experiment = self._init_experiment_from_sqa(experiment_sqa)
        trials = [
            self.trial_from_sqa(trial_sqa=trial, experiment=experiment)
            for trial in experiment_sqa.trials
        ]

        data_by_trial = defaultdict(dict)
        for data_sqa in experiment_sqa.data:
            trial_index = data_sqa.trial_index
            timestamp = data_sqa.time_created
            data_by_trial[trial_index][timestamp] = self.data_from_sqa(
                data_sqa=data_sqa
            )
        data_by_trial = {
            trial_index: OrderedDict(sorted(data_by_timestamp.items()))
            for trial_index, data_by_timestamp in data_by_trial.items()
        }

        experiment._trials = {trial.index: trial for trial in trials}
        experiment._arms_by_name = {}
        for trial in trials:
            if trial.ttl_seconds is not None:
                experiment._trials_have_ttl = True
            for arm in trial.arms:
                experiment._register_arm(arm)
        if experiment.status_quo is not None:
            sq = not_none(experiment.status_quo)
            experiment._register_arm(sq)
        experiment._time_created = experiment_sqa.time_created
        experiment._experiment_type = self.get_enum_name(
            value=experiment_sqa.experiment_type, enum=self.config.experiment_type_enum
        )
        experiment._data_by_trial = dict(data_by_trial)

        return experiment

    def parameter_from_sqa(self, parameter_sqa: SQAParameter) -> Parameter:
        """Convert SQLAlchemy Parameter to Ax Parameter."""
        if parameter_sqa.domain_type == DomainType.RANGE:
            if parameter_sqa.lower is None or parameter_sqa.upper is None:
                raise SQADecodeError(  # pragma: no cover
                    "`lower` and `upper` must be set for RangeParameter."
                )
            return RangeParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                # pyre-fixme[6]: Expected `float` for 3rd param but got
                #  `Optional[float]`.
                lower=parameter_sqa.lower,
                upper=parameter_sqa.upper,
                log_scale=parameter_sqa.log_scale or False,
                digits=parameter_sqa.digits,
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
            )
        elif parameter_sqa.domain_type == DomainType.CHOICE:
            if parameter_sqa.choice_values is None:
                raise SQADecodeError(  # pragma: no cover
                    "`values` must be set for ChoiceParameter."
                )
            return ChoiceParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                # pyre-fixme[6]: Expected `List[Optional[Union[bool, float, int,
                #  str]]]` for 3rd param but got `Optional[List[Optional[Union[bool,
                #  float, int, str]]]]`.
                values=parameter_sqa.choice_values,
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
            )
        elif parameter_sqa.domain_type == DomainType.FIXED:
            # Don't throw an error if parameter_sqa.fixed_value is None;
            # that might be the actual value!
            return FixedParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                value=parameter_sqa.fixed_value,
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAParameter because {parameter_sqa.domain_type} "
                "is an invalid domain type."
            )

    def parameter_constraint_from_sqa(
        self,
        parameter_constraint_sqa: SQAParameterConstraint,
        parameters: List[Parameter],
    ) -> ParameterConstraint:
        """Convert SQLAlchemy ParameterConstraint to Ax ParameterConstraint."""
        parameter_map = {p.name: p for p in parameters}
        if parameter_constraint_sqa.type == ParameterConstraintType.ORDER:
            lower_name = None
            upper_name = None
            for k, v in parameter_constraint_sqa.constraint_dict.items():
                if v == 1:
                    lower_name = k
                elif v == -1:
                    upper_name = k
            if not lower_name or not upper_name:
                raise SQADecodeError(
                    "Cannot decode SQAParameterConstraint because `lower_name` or "
                    "`upper_name` was not found."
                )
            # pyre-fixme[6]: Expected `str` for 1st param but got `None`.
            lower_parameter = parameter_map[lower_name]
            # pyre-fixme[6]: Expected `str` for 1st param but got `None`.
            upper_parameter = parameter_map[upper_name]
            return OrderConstraint(
                lower_parameter=lower_parameter, upper_parameter=upper_parameter
            )
        elif parameter_constraint_sqa.type == ParameterConstraintType.SUM:
            # This operation is potentially very inefficient.
            # It is O(#constrained_parameters * #total_parameters)
            parameter_names = list(parameter_constraint_sqa.constraint_dict.keys())
            constraint_parameters = [
                next(
                    search_space_param
                    for search_space_param in parameters
                    if search_space_param.name == c_p_name
                )
                for c_p_name in parameter_names
            ]
            a_values = list(parameter_constraint_sqa.constraint_dict.values())
            if len(a_values) == 0:
                raise SQADecodeError(
                    "Cannot decode SQAParameterConstraint because `constraint_dict` "
                    "is empty."
                )
            a = a_values[0]
            is_upper_bound = a == 1
            bound = parameter_constraint_sqa.bound * a
            return SumConstraint(
                parameters=constraint_parameters,
                is_upper_bound=is_upper_bound,
                bound=bound,
            )
        else:
            return ParameterConstraint(
                constraint_dict=dict(parameter_constraint_sqa.constraint_dict),
                bound=parameter_constraint_sqa.bound,
            )

    def search_space_from_sqa(
        self,
        parameters_sqa: List[SQAParameter],
        parameter_constraints_sqa: List[SQAParameterConstraint],
    ) -> Optional[SearchSpace]:
        """Convert a list of SQLAlchemy Parameters and ParameterConstraints to an
        Ax SearchSpace.
        """
        parameters = [
            self.parameter_from_sqa(parameter_sqa=parameter_sqa)
            for parameter_sqa in parameters_sqa
        ]
        parameter_constraints = [
            self.parameter_constraint_from_sqa(
                parameter_constraint_sqa=parameter_constraint_sqa, parameters=parameters
            )
            for parameter_constraint_sqa in parameter_constraints_sqa
        ]

        if len(parameters) == 0:
            return None

        return SearchSpace(
            parameters=parameters, parameter_constraints=parameter_constraints
        )

    def get_init_args_from_properties(
        self, object_sqa: SQABase, class_: Base
    ) -> Dict[str, Any]:
        """Given a SQAAlchemy instance with a properties blob, extract the
        arguments required for its class's initializer.
        """
        args = dict(getattr(object_sqa, "properties", None) or {})
        signature = inspect.signature(class_.__init__)
        exclude_args = ["self", "args", "kwargs"]
        for arg, info in signature.parameters.items():
            if arg in exclude_args or arg in args:
                continue
            value = getattr(object_sqa, arg, None)
            if value is None:
                # Only necessary to raise an exception if there is no default
                # value for this argument
                if info.default is inspect.Parameter.empty:
                    raise SQADecodeError(
                        f"Cannot decode because required argument {arg} is missing."
                    )
                else:
                    # Constructor will use default value
                    continue  # pragma: no cover
            args[arg] = value
        return args

    def metric_from_sqa_util(self, metric_sqa: SQAMetric) -> Metric:
        """Convert SQLAlchemy Metric to Ax Metric"""
        metric_class = REVERSE_METRIC_REGISTRY.get(metric_sqa.metric_type)
        if metric_class is None:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.metric_type} "
                f"is an invalid type."
            )
        args = self.get_init_args_from_properties(
            # pyre-fixme[6]: Expected `SQABase` for ...es` but got `SQAMetric`.
            object_sqa=metric_sqa,
            class_=metric_class,
        )
        metric = metric_class(**args)
        return metric

    def metric_from_sqa(
        self, metric_sqa: SQAMetric
    ) -> Union[Metric, Objective, OutcomeConstraint]:
        """Convert SQLAlchemy Metric to Ax Metric, Objective, or OutcomeConstraint."""

        metric = self.metric_from_sqa_util(metric_sqa)

        if metric_sqa.intent == MetricIntent.TRACKING:
            return metric
        elif metric_sqa.intent == MetricIntent.OBJECTIVE:
            if metric_sqa.minimize is None:
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to Objective because minimize is None."
                )
            if metric_sqa.scalarized_objective_weight is not None:
                raise SQADecodeError(  # pragma: no cover
                    "The metric corresponding to regular objective does not \
                    have weight attribute"
                )
            return Objective(metric=metric, minimize=metric_sqa.minimize)
        elif (
            metric_sqa.intent == MetricIntent.MULTI_OBJECTIVE
        ):  # metric_sqa is a parent whose children are individual
            # metrics in MultiObjective
            if metric_sqa.minimize is None:
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to MultiObjective \
                    because minimize is None."
                )
            metrics_sqa_children = metric_sqa.scalarized_objective_children_metrics
            if metrics_sqa_children is None:
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to MultiObjective \
                    because the parent metric has no children metrics."
                )

            # Extracting metric and weight for each child
            metrics = [
                self.metric_from_sqa_util(child) for child in metrics_sqa_children
            ]

            return MultiObjective(
                metrics=list(metrics),
                # pyre-fixme[6]: Expected `bool` for 2nd param but got `Optional[bool]`.
                minimize=metric_sqa.minimize,
            )
        elif (
            metric_sqa.intent == MetricIntent.SCALARIZED_OBJECTIVE
        ):  # metric_sqa is a parent whose children are individual
            # metrics in Scalarized Objective
            if metric_sqa.minimize is None:
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to Scalarized Objective \
                    because minimize is None."
                )
            metrics_sqa_children = metric_sqa.scalarized_objective_children_metrics
            if metrics_sqa_children is None:
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to Scalarized Objective \
                    because the parent metric has no children metrics."
                )

            # Extracting metric and weight for each child
            metrics, weights = zip(
                *[
                    (
                        self.metric_from_sqa_util(child),
                        child.scalarized_objective_weight,
                    )
                    for child in metrics_sqa_children
                ]
            )
            return ScalarizedObjective(
                metrics=list(metrics),
                weights=list(weights),
                # pyre-fixme[6]: Expected `bool` for 3nd param but got `Optional[bool]`.
                minimize=metric_sqa.minimize,
            )
        elif metric_sqa.intent == MetricIntent.OUTCOME_CONSTRAINT:
            if (
                metric_sqa.bound is None
                or metric_sqa.op is None
                or metric_sqa.relative is None
            ):
                raise SQADecodeError(  # pragma: no cover
                    "Cannot decode SQAMetric to OutcomeConstraint because "
                    "bound, op, or relative is None."
                )
            return OutcomeConstraint(
                metric=metric,
                # pyre-fixme[6]: Expected `float` for 2nd param but got
                #  `Optional[float]`.
                bound=metric_sqa.bound,
                op=metric_sqa.op,
                relative=metric_sqa.relative,
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.intent} "
                f"is an invalid intent."
            )

    def opt_config_and_tracking_metrics_from_sqa(
        self, metrics_sqa: List[SQAMetric]
    ) -> Tuple[Optional[OptimizationConfig], List[Metric]]:
        """Convert a list of SQLAlchemy Metrics to a a tuple of Ax OptimizationConfig
        and tracking metrics.
        """
        objective = None
        outcome_constraints = []
        tracking_metrics = []
        for metric_sqa in metrics_sqa:
            metric = self.metric_from_sqa(metric_sqa=metric_sqa)
            if isinstance(metric, Objective):
                objective = metric
            elif isinstance(metric, OutcomeConstraint):
                outcome_constraints.append(metric)
            else:
                tracking_metrics.append(metric)

        if objective is None:
            return None, tracking_metrics

        return (
            OptimizationConfig(
                objective=objective, outcome_constraints=outcome_constraints
            ),
            tracking_metrics,
        )

    def arm_from_sqa(self, arm_sqa: SQAArm) -> Arm:
        """Convert SQLAlchemy Arm to Ax Arm."""
        return Arm(parameters=arm_sqa.parameters, name=arm_sqa.name)

    def abandoned_arm_from_sqa(
        self, abandoned_arm_sqa: SQAAbandonedArm
    ) -> AbandonedArm:
        """Convert SQLAlchemy AbandonedArm to Ax AbandonedArm."""
        return AbandonedArm(
            name=abandoned_arm_sqa.name,
            reason=abandoned_arm_sqa.abandoned_reason,
            time=abandoned_arm_sqa.time_abandoned,
        )

    def generator_run_from_sqa(
        self, generator_run_sqa: SQAGeneratorRun
    ) -> GeneratorRun:
        """Convert SQLAlchemy GeneratorRun to Ax GeneratorRun."""
        arms = []
        weights = []
        opt_config = None
        search_space = None

        for arm_sqa in generator_run_sqa.arms:
            arms.append(self.arm_from_sqa(arm_sqa=arm_sqa))
            weights.append(arm_sqa.weight)

        opt_config, tracking_metrics = self.opt_config_and_tracking_metrics_from_sqa(
            metrics_sqa=generator_run_sqa.metrics
        )
        if len(tracking_metrics) > 0:
            raise SQADecodeError(  # pragma: no cover
                "GeneratorRun should not have tracking metrics."
            )

        search_space = self.search_space_from_sqa(
            parameters_sqa=generator_run_sqa.parameters,
            parameter_constraints_sqa=generator_run_sqa.parameter_constraints,
        )

        best_arm_predictions = None
        model_predictions = None
        if (
            generator_run_sqa.best_arm_parameters is not None
            and generator_run_sqa.best_arm_predictions is not None
        ):
            best_arm = Arm(
                name=generator_run_sqa.best_arm_name,
                # pyre-fixme[6]: Expected `Dict[str, Optional[Union[bool, float,
                #  int, str]]]` for 2nd param but got `Optional[Dict[str,
                #  Optional[Union[bool, float, int, str]]]]`.
                parameters=generator_run_sqa.best_arm_parameters,
            )
            best_arm_predictions = (
                best_arm,
                # pyre-fixme[6]: Expected `Iterable[_T_co]` for 1st param but got
                #  `Optional[Tuple[Dict[str, float], Optional[Dict[str, Dict[str,
                #  float]]]]]`.
                tuple(generator_run_sqa.best_arm_predictions),
            )
        model_predictions = (
            # pyre-fixme[6]: Expected `Iterable[_T_co]` for 1st param but got
            #  `Optional[Tuple[Dict[str, List[float]], Dict[str, Dict[str,
            #  List[float]]]]]`.
            tuple(generator_run_sqa.model_predictions)
            if generator_run_sqa.model_predictions is not None
            else None
        )

        generator_run = GeneratorRun(
            arms=arms,
            weights=weights,
            optimization_config=opt_config,
            search_space=search_space,
            fit_time=generator_run_sqa.fit_time,
            gen_time=generator_run_sqa.gen_time,
            # pyre-fixme[6]: Expected `Optional[Tuple[Arm, Optional[Tuple[Dict[str,
            #  float], Optional[Dict[str, Dict[str, float]]]]]]]` for 7th param but got
            #  `Optional[Tuple[Arm, Tuple[Any, ...]]]`.
            best_arm_predictions=best_arm_predictions,
            model_predictions=model_predictions,
            model_key=generator_run_sqa.model_key,
            model_kwargs=object_from_json(generator_run_sqa.model_kwargs),
            bridge_kwargs=object_from_json(generator_run_sqa.bridge_kwargs),
            gen_metadata=object_from_json(generator_run_sqa.gen_metadata),
            model_state_after_gen=object_from_json(
                generator_run_sqa.model_state_after_gen
            ),
            generation_step_index=generator_run_sqa.generation_step_index,
            candidate_metadata_by_arm_signature=object_from_json(
                generator_run_sqa.candidate_metadata_by_arm_signature
            ),
        )
        generator_run._time_created = generator_run_sqa.time_created
        generator_run._generator_run_type = self.get_enum_name(
            value=generator_run_sqa.generator_run_type,
            enum=self.config.generator_run_type_enum,
        )
        generator_run._index = generator_run_sqa.index
        return generator_run

    def generation_strategy_from_sqa(
        self, gs_sqa: SQAGenerationStrategy
    ) -> GenerationStrategy:
        """Convert SQALchemy generation strategy to Ax `GenerationStrategy`."""
        steps = object_from_json(gs_sqa.steps)
        gs = GenerationStrategy(name=gs_sqa.name, steps=steps)
        gs._curr = gs._steps[gs_sqa.curr_index]
        gs._generator_runs = [
            self.generator_run_from_sqa(gr) for gr in gs_sqa.generator_runs
        ]
        if len(gs._generator_runs) > 0:
            # Generation strategy had an initialized model.
            # pyre-ignore[16]: SQAGenerationStrategy does not have `experiment` attr.
            gs._experiment = self.experiment_from_sqa(gs_sqa.experiment)
            gs._restore_model_from_generator_run()
        gs._db_id = gs_sqa.id
        return gs

    def runner_from_sqa(self, runner_sqa: SQARunner) -> Runner:
        """Convert SQLAlchemy Runner to Ax Runner."""
        runner_class = REVERSE_RUNNER_REGISTRY.get(runner_sqa.runner_type)
        if runner_class is None:
            raise SQADecodeError(
                f"Cannot decode SQARunner because {runner_sqa.runner_type} "
                f"is an invalid type."
            )
        args = self.get_init_args_from_properties(
            # pyre-fixme[6]: Expected `SQABase` for ...es` but got `SQARunner`.
            object_sqa=runner_sqa,
            class_=runner_class,
        )
        # pyre-fixme[45]: Cannot instantiate abstract class `Runner`.
        return runner_class(**args)

    def trial_from_sqa(self, trial_sqa: SQATrial, experiment: Experiment) -> BaseTrial:
        """Convert SQLAlchemy Trial to Ax Trial."""
        if trial_sqa.is_batch:
            trial = BatchTrial(
                experiment=experiment,
                optimize_for_power=trial_sqa.optimize_for_power,
                ttl_seconds=trial_sqa.ttl_seconds,
            )
            generator_run_structs = [
                GeneratorRunStruct(
                    generator_run=self.generator_run_from_sqa(
                        generator_run_sqa=generator_run_sqa
                    ),
                    weight=generator_run_sqa.weight or 1.0,
                )
                for generator_run_sqa in trial_sqa.generator_runs
            ]
            if trial_sqa.status_quo_name is not None:
                new_generator_run_structs = []
                for struct in generator_run_structs:
                    if (
                        struct.generator_run.generator_run_type
                        == GeneratorRunType.STATUS_QUO.name
                    ):
                        status_quo_weight = struct.generator_run.weights[0]
                        trial._status_quo = struct.generator_run.arms[0]
                        trial._status_quo_weight_override = status_quo_weight
                    else:
                        new_generator_run_structs.append(struct)
                generator_run_structs = new_generator_run_structs
            trial._generator_run_structs = generator_run_structs
            trial._abandoned_arms_metadata = {
                abandoned_arm_sqa.name: self.abandoned_arm_from_sqa(
                    abandoned_arm_sqa=abandoned_arm_sqa
                )
                for abandoned_arm_sqa in trial_sqa.abandoned_arms
            }
        else:
            trial = Trial(experiment=experiment, ttl_seconds=trial_sqa.ttl_seconds)
            if trial_sqa.generator_runs:
                if len(trial_sqa.generator_runs) != 1:
                    raise SQADecodeError(  # pragma: no cover
                        "Cannot decode SQATrial to Trial because trial is not batched "
                        "but has more than one generator run."
                    )
                trial._generator_run = self.generator_run_from_sqa(
                    generator_run_sqa=trial_sqa.generator_runs[0]
                )
        trial._index = trial_sqa.index
        trial._trial_type = trial_sqa.trial_type
        # Swap `DISPATCHED` for `RUNNING`, since `DISPATCHED` is deprecated and nearly
        # equivalent to `RUNNING`.
        trial._status = (
            trial_sqa.status
            if trial_sqa.status != TrialStatus.DISPATCHED
            else TrialStatus.RUNNING
        )
        trial._time_created = trial_sqa.time_created
        trial._time_completed = trial_sqa.time_completed
        trial._time_staged = trial_sqa.time_staged
        trial._time_run_started = trial_sqa.time_run_started
        trial._abandoned_reason = trial_sqa.abandoned_reason
        # pyre-fixme[9]: _run_metadata has type `Dict[str, Any]`; used as
        #  `Optional[Dict[str, Any]]`.
        trial._run_metadata = (
            # pyre-fixme[6]: Expected `Mapping[Variable[_KT], Variable[_VT]]` for
            #  1st param but got `Optional[Dict[str, typing.Any]]`.
            dict(trial_sqa.run_metadata)
            if trial_sqa.run_metadata is not None
            else None
        )
        trial._num_arms_created = trial_sqa.num_arms_created
        trial._runner = (
            self.runner_from_sqa(trial_sqa.runner) if trial_sqa.runner else None
        )
        trial._generation_step_index = trial_sqa.generation_step_index
        return trial

    def data_from_sqa(self, data_sqa: SQAData) -> Data:
        """Convert SQLAlchemy Data to AE Data."""

        # Need dtype=False, otherwise infers arm_names like "4_1" should be int 41
        return Data(
            description=data_sqa.description,
            df=pd.read_json(data_sqa.data_json, dtype=False),
        )
