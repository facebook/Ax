#!/usr/bin/env python3

from enum import EnumMeta
from typing import List, Optional, Tuple, Union

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.batch_trial import AbandonedArm, BatchTrial, GeneratorRunStruct
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun, GeneratorRunType
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
)
from ae.lazarus.ae.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ae.lazarus.ae.core.runner import Runner
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.exceptions.storage import SQADecodeError
from ae.lazarus.ae.metrics.registry import MetricRegistry
from ae.lazarus.ae.runners.registry import RunnerRegistry
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
from ae.lazarus.ae.storage.utils import (
    DomainType,
    MetricIntent,
    ParameterConstraintType,
)


class Decoder:
    """Base class that contains methods for loading an AE experiment from SQLAlchemy.

    Create a subclass that inherits from Decoder to implement custom load
    functionality for a given user-facing type. This class can then be passed
    into _load_experiment (defined in load.py).

    Class attributes that can be overwritten by a subclass include:
    -- metric_registry: Maps int constants to corresponding Metric classes.
            Ensures that when we load metric types, they will correspond
            to an existing Metric class.
    -- runner_registry: Maps int constants to corresponding Runner classes.
            Ensures that when we load runner types, they will correspond
            to an existing Runner class.
    -- experiment_type_enum: Enum containing valid Experiment types.
    -- generator_run_type_enum: Enum containing valid Generator Run types.
    """

    metric_registry: MetricRegistry = MetricRegistry()
    runner_registry: RunnerRegistry = RunnerRegistry()
    experiment_type_enum: Optional[EnumMeta] = None
    generator_run_type_enum: Optional[EnumMeta] = GeneratorRunType

    @classmethod
    def get_enum_name(
        cls, value: Optional[int], enum: Optional[EnumMeta]
    ) -> Optional[str]:
        """Given an enum value (int) and an enum (of ints), return the
        corresponding enum name. If the value is not present in the enum,
        throw an error.
        """
        if value is None or enum is None:
            return None

        for k, v in enum.__members__.items():
            # pyre-fixme[16]: `EnumMeta` has not attribute `value`
            if v.value == value:
                return k

        raise SQADecodeError(f"Value {value} is invalid for enum {enum}.")

    @classmethod
    def experiment_from_sqa(cls, experiment_sqa: SQAExperiment) -> Experiment:
        """Convert SQLAlchemy Experiment to AE Experiment."""
        optimization_config, tracking_metrics = cls.optimization_config_and_tracking_metrics_from_sqa(
            metrics_sqa=experiment_sqa.metrics
        )
        search_space = cls.search_space_from_sqa(
            parameters_sqa=experiment_sqa.parameters,
            parameter_constraints_sqa=experiment_sqa.parameter_constraints,
        )
        if search_space is None:
            raise SQADecodeError(  # pragma: no cover
                "Experiment SearchSpace cannot be None."
            )
        runner = (
            cls.runner_from_sqa(experiment_sqa.runner)
            if experiment_sqa.runner
            else None
        )
        status_quo = (
            Arm(
                params=experiment_sqa.status_quo_parameters,
                name=experiment_sqa.status_quo_name,
            )
            if experiment_sqa.status_quo_parameters is not None
            else None
        )

        experiment = Experiment(
            name=experiment_sqa.name,
            description=experiment_sqa.description,
            search_space=search_space,
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
            runner=runner,
            status_quo=status_quo,
            is_test=experiment_sqa.is_test,
        )
        trials = [
            cls.trial_from_sqa(trial_sqa=trial, experiment=experiment)
            for trial in experiment_sqa.trials
        ]
        experiment._trials = {trial.index: trial for trial in trials}
        experiment._time_created = experiment_sqa.time_created
        experiment._experiment_type = cls.get_enum_name(
            value=experiment_sqa.experiment_type, enum=cls.experiment_type_enum
        )

        return experiment

    @classmethod
    def parameter_from_sqa(cls, parameter_sqa: SQAParameter) -> Parameter:
        """Convert SQLAlchemy Parameter to AE Parameter."""
        if parameter_sqa.domain_type == DomainType.RANGE:
            if parameter_sqa.lower is None or parameter_sqa.upper is None:
                raise SQADecodeError(  # pragma: no cover
                    "`lower` and `upper` must be set for RangeParameter."
                )
            return RangeParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                lower=parameter_sqa.lower,
                upper=parameter_sqa.upper,
                log_scale=parameter_sqa.log_scale or False,
                digits=parameter_sqa.digits,
            )
        elif parameter_sqa.domain_type == DomainType.CHOICE:
            if parameter_sqa.choice_values is None:
                raise SQADecodeError(  # pragma: no cover
                    "`values` must be set for ChoiceParameter."
                )
            return ChoiceParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                values=parameter_sqa.choice_values,
            )
        elif parameter_sqa.domain_type == DomainType.FIXED:
            if parameter_sqa.fixed_value is None:
                raise SQADecodeError(  # pragma: no cover
                    "`value` must be set for FixedParameter."
                )
            return FixedParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                value=parameter_sqa.fixed_value,
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAParameter because {parameter_sqa.domain_type} "
                "is an invalid domain type."
            )

    @classmethod
    def parameter_constraint_from_sqa(
        cls, parameter_constraint_sqa: SQAParameterConstraint
    ) -> ParameterConstraint:
        """Convert SQLAlchemy ParameterConstraint to AE ParameterConstraint."""
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
            return OrderConstraint(lower_name=lower_name, upper_name=upper_name)
        elif parameter_constraint_sqa.type == ParameterConstraintType.SUM:
            parameter_names = list(parameter_constraint_sqa.constraint_dict.keys())
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
                parameter_names=parameter_names,
                is_upper_bound=is_upper_bound,
                bound=bound,
            )
        else:
            return ParameterConstraint(
                constraint_dict=dict(parameter_constraint_sqa.constraint_dict),
                bound=parameter_constraint_sqa.bound,
            )

    @classmethod
    def search_space_from_sqa(
        cls,
        parameters_sqa: List[SQAParameter],
        parameter_constraints_sqa: List[SQAParameterConstraint],
    ) -> Optional[SearchSpace]:
        """Convert a list of SQLAlchemy Parameters and ParameterConstraints to an
        AE SearchSpace.
        """
        parameters = [
            cls.parameter_from_sqa(parameter_sqa=parameter_sqa)
            for parameter_sqa in parameters_sqa
        ]
        parameter_constraints = [
            cls.parameter_constraint_from_sqa(
                parameter_constraint_sqa=parameter_constraint_sqa
            )
            for parameter_constraint_sqa in parameter_constraints_sqa
        ]

        if len(parameters) == 0:
            return None

        return SearchSpace(
            parameters=parameters, parameter_constraints=parameter_constraints
        )

    @classmethod
    def metric_from_sqa(
        cls, metric_sqa: SQAMetric
    ) -> Union[Metric, Objective, OutcomeConstraint]:
        """Convert SQLAlchemy Metric to AE Metric, Objective, or OutcomeConstraint."""
        metric_class = cls.metric_registry.TYPE_TO_CLASS.get(metric_sqa.metric_type)
        if metric_class is None:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.metric_type} "
                f"is an invalid type."
            )
        properties = metric_sqa.properties or {}
        # pyre-fixme[29]: `Type[Any]` is not a function.
        metric = metric_class(
            name=metric_sqa.name,
            lower_is_better=metric_sqa.lower_is_better,
            **properties,
        )

        if metric_sqa.intent == MetricIntent.TRACKING:
            return metric
        elif metric_sqa.intent == MetricIntent.OBJECTIVE:
            return Objective(metric=metric, minimize=metric_sqa.minimize)
        elif metric_sqa.intent == MetricIntent.OUTCOME_CONSTRAINT:
            return OutcomeConstraint(
                metric=metric,
                bound=metric_sqa.bound,
                op=metric_sqa.op,
                relative=metric_sqa.relative,
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.intent} "
                f"is an invalid intent."
            )

    @classmethod
    def optimization_config_and_tracking_metrics_from_sqa(
        cls, metrics_sqa: List[SQAMetric]
    ) -> Tuple[Optional[OptimizationConfig], List[Metric]]:
        """Convert a list of SQLAlchemy Metrics to a a tuple of AE OptimizationConfig
        and tracking metrics.
        """
        objective = None
        outcome_constraints = []
        tracking_metrics = []
        for metric_sqa in metrics_sqa:
            metric = cls.metric_from_sqa(metric_sqa=metric_sqa)
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

    @classmethod
    def arm_from_sqa(cls, arm_sqa: SQAArm) -> Arm:
        """Convert SQLAlchemy Arm to AE Arm."""
        return Arm(params=arm_sqa.parameters, name=arm_sqa.name)

    @classmethod
    def abandoned_arm_from_sqa(cls, abandoned_arm_sqa: SQAAbandonedArm) -> AbandonedArm:
        """Convert SQLAlchemy AbandonedArm to AE AbandonedArm."""
        return AbandonedArm(
            name=abandoned_arm_sqa.name,
            reason=abandoned_arm_sqa.abandoned_reason,
            time=abandoned_arm_sqa.time_abandoned,
        )

    @classmethod
    def generator_run_from_sqa(cls, generator_run_sqa: SQAGeneratorRun) -> GeneratorRun:
        """Convert SQLAlchemy GeneratorRun to AE GeneratorRun."""
        arms = []
        weights = []
        optimization_config = None
        search_space = None

        for arm_sqa in generator_run_sqa.arms:
            arms.append(cls.arm_from_sqa(arm_sqa=arm_sqa))
            weights.append(arm_sqa.weight)

        optimization_config, tracking_metrics = cls.optimization_config_and_tracking_metrics_from_sqa(
            metrics_sqa=generator_run_sqa.metrics
        )
        if len(tracking_metrics) > 0:
            raise SQADecodeError(  # pragma: no cover
                "GeneratorRun should not have tracking metrics."
            )

        search_space = cls.search_space_from_sqa(
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
                params=generator_run_sqa.best_arm_parameters,
            )
            best_arm_predictions = (
                best_arm,
                tuple(generator_run_sqa.best_arm_predictions),
            )
        model_predictions = (
            tuple(generator_run_sqa.model_predictions)
            if generator_run_sqa.model_predictions is not None
            else None
        )

        generator_run = GeneratorRun(
            arms=arms,
            weights=weights,
            optimization_config=optimization_config,
            search_space=search_space,
            fit_time=generator_run_sqa.fit_time,
            gen_time=generator_run_sqa.gen_time,
            best_arm_predictions=best_arm_predictions,
            model_predictions=model_predictions,
        )
        generator_run._time_created = generator_run_sqa.time_created
        generator_run._generator_run_type = cls.get_enum_name(
            value=generator_run_sqa.generator_run_type, enum=cls.generator_run_type_enum
        )
        generator_run._index = generator_run_sqa.index
        return generator_run

    @classmethod
    def runner_from_sqa(cls, runner_sqa: SQARunner) -> Runner:
        """Convert SQLAlchemy Runner to AE Runner."""
        runner_class = cls.runner_registry.TYPE_TO_CLASS.get(runner_sqa.runner_type)
        if runner_class is None:
            raise SQADecodeError(
                f"Cannot decode SQARunner because {runner_sqa.runner_type} "
                f"is an invalid type."
            )
        properties = runner_sqa.properties or {}
        # pyre-fixme[29]: `Type[Any]` is not a function.
        return runner_class(**properties)

    @classmethod
    def trial_from_sqa(cls, trial_sqa: SQATrial, experiment: Experiment) -> BaseTrial:
        """Convert SQLAlchemy Trial to AE Trial."""
        if trial_sqa.is_batch:
            trial = BatchTrial(experiment=experiment)
            generator_run_structs = [
                GeneratorRunStruct(
                    generator_run=cls.generator_run_from_sqa(
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
                        trial._status_quo = struct.generator_run.arms[0]
                        trial._status_quo_weight = struct.generator_run.weights[0]
                    else:
                        new_generator_run_structs.append(struct)
                generator_run_structs = new_generator_run_structs
            trial._generator_run_structs = generator_run_structs
            trial._abandoned_arms_metadata = {
                abandoned_arm_sqa.name: cls.abandoned_arm_from_sqa(
                    abandoned_arm_sqa=abandoned_arm_sqa
                )
                for abandoned_arm_sqa in trial_sqa.abandoned_arms
            }
        else:
            trial = Trial(experiment=experiment)
            if trial_sqa.generator_runs:
                if len(trial_sqa.generator_runs) != 1:
                    raise SQADecodeError(  # pragma: no cover
                        "Cannot decode SQATrial to Trial because trial is not batched "
                        "but has more than one generator run."
                    )
                trial._generator_run = cls.generator_run_from_sqa(
                    generator_run_sqa=trial_sqa.generator_runs[0]
                )
        trial._index = trial_sqa.index
        trial._trial_type = trial_sqa.trial_type
        trial._status = trial_sqa.status
        trial._time_created = trial_sqa.time_created
        trial._time_completed = trial_sqa.time_completed
        trial._time_staged = trial_sqa.time_staged
        trial._time_run_started = trial_sqa.time_run_started
        trial._abandoned_reason = trial_sqa.abandoned_reason
        trial._run_metadata = (
            dict(trial_sqa.run_metadata) if trial_sqa.run_metadata is not None else None
        )
        trial._num_arms_created = trial_sqa.num_arms_created
        trial._runner = (
            cls.runner_from_sqa(trial_sqa.runner) if trial_sqa.runner else None
        )
        return trial
