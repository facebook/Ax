#!/usr/bin/env python3

import inspect
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.base import Base
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.batch_trial import AbandonedArm, BatchTrial
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
from ae.lazarus.ae.exceptions.storage import SQAEncodeError
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
from ae.lazarus.ae.storage.utils import (
    DomainType,
    MetricIntent,
    ParameterConstraintType,
)
from ae.lazarus.ae.utils.common.typeutils import not_none


class Encoder:
    """Base class that contains methods for storing an AE experiment to SQLAlchemy.

    Create a subclass that inherits from Encoder to implement custom save
    functionality (e.g. special validation) for a given user-facing typeself.
    This class can then be passed into _save_experiment (defined in save.py).

    Class attributes that can be overwritten by a subclass include:
    -- class_to_sqa_class: Mapping of user-facing class to SQLAlchemy class
            that it will be encoded to. This allows overwriting of the default
            classes to provide custom save functionality.
    -- metric_registry: Maps Metric classes to an int constant representing
            their type. Ensures that when we store metric types, they will
            correspond to an existing Metric class.
    -- runner_registry: Maps Runner classes to an int constaint representing
            their type. Ensures that when we store runner types, they will
            correspond to an existing Runner class.
    -- experiment_type_enum: Enum containing valid Experiment types.
    -- generator_run_type_enum: Enum containing valid Generator Run types.
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

    @classmethod
    def get_enum_value(
        cls, value: Optional[str], enum: Optional[Enum]
    ) -> Optional[int]:
        """Given an enum name (string) and an enum (of ints), return the
        corresponding enum value. If the name is not present in the enum,
        throw an error.
        """
        if value is None or enum is None:
            return None

        try:
            return enum[value].value  # pyre-ignore T29651755
        except KeyError:
            raise SQAEncodeError(f"Value {value} is invalid for enum {enum}.")

    @classmethod
    def experiment_to_sqa(cls, experiment: Experiment) -> SQAExperiment:
        """Convert AE Experiment to SQLAlchemy.

        In addition to creating and storing a new Experiment object, we need to
        create and store copies of the Trials, Metrics, Parameters,
        ParameterConstraints, and Runner owned by this Experiment.
        """
        optimization_metrics = cls.optimization_config_to_sqa(
            experiment.optimization_config
        )
        tracking_metrics = [
            cls.metric_to_sqa(metric)
            for metric in experiment.metrics.values()
            if experiment.optimization_config is None
            or metric.name not in not_none(experiment.optimization_config).metrics
        ]
        parameters, parameter_constraints = cls.search_space_to_sqa(
            experiment.search_space
        )

        status_quo_name = None
        status_quo_parameters = None
        if experiment.status_quo is not None:
            status_quo_name = experiment.status_quo.name
            status_quo_parameters = experiment.status_quo.params

        trials = [cls.trial_to_sqa(trial=trial) for trial in experiment.trials.values()]
        runner = cls.runner_to_sqa(experiment.runner) if experiment.runner else None

        experiment_type = cls.get_enum_value(
            value=experiment.experiment_type, enum=cls.experiment_type_enum
        )

        # pyre-fixme: Expected `Base` for 1st...yping.Type[Experiment]`.
        experiment_class: SQAExperiment = cls.class_to_sqa_class[Experiment]
        # pyre-fixme[29]: `SQAExperiment` is not a function.
        return experiment_class(
            description=experiment.description,
            is_test=experiment.is_test,
            name=experiment.name,
            status_quo_name=status_quo_name,
            status_quo_parameters=status_quo_parameters,
            time_created=experiment.time_created,
            experiment_type=experiment_type,
            metrics=optimization_metrics + tracking_metrics,
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            trials=trials,
            runner=runner,
        )

    @classmethod
    def parameter_to_sqa(cls, parameter: Parameter) -> SQAParameter:
        """Convert AE Parameter to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...typing.Type[Parameter]`.
        parameter_class: SQAParameter = cls.class_to_sqa_class[Parameter]
        if isinstance(parameter, RangeParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                name=parameter.name,
                domain_type=DomainType.RANGE,
                parameter_type=parameter.parameter_type,
                lower=float(parameter.lower),
                upper=float(parameter.upper),
                log_scale=parameter.log_scale,
                digits=parameter.digits,
            )
        elif isinstance(parameter, ChoiceParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                name=parameter.name,
                domain_type=DomainType.CHOICE,
                parameter_type=parameter.parameter_type,
                choice_values=parameter.values,
                is_ordered=parameter.is_ordered,
                is_task=parameter.is_task,
            )
        elif isinstance(parameter, FixedParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                name=parameter.name,
                domain_type=DomainType.FIXED,
                parameter_type=parameter.parameter_type,
                fixed_value=parameter.value,
            )
        else:
            raise SQAEncodeError(
                "Cannot encode parameter to SQLAlchemy because parameter's "
                "subclass is invalid."
            )  # pragma: no cover

    @classmethod
    def parameter_constraint_to_sqa(
        cls, parameter_constraint: ParameterConstraint
    ) -> SQAParameterConstraint:
        """Convert AE ParameterConstraint to SQLAlchemy."""
        # pyre-fixme[9]: parameter_constraint_cl... used as type `SQABase`.
        parameter_constraint_class: SQAParameterConstraint = cls.class_to_sqa_class[
            ParameterConstraint
        ]
        if isinstance(parameter_constraint, OrderConstraint):
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return parameter_constraint_class(
                type=ParameterConstraintType.ORDER,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        elif isinstance(parameter_constraint, SumConstraint):
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return parameter_constraint_class(
                type=ParameterConstraintType.SUM,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        else:
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return parameter_constraint_class(
                type=ParameterConstraintType.LINEAR,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )

    @classmethod
    def search_space_to_sqa(
        cls, search_space: Optional[SearchSpace]
    ) -> Tuple[List[SQAParameter], List[SQAParameterConstraint]]:
        """Convert AE SearchSpace to a list of SQLAlchemy Parameters and
        ParameterConstraints.
        """
        if search_space is None:
            return [], []

        parameters = [
            cls.parameter_to_sqa(parameter=parameter)
            for parameter in search_space.parameters.values()
        ]
        parameter_constraints = [
            cls.parameter_constraint_to_sqa(parameter_constraint=parameter_constraint)
            for parameter_constraint in search_space.parameter_constraints
        ]
        return parameters, parameter_constraints

    @classmethod
    def get_properties(
        cls, object: Base, exclude_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Given an AE object, return a dictionary of the attributes that are
        needed by its constructor, and which we therefore need to store
        in the properties blob in the DB.
        """
        properties = {}
        exclude_args = ["self"] + (exclude_fields or [])
        for arg in inspect.getfullargspec(object.__class__.__init__).args:
            if arg in exclude_args:
                continue
            try:
                value = getattr(object, arg)
            except AttributeError:
                raise SQAEncodeError(
                    f"{object.__class__} is missing a value for {arg}, "
                    f"which is needed by its constructor."
                )
            properties[arg] = value
        return properties

    @classmethod
    def get_metric_type_and_properties(
        cls, metric: Metric
    ) -> Tuple[str, Dict[str, Any]]:
        """Given an AE Metric, convert its type into a member of MetricType enum,
        and construct a dictionary to be stored in the database `properties`
        json blob.
        """
        metric_type = cls.metric_registry.CLASS_TO_TYPE.get(type(metric))
        if metric_type is None:
            raise SQAEncodeError(
                "Cannot encode metric to SQLAlchemy because metric's "
                "subclass is invalid."
            )  # pragma: no cover

        # name and lower_is_better are stored directly on the metric,
        # so we don't need to include these in the properties blob
        properties = cls.get_properties(
            object=metric, exclude_fields=["name", "lower_is_better"]
        )

        return metric_type, properties

    @classmethod
    def metric_to_sqa(cls, metric: Metric) -> SQAMetric:
        """Convert AE Metric to SQLAlchemy."""
        metric_type, properties = cls.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = cls.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.TRACKING,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    @classmethod
    def objective_to_sqa(cls, objective: Objective) -> SQAMetric:
        """Convert AE Objective to SQLAlchemy."""
        metric = objective.metric
        metric_type, properties = cls.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = cls.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.OBJECTIVE,
            minimize=objective.minimize,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    @classmethod
    def outcome_constraint_to_sqa(
        cls, outcome_constraint: OutcomeConstraint
    ) -> SQAMetric:
        """Convert AE OutcomeConstraint to SQLAlchemy."""
        metric = outcome_constraint.metric
        metric_type, properties = cls.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = cls.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.OUTCOME_CONSTRAINT,
            bound=outcome_constraint.bound,
            op=outcome_constraint.op,
            relative=outcome_constraint.relative,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    @classmethod
    def optimization_config_to_sqa(
        cls, optimization_config: Optional[OptimizationConfig]
    ) -> List[SQAMetric]:
        """Convert AE OptimizationConfig to a list of SQLAlchemy Metrics."""
        if optimization_config is None:
            return []

        objective_sqa = cls.objective_to_sqa(objective=optimization_config.objective)
        outcome_constraints_sqa = [
            cls.outcome_constraint_to_sqa(outcome_constraint=constraint)
            for constraint in optimization_config.outcome_constraints
        ]
        return [objective_sqa] + outcome_constraints_sqa

    @classmethod
    def arm_to_sqa(cls, arm: Arm, weight: Optional[float] = 1.0) -> SQAArm:
        """Convert AE Arm to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st... got `typing.Type[Arm]`.
        arm_class: SQAArm = cls.class_to_sqa_class[Arm]
        # pyre-fixme[29]: `SQAArm` is not a function.
        return arm_class(parameters=arm.params, name=arm._name, weight=weight)

    @classmethod
    def abandoned_arm_to_sqa(cls, abandoned_arm: AbandonedArm) -> SQAAbandonedArm:
        """Convert AE AbandonedArm to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...ing.Type[AbandonedArm]`.
        abandoned_arm_class: SQAAbandonedArm = cls.class_to_sqa_class[AbandonedArm]
        # pyre-fixme[29]: `SQAAbandonedArm` is not a function.
        return abandoned_arm_class(
            name=abandoned_arm.name,
            abandoned_reason=abandoned_arm.reason,
            time_abandoned=abandoned_arm.time,
        )

    @classmethod
    def generator_run_to_sqa(
        cls, generator_run: GeneratorRun, weight: Optional[float] = None
    ) -> SQAGeneratorRun:
        """Convert AE GeneratorRun to SQLAlchemy.

        In addition to creating and storing a new GeneratorRun object, we need to
        create and store copies of the Arms, Metrics, Parameters, and
        ParameterConstraints owned by this GeneratorRun.
        """
        arms = [
            cls.arm_to_sqa(arm=arm, weight=weight)
            for (arm, weight) in generator_run.arm_weights.items()
        ]

        metrics = cls.optimization_config_to_sqa(generator_run.optimization_config)
        parameters, parameter_constraints = cls.search_space_to_sqa(
            generator_run.search_space
        )

        best_arm_name = None
        best_arm_parameters = None
        best_arm_predictions = None
        if generator_run.best_arm_predictions is not None:
            best_arm = generator_run.best_arm_predictions[0]
            best_arm_predictions = list(generator_run.best_arm_predictions[1])
            # pyre-fixme[16]: Optional type has no attribute `_name`.
            best_arm_name = best_arm._name
            # pyre-fixme[16]: Optional type has no attribute `params`.
            best_arm_parameters = best_arm.params
        model_predictions = (
            list(generator_run.model_predictions)
            if generator_run.model_predictions is not None
            else None
        )

        generator_run_type = cls.get_enum_value(
            value=generator_run.generator_run_type, enum=cls.generator_run_type_enum
        )

        # pyre-fixme: Expected `Base` for 1st...ing.Type[GeneratorRun]`.
        generator_run_class: SQAGeneratorRun = cls.class_to_sqa_class[GeneratorRun]
        # pyre-fixme[29]: `SQAGeneratorRun` is not a function.
        return generator_run_class(
            arms=arms,
            metrics=metrics,
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            time_created=generator_run.time_created,
            generator_run_type=generator_run_type,
            weight=weight,
            index=generator_run.index,
            fit_time=generator_run.fit_time,
            gen_time=generator_run.gen_time,
            best_arm_name=best_arm_name,
            best_arm_parameters=best_arm_parameters,
            best_arm_predictions=best_arm_predictions,
            model_predictions=model_predictions,
        )

    @classmethod
    def runner_to_sqa(cls, runner: Runner) -> SQARunner:
        """Convert AE Runner to SQLAlchemy."""
        runner_type = cls.runner_registry.CLASS_TO_TYPE.get(type(runner))
        if runner_type is None:
            raise SQAEncodeError(
                "Cannot encode runner to SQLAlchemy because runner's "
                "subclass is invalid."
            )  # pragma: no cover
        properties = cls.get_properties(object=runner)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Runner]`.
        runner_class: SQARunner = cls.class_to_sqa_class[Runner]
        # pyre-fixme[29]: `SQARunner` is not a function.
        return runner_class(runner_type=runner_type, properties=properties)

    @classmethod
    def trial_to_sqa(cls, trial: BaseTrial) -> SQATrial:
        """Convert AE Trial to SQLAlchemy.

        In addition to creating and storing a new Trial object, we need to
        create and store the GeneratorRuns and Runner that it owns.
        """
        runner = cls.runner_to_sqa(runner=trial.runner) if trial.runner else None
        abandoned_arms = []
        generator_runs = []
        status_quo_name = None
        if isinstance(trial, BatchTrial):
            abandoned_arms = [
                cls.abandoned_arm_to_sqa(abandoned_arm=abandoned_arm)
                for abandoned_arm in trial.abandoned_arms_metadata
            ]
            generator_runs = [
                cls.generator_run_to_sqa(
                    generator_run=struct.generator_run, weight=struct.weight
                )
                for struct in trial.generator_run_structs
            ]
            if trial.status_quo is not None:
                status_quo_generator_run = GeneratorRun(
                    arms=[trial.status_quo],
                    weights=[trial._status_quo_weight],
                    type=GeneratorRunType.STATUS_QUO.name,
                )
                # this is a hack necessary to get equality tests passing;
                # otherwise you can encode same object and get two different results
                status_quo_generator_run._time_created = trial.time_created
                generator_runs.append(
                    cls.generator_run_to_sqa(generator_run=status_quo_generator_run)
                )
                status_quo_name = trial.status_quo.name
        elif isinstance(trial, Trial):
            if trial.generator_run:
                generator_runs = [
                    cls.generator_run_to_sqa(generator_run=trial.generator_run)
                ]

        # pyre-fixme: Expected `Base` for 1st...ot `typing.Type[Trial]`.
        trial_class: SQATrial = cls.class_to_sqa_class[Trial]
        # pyre-fixme[29]: `SQATrial` is not a function.
        return trial_class(
            abandoned_reason=trial.abandoned_reason,
            index=trial.index,
            is_batch=isinstance(trial, BatchTrial),
            num_arms_created=trial._num_arms_created,
            run_metadata=trial.run_metadata,
            status=trial.status,
            status_quo_name=status_quo_name,
            time_completed=trial.time_completed,
            time_created=trial.time_created,
            time_staged=trial.time_staged,
            time_run_started=trial.time_run_started,
            trial_type=trial.trial_type,
            abandoned_arms=abandoned_arms,
            generator_runs=generator_runs,
            runner=runner,
        )
