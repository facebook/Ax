#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.metric import Metric
from ax.core.objective import Objective
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
from ax.exceptions.storage import SQAEncodeError
from ax.storage.metric_registry import METRIC_REGISTRY
from ax.storage.runner_registry import RUNNER_REGISTRY
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
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.utils import (
    DomainType,
    MetricIntent,
    ParameterConstraintType,
    get_object_properties,
)
from ax.utils.common.typeutils import not_none


class Encoder:
    """Class that contains methods for storing an Ax experiment to SQLAlchemy.

    Instantiate with an instance of Config to customize the functionality.
    For even more flexibility, create a subclass.

    Attributes:
        config: Metadata needed to save and load an experiment to SQLAlchemy.
    """

    def __init__(self, config: SQAConfig) -> None:
        self.config = config

    def get_enum_value(
        self, value: Optional[str], enum: Optional[Enum]
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

    def experiment_to_sqa(self, experiment: Experiment) -> SQAExperiment:
        """Convert Ax Experiment to SQLAlchemy.

        In addition to creating and storing a new Experiment object, we need to
        create and store copies of the Trials, Metrics, Parameters,
        ParameterConstraints, and Runner owned by this Experiment.
        """
        optimization_metrics = self.optimization_config_to_sqa(
            experiment.optimization_config
        )
        tracking_metrics = [
            self.metric_to_sqa(metric)
            for metric in experiment.metrics.values()
            if experiment.optimization_config is None
            or metric.name not in not_none(experiment.optimization_config).metrics
        ]
        parameters, parameter_constraints = self.search_space_to_sqa(
            experiment.search_space
        )

        status_quo_name = None
        status_quo_parameters = None
        if experiment.status_quo is not None:
            status_quo_name = experiment.status_quo.name
            status_quo_parameters = experiment.status_quo.parameters

        trials = [
            self.trial_to_sqa(trial=trial) for trial in experiment.trials.values()
        ]
        runner = self.runner_to_sqa(experiment.runner) if experiment.runner else None

        experiment_data = []
        for trial_index, data_by_timestamp in experiment.data_by_trial.items():
            for timestamp, data in data_by_timestamp.items():
                experiment_data.append(
                    self.data_to_sqa(
                        data=data, trial_index=trial_index, timestamp=timestamp
                    )
                )

        experiment_type = self.get_enum_value(
            value=experiment.experiment_type, enum=self.config.experiment_type_enum
        )

        properties = {}
        if isinstance(experiment, SimpleExperiment):
            properties["subclass"] = "SimpleExperiment"

        # pyre-fixme: Expected `Base` for 1st...yping.Type[Experiment]`.
        experiment_class: SQAExperiment = self.config.class_to_sqa_class[Experiment]
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
            data=experiment_data,
            properties=properties,
        )

    def parameter_to_sqa(self, parameter: Parameter) -> SQAParameter:
        """Convert Ax Parameter to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...typing.Type[Parameter]`.
        parameter_class: SQAParameter = self.config.class_to_sqa_class[Parameter]
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
                is_fidelity=parameter.is_fidelity,
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
                is_fidelity=parameter.is_fidelity,
            )
        elif isinstance(parameter, FixedParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                name=parameter.name,
                domain_type=DomainType.FIXED,
                parameter_type=parameter.parameter_type,
                fixed_value=parameter.value,
                is_fidelity=parameter.is_fidelity,
            )
        else:
            raise SQAEncodeError(
                "Cannot encode parameter to SQLAlchemy because parameter's "
                "subclass is invalid."
            )  # pragma: no cover

    def parameter_constraint_to_sqa(
        self, parameter_constraint: ParameterConstraint
    ) -> SQAParameterConstraint:
        """Convert Ax ParameterConstraint to SQLAlchemy."""
        # pyre-fixme[9]: parameter_constraint_cl... used as type `SQABase`.
        param_constraint_cls: SQAParameterConstraint = self.config.class_to_sqa_class[
            ParameterConstraint
        ]
        if isinstance(parameter_constraint, OrderConstraint):
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return param_constraint_cls(
                type=ParameterConstraintType.ORDER,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        elif isinstance(parameter_constraint, SumConstraint):
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return param_constraint_cls(
                type=ParameterConstraintType.SUM,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        else:
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return param_constraint_cls(
                type=ParameterConstraintType.LINEAR,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )

    def search_space_to_sqa(
        self, search_space: Optional[SearchSpace]
    ) -> Tuple[List[SQAParameter], List[SQAParameterConstraint]]:
        """Convert Ax SearchSpace to a list of SQLAlchemy Parameters and
        ParameterConstraints.
        """
        if search_space is None:
            return [], []

        parameters = [
            self.parameter_to_sqa(parameter=parameter)
            for parameter in search_space.parameters.values()
        ]
        parameter_constraints = [
            self.parameter_constraint_to_sqa(parameter_constraint=parameter_constraint)
            for parameter_constraint in search_space.parameter_constraints
        ]
        return parameters, parameter_constraints

    def get_metric_type_and_properties(
        self, metric: Metric
    ) -> Tuple[int, Dict[str, Any]]:
        """Given an Ax Metric, convert its type into a member of MetricType enum,
        and construct a dictionary to be stored in the database `properties`
        json blob.
        """
        metric_type = METRIC_REGISTRY.get(type(metric))
        if metric_type is None:
            raise SQAEncodeError(
                "Cannot encode metric to SQLAlchemy because metric's "
                "subclass is invalid."
            )  # pragma: no cover

        # name and lower_is_better are stored directly on the metric,
        # so we don't need to include these in the properties blob
        properties = get_object_properties(
            object=metric, exclude_fields=["name", "lower_is_better"]
        )

        return metric_type, properties

    def metric_to_sqa(self, metric: Metric) -> SQAMetric:
        """Convert Ax Metric to SQLAlchemy."""
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.TRACKING,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    def objective_to_sqa(self, objective: Objective) -> SQAMetric:
        """Convert Ax Objective to SQLAlchemy."""
        metric = objective.metric
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.OBJECTIVE,
            minimize=objective.minimize,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    def outcome_constraint_to_sqa(
        self, outcome_constraint: OutcomeConstraint
    ) -> SQAMetric:
        """Convert Ax OutcomeConstraint to SQLAlchemy."""
        metric = outcome_constraint.metric
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
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

    def optimization_config_to_sqa(
        self, optimization_config: Optional[OptimizationConfig]
    ) -> List[SQAMetric]:
        """Convert Ax OptimizationConfig to a list of SQLAlchemy Metrics."""
        if optimization_config is None:
            return []

        objective_sqa = self.objective_to_sqa(objective=optimization_config.objective)
        outcome_constraints_sqa = [
            self.outcome_constraint_to_sqa(outcome_constraint=constraint)
            for constraint in optimization_config.outcome_constraints
        ]
        return [objective_sqa] + outcome_constraints_sqa

    def arm_to_sqa(self, arm: Arm, weight: Optional[float] = 1.0) -> SQAArm:
        """Convert Ax Arm to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st... got `typing.Type[Arm]`.
        arm_class: SQAArm = self.config.class_to_sqa_class[Arm]
        # pyre-fixme[29]: `SQAArm` is not a function.
        return arm_class(parameters=arm.parameters, name=arm._name, weight=weight)

    def abandoned_arm_to_sqa(self, abandoned_arm: AbandonedArm) -> SQAAbandonedArm:
        """Convert Ax AbandonedArm to SQLAlchemy."""
        # pyre-fixme[9]: abandoned_arm_class is ....sqa_store.db.SQABase]`.
        abandoned_arm_class: SQAAbandonedArm = self.config.class_to_sqa_class[
            # pyre-fixme[6]: Expected `typing.Type[B...ing.Type[AbandonedArm]`.
            AbandonedArm
        ]
        # pyre-fixme[29]: `SQAAbandonedArm` is not a function.
        return abandoned_arm_class(
            name=abandoned_arm.name,
            abandoned_reason=abandoned_arm.reason,
            time_abandoned=abandoned_arm.time,
        )

    def generator_run_to_sqa(
        self, generator_run: GeneratorRun, weight: Optional[float] = None
    ) -> SQAGeneratorRun:
        """Convert Ax GeneratorRun to SQLAlchemy.

        In addition to creating and storing a new GeneratorRun object, we need to
        create and store copies of the Arms, Metrics, Parameters, and
        ParameterConstraints owned by this GeneratorRun.
        """
        arms = [
            self.arm_to_sqa(arm=arm, weight=weight)
            for (arm, weight) in generator_run.arm_weights.items()
        ]

        metrics = self.optimization_config_to_sqa(generator_run.optimization_config)
        parameters, parameter_constraints = self.search_space_to_sqa(
            generator_run.search_space
        )

        best_arm_name = None
        best_arm_parameters = None
        best_arm_predictions = None
        if generator_run.best_arm_predictions is not None:
            best_arm = generator_run.best_arm_predictions[0]
            best_arm_predictions = list(generator_run.best_arm_predictions[1])
            best_arm_name = best_arm._name
            best_arm_parameters = best_arm.parameters
        model_predictions = (
            list(generator_run.model_predictions)
            if generator_run.model_predictions is not None
            else None
        )

        generator_run_type = self.get_enum_value(
            value=generator_run.generator_run_type,
            enum=self.config.generator_run_type_enum,
        )

        # pyre-fixme: Expected `Base` for 1st...ing.Type[GeneratorRun]`.
        generator_run_class: SQAGeneratorRun = self.config.class_to_sqa_class[
            GeneratorRun
        ]
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

    def runner_to_sqa(self, runner: Runner) -> SQARunner:
        """Convert Ax Runner to SQLAlchemy."""
        runner_type = RUNNER_REGISTRY.get(type(runner))
        if runner_type is None:
            raise SQAEncodeError(
                "Cannot encode runner to SQLAlchemy because runner's "
                "subclass is invalid."
            )  # pragma: no cover
        properties = get_object_properties(object=runner)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Runner]`.
        runner_class: SQARunner = self.config.class_to_sqa_class[Runner]
        # pyre-fixme[29]: `SQARunner` is not a function.
        return runner_class(runner_type=runner_type, properties=properties)

    def trial_to_sqa(self, trial: BaseTrial) -> SQATrial:
        """Convert Ax Trial to SQLAlchemy.

        In addition to creating and storing a new Trial object, we need to
        create and store the GeneratorRuns and Runner that it owns.
        """
        runner = self.runner_to_sqa(runner=trial.runner) if trial.runner else None
        abandoned_arms = []
        generator_runs = []
        status_quo_name = None
        if isinstance(trial, BatchTrial):
            abandoned_arms = [
                self.abandoned_arm_to_sqa(abandoned_arm=abandoned_arm)
                for abandoned_arm in trial.abandoned_arms_metadata
            ]
            generator_runs = [
                self.generator_run_to_sqa(
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
                    self.generator_run_to_sqa(generator_run=status_quo_generator_run)
                )
                status_quo_name = trial.status_quo.name
        elif isinstance(trial, Trial):
            if trial.generator_run:
                generator_runs = [
                    self.generator_run_to_sqa(generator_run=trial.generator_run)
                ]

        # pyre-fixme: Expected `Base` for 1st...ot `typing.Type[Trial]`.
        trial_class: SQATrial = self.config.class_to_sqa_class[Trial]
        # pyre-fixme[29]: `SQATrial` is not a function.
        return trial_class(
            abandoned_reason=trial.abandoned_reason,
            deployed_name=trial.deployed_name,
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

    def data_to_sqa(self, data: Data, trial_index: int, timestamp: int) -> SQAData:
        """Convert AE data to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...ot `typing.Type[Data]`.
        data_class: SQAData = self.config.class_to_sqa_class[Data]
        # pyre-fixme[29]: `SQAData` is not a function.
        return data_class(
            data_json=data.df.to_json(),
            description=data.description,
            time_created=timestamp,
            trial_index=trial_index,
        )
