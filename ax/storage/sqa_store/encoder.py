#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from ax.core.abstract_data import AbstractDataFrameData
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
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
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.json_store.encoder import object_to_json
from ax.storage.metric_registry import METRIC_REGISTRY
from ax.storage.runner_registry import RUNNER_REGISTRY
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
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger = get_logger(__name__)


class Encoder:
    """Class that contains methods for storing an Ax experiment to SQLAlchemy.

    Instantiate with an instance of Config to customize the functionality.
    For even more flexibility, create a subclass.

    Attributes:
        config: Metadata needed to save and load an experiment to SQLAlchemy.
    """

    def __init__(self, config: SQAConfig) -> None:
        self.config = config

    @classmethod
    def validate_experiment_metadata(
        cls,
        experiment: Experiment,
        existing_sqa_experiment_id: Optional[int],
    ) -> None:
        """Validates required experiment metadata."""
        if experiment.db_id is not None:
            if existing_sqa_experiment_id is None:
                raise ValueError(
                    f"Experiment with ID {experiment.db_id} was already saved to the "
                    "database with a different name. Changing the name of an "
                    "experiment is not allowed."
                )
            elif experiment.db_id != existing_sqa_experiment_id:
                raise ValueError(
                    f"experiment.db_id = {experiment.db_id} but the experiment in the "
                    f"database with the name {experiment.name} has the id "
                    f"{existing_sqa_experiment_id}."
                )
        else:
            # experiment.db_id is None
            if existing_sqa_experiment_id is not None:
                raise ValueError(
                    f"An experiment already exists with the name {experiment.name}. "
                    "If you need to override this existing experiment, first delete it "
                    "via `delete_experiment` in ax/ax/storage/sqa_store/delete.py, "
                    "and then resave."
                )

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

        tracking_metrics = []
        for metric in experiment._tracking_metrics.values():
            tracking_metrics.append(self.metric_to_sqa(metric))

        parameters, parameter_constraints = self.search_space_to_sqa(
            experiment.search_space
        )

        status_quo_name = None
        status_quo_parameters = None
        if experiment.status_quo is not None:
            status_quo_name = not_none(experiment.status_quo).name
            status_quo_parameters = not_none(experiment.status_quo).parameters

        trials = []
        for trial in experiment.trials.values():
            trial_sqa = self.trial_to_sqa(trial=trial)
            trials.append(trial_sqa)

        experiment_data = []
        for trial_index, data_by_timestamp in experiment.data_by_trial.items():
            for timestamp, data in data_by_timestamp.items():
                data = self.data_to_sqa(
                    data=data, trial_index=trial_index, timestamp=timestamp
                )
                experiment_data.append(data)

        experiment_type = self.get_enum_value(
            value=experiment.experiment_type, enum=self.config.experiment_type_enum
        )

        properties = experiment._properties
        runners = []
        if isinstance(experiment, MultiTypeExperiment):
            properties[Keys.SUBCLASS] = "MultiTypeExperiment"
            for trial_type, runner in experiment._trial_type_to_runner.items():
                runner_sqa = self.runner_to_sqa(runner, trial_type)
                runners.append(runner_sqa)

            for metric in tracking_metrics:
                metric.trial_type = experiment._metric_to_trial_type[metric.name]
                if metric.name in experiment._metric_to_canonical_name:
                    metric.canonical_name = experiment._metric_to_canonical_name[
                        metric.name
                    ]
        elif experiment.runner:
            runners.append(self.runner_to_sqa(not_none(experiment.runner)))

        if isinstance(experiment, SimpleExperiment):
            properties[Keys.SUBCLASS] = "SimpleExperiment"

        # pyre-ignore[9]: Expected `Base` for 1st...yping.Type[Experiment]`.
        experiment_class: Type[SQAExperiment] = self.config.class_to_sqa_class[
            Experiment
        ]
        exp_sqa = experiment_class(
            id=experiment.db_id,  # pyre-ignore
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
            runners=runners,
            data=experiment_data,
            properties=properties,
            default_trial_type=experiment.default_trial_type,
            default_data_type=experiment.default_data_type,
        )
        return exp_sqa

    def parameter_to_sqa(self, parameter: Parameter) -> SQAParameter:
        """Convert Ax Parameter to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...typing.Type[Parameter]`.
        parameter_class: SQAParameter = self.config.class_to_sqa_class[Parameter]
        if isinstance(parameter, RangeParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                id=parameter.db_id,
                name=parameter.name,
                domain_type=DomainType.RANGE,
                parameter_type=parameter.parameter_type,
                lower=float(parameter.lower),
                upper=float(parameter.upper),
                log_scale=parameter.log_scale,
                digits=parameter.digits,
                is_fidelity=parameter.is_fidelity,
                target_value=parameter.target_value,
            )
        elif isinstance(parameter, ChoiceParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                id=parameter.db_id,
                name=parameter.name,
                domain_type=DomainType.CHOICE,
                parameter_type=parameter.parameter_type,
                choice_values=parameter.values,
                is_ordered=parameter.is_ordered,
                is_task=parameter.is_task,
                is_fidelity=parameter.is_fidelity,
                target_value=parameter.target_value,
            )
        elif isinstance(parameter, FixedParameter):
            # pyre-fixme[29]: `SQAParameter` is not a function.
            return parameter_class(
                id=parameter.db_id,
                name=parameter.name,
                domain_type=DomainType.FIXED,
                parameter_type=parameter.parameter_type,
                fixed_value=parameter.value,
                is_fidelity=parameter.is_fidelity,
                target_value=parameter.target_value,
            )
        else:
            raise SQAEncodeError(
                "Cannot encode parameter to SQLAlchemy because parameter's "
                f"subclass ({type(parameter)}) is invalid."
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
                id=parameter_constraint.db_id,
                type=ParameterConstraintType.ORDER,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        elif isinstance(parameter_constraint, SumConstraint):
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return param_constraint_cls(
                id=parameter_constraint.db_id,
                type=ParameterConstraintType.SUM,
                constraint_dict=parameter_constraint.constraint_dict,
                bound=parameter_constraint.bound,
            )
        else:
            # pyre-fixme[29]: `SQAParameterConstraint` is not a function.
            return param_constraint_cls(
                id=parameter_constraint.db_id,
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
        parameters, parameter_constraints = [], []
        if search_space is not None:
            for parameter in search_space.parameters.values():
                parameters.append(self.parameter_to_sqa(parameter=parameter))

            for parameter_constraint in search_space.parameter_constraints:
                parameter_constraints.append(
                    self.parameter_constraint_to_sqa(
                        parameter_constraint=parameter_constraint
                    )
                )

        return parameters, parameter_constraints

    def get_metric_type_and_properties(
        self, metric: Metric
    ) -> Tuple[int, Dict[str, Any]]:
        """Given an Ax Metric, convert its type into a member of MetricType enum,
        and construct a dictionary to be stored in the database `properties`
        json blob.
        """
        metric_class = type(metric)
        metric_type = METRIC_REGISTRY.get(metric_class)
        if metric_type is None:
            raise SQAEncodeError(
                "Cannot encode metric to SQLAlchemy because metric's "
                f"subclass ({metric_class}) is missing from the registry. "
                "The metric registry currently contains the following: "
                f"{','.join(map(str, METRIC_REGISTRY.keys()))}"
            )  # pragma: no cover

        properties = metric_class.serialize_init_args(metric=metric)
        return metric_type, properties

    def metric_to_sqa(self, metric: Metric) -> SQAMetric:
        """Convert Ax Metric to SQLAlchemy."""
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            id=metric.db_id,
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.TRACKING,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    def get_children_metrics_by_name(
        self, metrics: List[Metric], weights: List[float]
    ) -> Dict[str, Tuple[Metric, float, SQAMetric, Tuple[int, Dict[str, Any]]]]:
        return {
            metric.name: (
                metric,
                weight,
                cast(SQAMetric, self.config.class_to_sqa_class[Metric]),
                self.get_metric_type_and_properties(metric=metric),
            )
            for (metric, weight) in zip(metrics, weights)
        }

    def objective_to_sqa(self, objective: Objective) -> SQAMetric:
        """Convert Ax Objective to SQLAlchemy."""
        if isinstance(objective, ScalarizedObjective):
            objective_sqa = self.scalarized_objective_to_sqa(objective)

        elif isinstance(objective, MultiObjective):
            objective_sqa = self.multi_objective_to_sqa(objective)

        else:
            metric_type, properties = self.get_metric_type_and_properties(
                metric=objective.metric
            )
            metric_class = cast(SQAMetric, self.config.class_to_sqa_class[Metric])
            objective_sqa = (
                metric_class(  # pyre-ignore[29]: `SQAMetric` is not a function.
                    id=objective.metric.db_id,
                    name=objective.metric.name,
                    metric_type=metric_type,
                    intent=MetricIntent.OBJECTIVE,
                    minimize=objective.minimize,
                    properties=properties,
                    lower_is_better=objective.metric.lower_is_better,
                )
            )

        return checked_cast(SQAMetric, objective_sqa)

    def multi_objective_to_sqa(self, multi_objective: MultiObjective) -> SQAMetric:
        """Convert Ax Multi Objective to SQLAlchemy.

        Returns: A parent `SQAMetric`, whose children are the `SQAMetric`-s
            corresponding to `metrics` attribute of `MultiObjective`.
            NOTE: The parent is used as a placeholder for storage purposes.
        """
        # Constructing children SQAMetric classes (these are the real metrics in
        # the `MultiObjective`).
        children_objectives = []
        for objective in multi_objective.objectives:
            objective_cls = cast(SQAMetric, self.config.class_to_sqa_class[Metric])
            type_and_properties = self.get_metric_type_and_properties(
                metric=objective.metric
            )
            children_objectives.append(
                objective_cls(  # pyre-ignore[29]: `SQAMetric` is not a func.
                    id=objective.metric.db_id,
                    name=objective.metric.name,
                    metric_type=type_and_properties[0],
                    intent=MetricIntent.OBJECTIVE,
                    minimize=objective.minimize,
                    properties=type_and_properties[1],
                    lower_is_better=objective.metric.lower_is_better,
                )
            )

        # Constructing a parent SQAMetric class (not a real metric, only a placeholder
        # to group the metrics together).
        parent_metric_cls = cast(SQAMetric, self.config.class_to_sqa_class[Metric])
        parent_metric = (
            parent_metric_cls(  # pyre-ignore[29]: `SQAMetric` is not a func.
                id=multi_objective.db_id,
                name="multi_objective",
                metric_type=METRIC_REGISTRY[Metric],
                intent=MetricIntent.MULTI_OBJECTIVE,
                scalarized_objective_children_metrics=children_objectives,
            )
        )
        return parent_metric

    def scalarized_objective_to_sqa(self, objective: ScalarizedObjective) -> SQAMetric:
        """Convert Ax Scalarized Objective to SQLAlchemy.

        Returns: A parent `SQAMetric`, whose children are the `SQAMetric`-s
            corresponding to `metrics` attribute of `ScalarizedObjective`.
            NOTE: The parent is used as a placeholder for storage purposes.
        """
        metrics, weights = objective.metrics, objective.weights
        if (not (metrics and weights)) or len(metrics) != len(weights):
            raise SQAEncodeError(  # pragma: no cover
                "Metrics and weights in scalarized objective "
                "must be lists of equal length."
            )
        metrics_by_name = self.get_children_metrics_by_name(
            metrics=metrics, weights=weights
        )

        # Constructing children SQAMetric classes (these are the real metrics in
        # the `ScalarizedObjective`).
        children_metrics = []
        for metric_name in metrics_by_name:
            m, w, metric_cls, type_and_properties = metrics_by_name[metric_name]
            children_metrics.append(
                metric_cls(  # pyre-ignore[29]: `SQAMetric` is not a function.
                    id=m.db_id,
                    name=metric_name,
                    metric_type=type_and_properties[0],
                    intent=MetricIntent.OBJECTIVE,
                    minimize=objective.minimize,
                    properties=type_and_properties[1],
                    lower_is_better=m.lower_is_better,
                    scalarized_objective_weight=w,
                )
            )

        # Constructing a parent SQAMetric class
        parent_metric_cls = cast(SQAMetric, self.config.class_to_sqa_class[Metric])
        parent_metric = parent_metric_cls(  # pyre-ignore[29]: `SQAMetric` not a func.
            id=objective.db_id,
            name="scalarized_objective",
            metric_type=METRIC_REGISTRY[Metric],
            intent=MetricIntent.SCALARIZED_OBJECTIVE,
            minimize=objective.minimize,
            lower_is_better=objective.minimize,
            scalarized_objective_children_metrics=children_metrics,
        )
        return parent_metric

    def outcome_constraint_to_sqa(
        self, outcome_constraint: OutcomeConstraint
    ) -> SQAMetric:
        """Convert Ax OutcomeConstraint to SQLAlchemy."""
        if isinstance(outcome_constraint, ScalarizedOutcomeConstraint):
            return self.scalarized_outcome_constraint_to_sqa(outcome_constraint)

        metric = outcome_constraint.metric
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        constraint_sqa = metric_class(
            id=metric.db_id,
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.OUTCOME_CONSTRAINT,
            bound=outcome_constraint.bound,
            op=outcome_constraint.op,
            relative=outcome_constraint.relative,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )
        return constraint_sqa

    def scalarized_outcome_constraint_to_sqa(
        self, outcome_constraint: ScalarizedOutcomeConstraint
    ) -> SQAMetric:
        """Convert Ax SCalarized OutcomeConstraint to SQLAlchemy."""
        metrics, weights = outcome_constraint.metrics, outcome_constraint.weights

        if metrics is None or weights is None or len(metrics) != len(weights):
            raise SQAEncodeError(
                "Metrics and weights in scalarized OutcomeConstraint \
                must be lists of equal length."
            )  # pragma: no cover

        metrics_by_name = self.get_children_metrics_by_name(
            metrics=metrics, weights=weights
        )
        # Constructing children SQAMetric classes (these are the real metrics in
        # the `ScalarizedObjective`).
        children_metrics = []
        for metric_name in metrics_by_name:
            m, w, metric_cls, type_and_properties = metrics_by_name[metric_name]
            children_metrics.append(
                metric_cls(  # pyre-ignore[29]: `SQAMetric` is not a function.
                    id=m.db_id,
                    name=metric_name,
                    metric_type=type_and_properties[0],
                    intent=MetricIntent.OUTCOME_CONSTRAINT,
                    properties=type_and_properties[1],
                    lower_is_better=m.lower_is_better,
                    scalarized_outcome_constraint_weight=w,
                    bound=outcome_constraint.bound,
                    op=outcome_constraint.op,
                    relative=outcome_constraint.relative,
                )
            )

        # Constructing a parent SQAMetric class
        parent_metric_cls = cast(SQAMetric, self.config.class_to_sqa_class[Metric])
        parent_metric = parent_metric_cls(  # pyre-ignore[29]: `SQAMetric` not a func.
            id=outcome_constraint.db_id,
            name="scalarized_outcome_constraint",
            metric_type=METRIC_REGISTRY[Metric],
            intent=MetricIntent.SCALARIZED_OUTCOME_CONSTRAINT,
            bound=outcome_constraint.bound,
            op=outcome_constraint.op,
            relative=outcome_constraint.relative,
            scalarized_outcome_constraint_children_metrics=children_metrics,
        )
        return parent_metric

    def objective_threshold_to_sqa(
        self, objective_threshold: ObjectiveThreshold
    ) -> SQAMetric:
        """Convert Ax OutcomeConstraint to SQLAlchemy."""
        metric = objective_threshold.metric
        metric_type, properties = self.get_metric_type_and_properties(metric=metric)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Metric]`.
        metric_class: SQAMetric = self.config.class_to_sqa_class[Metric]
        # pyre-fixme[29]: `SQAMetric` is not a function.
        return metric_class(
            id=metric.db_id,
            name=metric.name,
            metric_type=metric_type,
            intent=MetricIntent.OBJECTIVE_THRESHOLD,
            bound=objective_threshold.bound,
            op=objective_threshold.op,
            relative=objective_threshold.relative,
            properties=properties,
            lower_is_better=metric.lower_is_better,
        )

    def optimization_config_to_sqa(
        self, optimization_config: Optional[OptimizationConfig]
    ) -> List[SQAMetric]:
        """Convert Ax OptimizationConfig to a list of SQLAlchemy Metrics."""
        if optimization_config is None:
            return []

        metrics_sqa = []
        obj_sqa = self.objective_to_sqa(objective=optimization_config.objective)
        metrics_sqa.append(obj_sqa)
        for constraint in optimization_config.outcome_constraints:
            constraint_sqa = self.outcome_constraint_to_sqa(
                outcome_constraint=constraint
            )
            metrics_sqa.append(constraint_sqa)
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            for threshold in optimization_config.objective_thresholds:
                threshold_sqa = self.objective_threshold_to_sqa(
                    objective_threshold=threshold
                )
                metrics_sqa.append(threshold_sqa)
        return metrics_sqa

    def arm_to_sqa(self, arm: Arm, weight: Optional[float] = 1.0) -> SQAArm:
        """Convert Ax Arm to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st... got `typing.Type[Arm]`.
        arm_class: SQAArm = self.config.class_to_sqa_class[Arm]
        # pyre-fixme[29]: `SQAArm` is not a function.
        return arm_class(
            id=arm.db_id, parameters=arm.parameters, name=arm._name, weight=weight
        )

    def abandoned_arm_to_sqa(self, abandoned_arm: AbandonedArm) -> SQAAbandonedArm:
        """Convert Ax AbandonedArm to SQLAlchemy."""
        # pyre-fixme[9]: abandoned_arm_class is ....sqa_store.db.SQABase]`.
        abandoned_arm_class: SQAAbandonedArm = self.config.class_to_sqa_class[
            AbandonedArm
        ]
        # pyre-fixme[29]: `SQAAbandonedArm` is not a function.
        return abandoned_arm_class(
            id=abandoned_arm.db_id,
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
        arms = []
        for arm, arm_weight in generator_run.arm_weights.items():
            arms.append(self.arm_to_sqa(arm=arm, weight=arm_weight))

        metrics = self.optimization_config_to_sqa(generator_run.optimization_config)
        parameters, parameter_constraints = self.search_space_to_sqa(
            generator_run.search_space
        )

        best_arm_name = None
        best_arm_parameters = None
        best_arm_predictions = None
        if generator_run.best_arm_predictions is not None:
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            best_arm = generator_run.best_arm_predictions[0]
            best_arm_predictions = list(generator_run.best_arm_predictions[1])
            best_arm_name = best_arm._name
            best_arm_parameters = best_arm.parameters
        model_predictions = (
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Optional[typing.Tuple[Dict[str, List[float]], Dict[str, Dict[str,
            #  List[float]]]]]`.
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
        gr_sqa = generator_run_class(  # pyre-ignore[29]: `SQAGeneratorRun` not a func.
            id=generator_run.db_id,
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
            model_key=generator_run._model_key,
            model_kwargs=object_to_json(generator_run._model_kwargs),
            bridge_kwargs=object_to_json(generator_run._bridge_kwargs),
            gen_metadata=object_to_json(generator_run._gen_metadata),
            model_state_after_gen=object_to_json(generator_run._model_state_after_gen),
            generation_step_index=generator_run._generation_step_index,
            candidate_metadata_by_arm_signature=object_to_json(
                generator_run._candidate_metadata_by_arm_signature
            ),
        )
        return gr_sqa

    def generation_strategy_to_sqa(
        self, generation_strategy: GenerationStrategy, experiment_id: Optional[int]
    ) -> SQAGenerationStrategy:
        """Convert an Ax `GenerationStrategy` to SQLAlchemy, preserving its state,
        so that the restored generation strategy can be resumed from the point
        at which it was interrupted and stored.
        """
        # pyre-ignore[9]: Expected Base, but redeclared to `SQAGenerationStrategy`.
        gs_class: SQAGenerationStrategy = self.config.class_to_sqa_class[
            cast(Type[Base], GenerationStrategy)
        ]
        generator_runs_sqa = []
        for gr in generation_strategy._generator_runs:
            gr_sqa = self.generator_run_to_sqa(gr)
            generator_runs_sqa.append(gr_sqa)

        # pyre-fixme[29]: `SQAGenerationStrategy` is not a function.
        gs_sqa = gs_class(
            id=generation_strategy.db_id,
            name=generation_strategy.name,
            steps=object_to_json(generation_strategy._steps),
            curr_index=generation_strategy._curr.index,
            generator_runs=generator_runs_sqa,
            experiment_id=experiment_id,
        )
        return gs_sqa

    def runner_to_sqa(
        self, runner: Runner, trial_type: Optional[str] = None
    ) -> SQARunner:
        """Convert Ax Runner to SQLAlchemy."""
        runner_class = type(runner)
        runner_type = RUNNER_REGISTRY.get(runner_class)
        if runner_type is None:
            raise SQAEncodeError(
                "Cannot encode runner to SQLAlchemy because runner's "
                f"subclass ({runner_class}) is missing from the registry. "
                "The runner registry currently contains the following: "
                f"{','.join(map(str, RUNNER_REGISTRY.keys()))}"
            )  # pragma: no cover
        properties = runner_class.serialize_init_args(runner=runner)

        # pyre-fixme: Expected `Base` for 1st...t `typing.Type[Runner]`.
        runner_class: SQARunner = self.config.class_to_sqa_class[Runner]
        # pyre-fixme[29]: `SQARunner` is not a function.
        return runner_class(
            id=runner.db_id,
            runner_type=runner_type,
            properties=properties,
            trial_type=trial_type,
        )

    def trial_to_sqa(self, trial: BaseTrial) -> SQATrial:
        """Convert Ax Trial to SQLAlchemy.

        In addition to creating and storing a new Trial object, we need to
        create and store the GeneratorRuns and Runner that it owns.
        """
        runner = None
        if trial.runner:
            runner = self.runner_to_sqa(runner=not_none(trial.runner))

        abandoned_arms = []
        generator_runs = []
        status_quo_name = None
        optimize_for_power = None

        if isinstance(trial, Trial) and trial.generator_run:
            gr_sqa = self.generator_run_to_sqa(
                generator_run=not_none(trial.generator_run)
            )
            generator_runs.append(gr_sqa)

        elif isinstance(trial, BatchTrial):
            for abandoned_arm in trial.abandoned_arms_metadata:
                abandoned_arms.append(
                    self.abandoned_arm_to_sqa(abandoned_arm=abandoned_arm)
                )
            for struct in trial.generator_run_structs:
                gr_sqa = self.generator_run_to_sqa(
                    generator_run=struct.generator_run, weight=struct.weight
                )
                generator_runs.append(gr_sqa)

            trial_status_quo = trial.status_quo
            trial_status_quo_weight_override = trial._status_quo_weight_override
            if (
                trial_status_quo is not None
                and trial_status_quo_weight_override is not None
            ):
                status_quo_generator_run = GeneratorRun(
                    arms=[trial_status_quo],
                    weights=[trial_status_quo_weight_override],
                    type=GeneratorRunType.STATUS_QUO.name,
                )
                # This is a hack necessary to get equality tests passing;
                # otherwise you can encode same object and get two different results.
                status_quo_generator_run._time_created = trial.time_created
                gr_sqa = self.generator_run_to_sqa(
                    generator_run=status_quo_generator_run
                )

                # pyre-ignore Attribute `id` declared in class `SQAGeneratorRun`
                # has type `int` but is used as type `Optional[int]`.
                gr_sqa.id = trial._status_quo_generator_run_db_id
                # pyre-ignore Attribute `id` declared in class `SQAArm`
                # has type `int` but is used as type `Optional[int]`.
                gr_sqa.arms[0].id = trial._status_quo_arm_db_id

                generator_runs.append(gr_sqa)
                status_quo_name = trial_status_quo.name

            optimize_for_power = getattr(trial, "optimize_for_power", None)
            if optimize_for_power is None:
                logger.warning(
                    f"optimize_for_power not present in BatchTrial: {trial.__dict__}"
                )

        # pyre-ignore[9]: Expected `Base` for 1st...ot `typing.Type[Trial]`.
        trial_class: SQATrial = self.config.class_to_sqa_class[Trial]
        trial_sqa = trial_class(  # pyre-fixme[29]: `SQATrial` is not a function.
            id=trial.db_id,
            abandoned_reason=trial.abandoned_reason,
            deployed_name=trial.deployed_name,
            index=trial.index,
            is_batch=isinstance(trial, BatchTrial),
            num_arms_created=trial._num_arms_created,
            optimize_for_power=optimize_for_power,
            ttl_seconds=trial.ttl_seconds,
            run_metadata=trial.run_metadata,
            stop_metadata=trial.stop_metadata,
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
            generation_step_index=trial._generation_step_index,
            properties=trial._properties,
        )
        return trial_sqa

    def data_to_sqa(
        self, data: AbstractDataFrameData, trial_index: Optional[int], timestamp: int
    ) -> SQAData:
        """Convert Ax data to SQLAlchemy."""
        # pyre-fixme: Expected `Base` for 1st...ot `typing.Type[Data]`.
        data_class: SQAData = self.config.class_to_sqa_class[Data]
        import json

        # pyre-fixme[29]: `SQAData` is not a function.
        return data_class(
            id=data.db_id,
            data_json=data.df.to_json(),
            description=data.description,
            time_created=timestamp,
            trial_index=trial_index,
            structure_metadata_json=json.dumps(
                object_to_json(data.serialize_init_args(data))
            ),
        )
