#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from enum import Enum
from io import StringIO
from logging import Logger
from typing import Any, Callable, cast, Union

import pandas as pd
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckAnalysisCard
from ax.analysis.markdown.markdown_analysis import MarkdownAnalysisCard
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.core.analysis_card import (
    AnalysisCard,
    AnalysisCardBase,
    AnalysisCardGroup,
    ErrorAnalysisCard,
)
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import AbandonedArm, BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import (
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.exceptions.storage import JSONDecodeError, SQADecodeError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.storage.json_store.decoder import _DEPRECATED_GENERATOR_KWARGS, object_from_json
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.sqa_classes import (
    SQAAbandonedArm,
    SQAAnalysisCard,
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
from ax.storage.sqa_store.utils import are_relationships_loaded
from ax.storage.utils import (
    data_by_trial_to_data,
    DomainType,
    EXPECT_RELATIVIZED_OUTCOMES,
    MetricIntent,
    PREFERENCE_PROFILE_NAME,
)
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pandas import read_json
from pyre_extensions import assert_is_instance, none_throws
from sqlalchemy.orm.exc import DetachedInstanceError

logger: Logger = get_logger(__name__)


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
        self, value: int | None, enum: Enum | type[Enum] | None
    ) -> str | None:
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

    def _auxiliary_experiments_by_purpose_from_experiment_sqa(
        self, experiment_sqa: SQAExperiment, reduced_state: bool = False
    ) -> dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]] | None:
        auxiliary_experiments_by_purpose = {}

        # Legacy logic
        if experiment_sqa.auxiliary_experiments_by_purpose:
            aux_exps_dict = none_throws(experiment_sqa.auxiliary_experiments_by_purpose)
            for aux_exp_purpose_str, aux_exps_json in aux_exps_dict.items():
                aux_exp_purpose = next(
                    member
                    for member in self.config.auxiliary_experiment_purpose_enum
                    if member.value == aux_exp_purpose_str
                )
                if aux_exp_purpose not in auxiliary_experiments_by_purpose:
                    auxiliary_experiments_by_purpose[aux_exp_purpose] = []
                for aux_exp_json in aux_exps_json:
                    # keeping this for backward compatibility since previously
                    # we used to save only the experiment name
                    if isinstance(aux_exp_json, str):
                        aux_exp_json = {"experiment_name": aux_exp_json}
                    aux_experiment = auxiliary_experiment_from_name(
                        experiment_name=aux_exp_json["experiment_name"],
                        config=self.config,
                        is_active=True,
                        reduced_state=reduced_state,
                    )
                    auxiliary_experiments_by_purpose[aux_exp_purpose].append(
                        aux_experiment
                    )

        # New logic
        if experiment_sqa.auxiliary_experiments:
            for auxiliary_experiment_sqa in experiment_sqa.auxiliary_experiments:
                purpose = self.config.auxiliary_experiment_purpose_enum(
                    auxiliary_experiment_sqa.purpose
                )
                if purpose not in auxiliary_experiments_by_purpose:
                    auxiliary_experiments_by_purpose[purpose] = []
                # If the auxiliary experiment is already loaded, we don't need to
                # load it again.
                if any(
                    auxiliary_experiment_sqa.source_experiment_id
                    == aux_exp.experiment.db_id
                    for aux_exp in auxiliary_experiments_by_purpose[purpose]
                ):
                    continue
                aux_experiment = auxiliary_experiment_from_name(
                    experiment_name=auxiliary_experiment_sqa.source_experiment.name,
                    config=self.config,
                    is_active=auxiliary_experiment_sqa.is_active,
                    reduced_state=reduced_state,
                )
                auxiliary_experiments_by_purpose[purpose].append(aux_experiment)
        return auxiliary_experiments_by_purpose

    # TODO[@mpolson64]: Stop storing target arm in experiment properties
    # as part of the storage refactor.
    def _get_pruning_target_parameterization_from_experiment_properties(
        self, properties: dict[str, Any]
    ) -> Arm | None:
        pruning_target_parameterization = properties.pop(
            "pruning_target_parameterization", None
        )
        if pruning_target_parameterization is not None:
            pruning_target_parameterization = assert_is_instance(
                object_from_json(object_json=pruning_target_parameterization), Arm
            )
        return pruning_target_parameterization

    def _init_experiment_from_sqa(
        self,
        experiment_sqa: SQAExperiment,
        load_auxiliary_experiments: bool = True,
        reduced_state: bool = False,
    ) -> Experiment:
        """First step of conversion within experiment_from_sqa."""
        # `experiment_sqa.properties` is `sqlalchemy.ext.mutable.MutableDict`
        # so need to convert it to regular dict.
        properties = dict(experiment_sqa.properties or {})
        opt_config, tracking_metrics = self.opt_config_and_tracking_metrics_from_sqa(
            metrics_sqa=experiment_sqa.metrics,
            pruning_target_parameterization=(
                self._get_pruning_target_parameterization_from_experiment_properties(
                    properties=properties
                )
            ),
        )
        search_space = self.search_space_from_sqa(
            parameters_sqa=experiment_sqa.parameters,
            parameter_constraints_sqa=experiment_sqa.parameter_constraints,
        )
        if search_space is None:
            raise SQADecodeError("Experiment SearchSpace cannot be None.")
        status_quo = (
            Arm(
                parameters=experiment_sqa.status_quo_parameters,
                name=experiment_sqa.status_quo_name,
            )
            if experiment_sqa.status_quo_parameters is not None
            else None
        )
        if len(experiment_sqa.runners) == 0:
            runner = None
        elif len(experiment_sqa.runners) == 1:
            runner = self.runner_from_sqa(runner_sqa=experiment_sqa.runners[0])
        else:
            raise ValueError(
                "Multiple runners on experiment only supported for MultiTypeExperiment."
            )

        auxiliary_experiments_by_purpose = (
            (
                self._auxiliary_experiments_by_purpose_from_experiment_sqa(
                    experiment_sqa=experiment_sqa,
                    reduced_state=reduced_state,
                )
            )
            if load_auxiliary_experiments
            else {}
        )

        return Experiment(
            name=experiment_sqa.name,
            description=experiment_sqa.description,
            search_space=search_space,
            optimization_config=opt_config,
            tracking_metrics=tracking_metrics,
            runner=runner,
            status_quo=status_quo,
            is_test=experiment_sqa.is_test,
            properties=properties,
            auxiliary_experiments_by_purpose=auxiliary_experiments_by_purpose,
        )

    def _init_mt_experiment_from_sqa(
        self,
        experiment_sqa: SQAExperiment,
    ) -> MultiTypeExperiment:
        """First step of conversion within experiment_from_sqa."""
        properties = dict(experiment_sqa.properties or {})
        opt_config, tracking_metrics = self.opt_config_and_tracking_metrics_from_sqa(
            metrics_sqa=experiment_sqa.metrics,
            pruning_target_parameterization=(
                self._get_pruning_target_parameterization_from_experiment_properties(
                    properties=properties
                )
            ),
        )
        search_space = self.search_space_from_sqa(
            parameters_sqa=experiment_sqa.parameters,
            parameter_constraints_sqa=experiment_sqa.parameter_constraints,
        )
        if search_space is None:
            raise SQADecodeError("Experiment SearchSpace cannot be None.")
        status_quo = (
            Arm(
                parameters=experiment_sqa.status_quo_parameters,
                name=experiment_sqa.status_quo_name,
            )
            if experiment_sqa.status_quo_parameters is not None
            else None
        )

        default_trial_type = none_throws(experiment_sqa.default_trial_type)
        trial_type_to_runner = {
            none_throws(sqa_runner.trial_type): self.runner_from_sqa(sqa_runner)
            for sqa_runner in experiment_sqa.runners
        }
        if len(trial_type_to_runner) == 0:
            trial_type_to_runner = {default_trial_type: None}
            trial_types_with_metrics = {
                metric.trial_type
                for metric in experiment_sqa.metrics
                if metric.trial_type
            }
            # trial_type_to_runner is instantiated to map all trial types to None,
            # so the trial types are associated with the experiment. This is
            # important for adding metrics.
            trial_type_to_runner.update(dict.fromkeys(trial_types_with_metrics))

        experiment = MultiTypeExperiment(
            name=experiment_sqa.name,
            description=experiment_sqa.description,
            search_space=search_space,
            default_trial_type=default_trial_type,
            default_runner=trial_type_to_runner.get(default_trial_type),
            optimization_config=opt_config,
            status_quo=status_quo,
            properties=properties,
        )
        # pyre-ignore Imcompatible attribute type [8]: attribute _trial_type_to_runner
        # has type Dict[str, Optional[Runner]] but is used as type
        # Uniont[Dict[str, Optional[Runner]], Dict[str, None]]
        experiment._trial_type_to_runner = trial_type_to_runner
        sqa_metric_dict = {metric.name: metric for metric in experiment_sqa.metrics}
        for tracking_metric in tracking_metrics:
            sqa_metric = sqa_metric_dict[tracking_metric.name]
            experiment.add_tracking_metric(
                tracking_metric,
                trial_type=none_throws(sqa_metric.trial_type),
                canonical_name=sqa_metric.canonical_name,
            )
        return experiment

    def experiment_from_sqa(
        self,
        experiment_sqa: SQAExperiment,
        reduced_state: bool = False,
        load_auxiliary_experiments: bool = True,
    ) -> Experiment:
        """Convert SQLAlchemy Experiment to Ax Experiment.

        Args:
            experiment_sqa: `SQAExperiment` to decode.
            reduced_state: Whether to load experiment with a slightly reduced state
                (without abandoned arms on experiment and without model state,
                search space, and optimization config on generator runs).
            load_auxiliary_experiment: whether to load auxiliary experiments.
        """
        subclass = (experiment_sqa.properties or {}).get(Keys.SUBCLASS)
        if subclass == "MultiTypeExperiment":
            experiment = self._init_mt_experiment_from_sqa(experiment_sqa)
        else:
            experiment = self._init_experiment_from_sqa(
                experiment_sqa,
                load_auxiliary_experiments=load_auxiliary_experiments,
                reduced_state=reduced_state,
            )
        trials = [
            self.trial_from_sqa(
                trial_sqa=trial,
                experiment=experiment,
                reduced_state=reduced_state,
            )
            for trial in experiment_sqa.trials
        ]

        data_by_trial = defaultdict(dict)
        for data_sqa in experiment_sqa.data:
            trial_index = data_sqa.trial_index
            timestamp = data_sqa.time_created
            data_by_trial[trial_index][timestamp] = self.data_from_sqa(
                data_sqa=data_sqa
            )
        experiment.data = data_by_trial_to_data(data_by_trial=data_by_trial)

        trial_type_to_runner = {
            sqa_runner.trial_type: self.runner_from_sqa(sqa_runner)
            for sqa_runner in experiment_sqa.runners
        }
        if len(trial_type_to_runner) == 0:
            trial_type_to_runner = {None: None}

        experiment._trials = {trial.index: trial for trial in trials}
        experiment._arms_by_name = {}
        for trial in trials:
            if trial.ttl_seconds is not None:
                experiment._trials_have_ttl = True
            for arm in trial.arms:
                experiment._register_arm(arm)
        if experiment.status_quo is not None:
            sq = none_throws(experiment.status_quo)
            experiment._register_arm(sq)
        experiment._time_created = experiment_sqa.time_created
        experiment._experiment_type = self.get_enum_name(
            value=experiment_sqa.experiment_type, enum=self.config.experiment_type_enum
        )
        # `_trial_type_to_runner` is set in _init_mt_experiment_from_sqa
        if subclass != "MultiTypeExperiment":
            experiment._trial_type_to_runner = cast(
                dict[str | None, Runner | None], trial_type_to_runner
            )
        experiment.db_id = experiment_sqa.id
        return experiment

    def parameter_from_sqa(self, parameter_sqa: SQAParameter) -> Parameter:
        """Convert SQLAlchemy Parameter to Ax Parameter."""
        if parameter_sqa.domain_type == DomainType.RANGE:
            if parameter_sqa.lower is None or parameter_sqa.upper is None:
                raise SQADecodeError(
                    "`lower` and `upper` must be set for RangeParameter; one or both "
                    f"not found on parameter {parameter_sqa.name}."
                )
            if parameter_sqa.dependents is not None:
                raise SQADecodeError(
                    "`dependents` unexpectedly non-null on range parameter "
                    f"{parameter_sqa.name}."
                )
            parameter = RangeParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                lower=float(none_throws(parameter_sqa.lower)),
                upper=float(none_throws(parameter_sqa.upper)),
                log_scale=parameter_sqa.log_scale or False,
                digits=parameter_sqa.digits,
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
                backfill_value=parameter_sqa.backfill_value,
                default_value=parameter_sqa.default_value,
            )
        elif parameter_sqa.domain_type == DomainType.CHOICE:
            target_value = parameter_sqa.target_value
            if parameter_sqa.choice_values is None:
                raise SQADecodeError(
                    "`values` must be set for ChoiceParameter; not found on"
                    f" parameter {parameter_sqa.name}."
                )
            if bool(parameter_sqa.is_task) and target_value is None:
                target_value = none_throws(parameter_sqa.choice_values)[0]
                logger.debug(
                    f"Target value is null for parameter {parameter_sqa.name}. "
                    f"Defaulting to first choice {target_value}."
                )
            parameter = ChoiceParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                values=none_throws(parameter_sqa.choice_values),
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=target_value,
                is_ordered=parameter_sqa.is_ordered,
                is_task=bool(parameter_sqa.is_task),
                log_scale=parameter_sqa.log_scale,
                dependents=parameter_sqa.dependents,
                backfill_value=parameter_sqa.backfill_value,
                default_value=parameter_sqa.default_value,
            )
        elif parameter_sqa.domain_type == DomainType.FIXED:
            # Don't throw an error if parameter_sqa.fixed_value is None;
            # that might be the actual value!
            parameter = FixedParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                value=parameter_sqa.fixed_value,
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
                dependents=parameter_sqa.dependents,
                backfill_value=parameter_sqa.backfill_value,
                default_value=parameter_sqa.default_value,
            )
        elif parameter_sqa.domain_type == DomainType.DERIVED:
            parameter = DerivedParameter(
                name=parameter_sqa.name,
                parameter_type=parameter_sqa.parameter_type,
                expression_str=none_throws(parameter_sqa.expression_str),
                is_fidelity=parameter_sqa.is_fidelity or False,
                target_value=parameter_sqa.target_value,
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAParameter because {parameter_sqa.domain_type} "
                "is an invalid domain type."
            )

        parameter.db_id = parameter_sqa.id
        return parameter

    def parameter_constraint_from_sqa(
        self,
        parameter_constraint_sqa: SQAParameterConstraint,
        parameters: list[Parameter],
    ) -> ParameterConstraint:
        """Convert SQLAlchemy ParameterConstraint to Ax ParameterConstraint."""
        if len(parameter_constraint_sqa.constraint_dict) == 0:
            raise SQADecodeError(
                "ParameterConstraint must have at least one parameter in its "
                f"constraint_dict; found 0 on {parameter_constraint_sqa.id}."
            )

        expr = " + ".join(
            f"{coeff} * {param}"
            for param, coeff in parameter_constraint_sqa.constraint_dict.items()
        )
        constraint = ParameterConstraint(
            inequality=f"{expr} <= {parameter_constraint_sqa.bound}",
        )

        constraint.db_id = parameter_constraint_sqa.id
        return constraint

    def search_space_from_sqa(
        self,
        parameters_sqa: list[SQAParameter],
        parameter_constraints_sqa: list[SQAParameterConstraint],
    ) -> SearchSpace | None:
        """Convert a list of SQLAlchemy Parameters and ParameterConstraints to an
        Ax SearchSpace.
        """
        parameters = []
        for parameter_sqa in parameters_sqa:
            parameters.append(self.parameter_from_sqa(parameter_sqa=parameter_sqa))

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

    def metric_from_sqa(
        self, metric_sqa: SQAMetric
    ) -> Metric | Objective | OutcomeConstraint:
        """Convert SQLAlchemy Metric to Ax Metric, Objective, or OutcomeConstraint."""

        metric = self._metric_from_sqa_util(metric_sqa)

        if metric_sqa.intent == MetricIntent.TRACKING:
            return metric
        elif metric_sqa.intent == MetricIntent.OBJECTIVE:
            return self._objective_from_sqa(metric=metric, metric_sqa=metric_sqa)
        elif (
            metric_sqa.intent == MetricIntent.MULTI_OBJECTIVE
            # metric_sqa is a parent whose children are individual
            # metrics in MultiObjective
            or metric_sqa.intent == MetricIntent.PREFERENCE_OBJECTIVE
            # PREFERENCE_OBJECTIVE stores a MultiObjective, similar to
            # MULTI_OBJECTIVE. The config-level properties
            # (preference_profile_name, expect_relativized_outcomes) are stored
            # in the parent metric's properties field and are extracted in
            # opt_config_and_tracking_metrics_from_sqa to create the full
            # PreferenceOptimizationConfig.
        ):
            return self._multi_objective_from_sqa(parent_metric_sqa=metric_sqa)
        elif (
            metric_sqa.intent == MetricIntent.SCALARIZED_OBJECTIVE
        ):  # metric_sqa is a parent whose children are individual
            # metrics in Scalarized Objective
            return self._scalarized_objective_from_sqa(parent_metric_sqa=metric_sqa)
        elif metric_sqa.intent == MetricIntent.OUTCOME_CONSTRAINT:
            return self._outcome_constraint_from_sqa(
                metric=metric, metric_sqa=metric_sqa
            )
        elif metric_sqa.intent == MetricIntent.SCALARIZED_OUTCOME_CONSTRAINT:
            return self._scalarized_outcome_constraint_from_sqa(
                metric=metric, metric_sqa=metric_sqa
            )
        elif metric_sqa.intent == MetricIntent.OBJECTIVE_THRESHOLD:
            return self._objective_threshold_from_sqa(
                metric=metric, metric_sqa=metric_sqa
            )
        else:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.intent} "
                f"is an invalid intent."
            )

    def opt_config_and_tracking_metrics_from_sqa(
        self, metrics_sqa: list[SQAMetric], pruning_target_parameterization: Arm | None
    ) -> tuple[OptimizationConfig | None, list[Metric]]:
        """Convert a list of SQLAlchemy Metrics to Ax OptimizationConfig
        and tracking metrics.
        """
        objective = None
        objective_thresholds = []
        outcome_constraints = []
        tracking_metrics = []
        preference_objective_sqa = None

        for metric_sqa in metrics_sqa:
            if metric_sqa.intent == MetricIntent.PREFERENCE_OBJECTIVE:
                preference_objective_sqa = metric_sqa

            metric = self.metric_from_sqa(metric_sqa=metric_sqa)
            if isinstance(metric, Objective):
                objective = metric
            elif isinstance(metric, ObjectiveThreshold):
                objective_thresholds.append(metric)
            elif isinstance(metric, OutcomeConstraint):
                outcome_constraints.append(metric)
            else:
                tracking_metrics.append(metric)

        if objective is None:
            return None, tracking_metrics

        if preference_objective_sqa is not None:
            if objective_thresholds:
                raise SQADecodeError(
                    "PreferenceOptimizationConfig cannot have objective thresholds."
                )
            properties = preference_objective_sqa.properties or {}
            optimization_config = PreferenceOptimizationConfig(
                objective=assert_is_instance(objective, MultiObjective),
                preference_profile_name=properties.get(PREFERENCE_PROFILE_NAME, ""),
                expect_relativized_outcomes=properties.get(
                    EXPECT_RELATIVIZED_OUTCOMES, False
                ),
                outcome_constraints=outcome_constraints,
                pruning_target_parameterization=pruning_target_parameterization,
            )
        elif objective_thresholds or type(objective) is MultiObjective:
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=assert_is_instance(
                    objective, Union[MultiObjective, ScalarizedObjective]
                ),
                outcome_constraints=outcome_constraints,
                objective_thresholds=objective_thresholds,
                pruning_target_parameterization=pruning_target_parameterization,
            )
        else:
            optimization_config = OptimizationConfig(
                objective=objective,
                outcome_constraints=outcome_constraints,
                pruning_target_parameterization=pruning_target_parameterization,
            )
        return (optimization_config, tracking_metrics)

    def arm_from_sqa(self, arm_sqa: SQAArm) -> Arm:
        """Convert SQLAlchemy Arm to Ax Arm."""
        arm = Arm(parameters=arm_sqa.parameters, name=arm_sqa.name)
        arm.db_id = arm_sqa.id
        return arm

    def abandoned_arm_from_sqa(
        self, abandoned_arm_sqa: SQAAbandonedArm
    ) -> AbandonedArm:
        """Convert SQLAlchemy AbandonedArm to Ax AbandonedArm."""
        arm = AbandonedArm(
            name=abandoned_arm_sqa.name,
            reason=abandoned_arm_sqa.abandoned_reason,
            time=abandoned_arm_sqa.time_abandoned,
        )
        arm.db_id = abandoned_arm_sqa.id
        return arm

    def generator_run_from_sqa(
        self,
        generator_run_sqa: SQAGeneratorRun,
        reduced_state: bool,
        immutable_search_space_and_opt_config: bool,
    ) -> GeneratorRun:
        """Convert SQLAlchemy GeneratorRun to Ax GeneratorRun.

        Args:
            generator_run_sqa: `SQAGeneratorRun` to decode.
            reduced_state: Whether to load generator runs with a slightly reduced state
                (without model state, search space, and optimization config).
            immutable_search_space_and_opt_config: Whether to load generator runs
                without search space and optimization config. Unlike `reduced_state`,
                we do still load model state.
        """
        arms = []
        weights = []
        opt_config = None
        search_space = None

        for arm_sqa in generator_run_sqa.arms:
            arms.append(self.arm_from_sqa(arm_sqa=arm_sqa))
            weights.append(arm_sqa.weight)
        if not reduced_state and not immutable_search_space_and_opt_config:
            # Check if metrics, parameters, and parameter constraints are present
            # on the generator run SQA object, since these attributes
            # were potentially lazy loaded.
            if are_relationships_loaded(
                sqa_object=generator_run_sqa,
                relationship_names=["metrics", "parameters", "parameter_constraints"],
            ):
                (
                    opt_config,
                    tracking_metrics,
                ) = self.opt_config_and_tracking_metrics_from_sqa(
                    metrics_sqa=generator_run_sqa.metrics,
                    pruning_target_parameterization=None,
                )
                if len(tracking_metrics) > 0:
                    raise SQADecodeError(
                        "GeneratorRun should not have tracking metrics."
                    )

                search_space = self.search_space_from_sqa(
                    parameters_sqa=generator_run_sqa.parameters,
                    parameter_constraints_sqa=generator_run_sqa.parameter_constraints,
                )
            else:
                opt_config = None
                search_space = None

        best_arm_predictions = None
        model_predictions = None
        if (
            generator_run_sqa.best_arm_parameters is not None
            and generator_run_sqa.best_arm_predictions is not None
        ):
            best_arm = Arm(
                name=generator_run_sqa.best_arm_name,
                parameters=none_throws(generator_run_sqa.best_arm_parameters),
            )
            best_arm_predictions = (
                best_arm,
                tuple(none_throws(generator_run_sqa.best_arm_predictions)),
            )
        model_predictions = (
            tuple(none_throws(generator_run_sqa.model_predictions))
            if generator_run_sqa.model_predictions is not None
            else None
        )

        generator_run = GeneratorRun(
            arms=arms,
            weights=weights,
            optimization_config=opt_config,
            search_space=search_space,
            fit_time=(
                None
                if generator_run_sqa.fit_time is None
                else float(generator_run_sqa.fit_time)
            ),
            gen_time=(
                None
                if generator_run_sqa.gen_time is None
                else float(generator_run_sqa.gen_time)
            ),
            best_arm_predictions=best_arm_predictions,  # pyre-ignore[6]
            # pyre-fixme[6]: Expected `Optional[Tuple[typing.Dict[str, List[float]],
            #  typing.Dict[str, typing.Dict[str, List[float]]]]]` for 8th param but got
            #  `Optional[typing.Tuple[Union[typing.Dict[str, List[float]],
            #  typing.Dict[str, typing.Dict[str, List[float]]]], ...]]`.
            model_predictions=model_predictions,
            generator_key=generator_run_sqa.model_key,
            generator_kwargs=(
                None
                if reduced_state
                else object_from_json(
                    generator_run_sqa.model_kwargs,
                    decoder_registry=self.config.json_decoder_registry,
                    class_decoder_registry=self.config.json_class_decoder_registry,
                )
            ),
            adapter_kwargs=(
                None
                if reduced_state
                else object_from_json(
                    generator_run_sqa.bridge_kwargs,
                    decoder_registry=self.config.json_decoder_registry,
                    class_decoder_registry=self.config.json_class_decoder_registry,
                )
            ),
            gen_metadata=(
                None
                if reduced_state
                else object_from_json(
                    generator_run_sqa.gen_metadata,
                    decoder_registry=self.config.json_decoder_registry,
                    class_decoder_registry=self.config.json_class_decoder_registry,
                )
            ),
            generator_state_after_gen=(
                None
                if reduced_state
                else object_from_json(
                    generator_run_sqa.model_state_after_gen,
                    decoder_registry=self.config.json_decoder_registry,
                    class_decoder_registry=self.config.json_class_decoder_registry,
                )
            ),
            candidate_metadata_by_arm_signature=object_from_json(
                generator_run_sqa.candidate_metadata_by_arm_signature,
                decoder_registry=self.config.json_decoder_registry,
                class_decoder_registry=self.config.json_class_decoder_registry,
            ),
            generation_node_name=generator_run_sqa.generation_node_name,
        )
        # Remove deprecated kwargs from generator kwargs & adapter kwargs.
        if generator_run._generator_kwargs is not None:
            generator_run._generator_kwargs = {
                k: v
                for k, v in generator_run._generator_kwargs.items()
                if k not in _DEPRECATED_GENERATOR_KWARGS
            }
        if generator_run._adapter_kwargs is not None:
            generator_run._adapter_kwargs = {
                k: v
                for k, v in generator_run._adapter_kwargs.items()
                if k not in _DEPRECATED_GENERATOR_KWARGS
            }
        generator_run._time_created = generator_run_sqa.time_created
        generator_run._generator_run_type = self.get_enum_name(
            value=generator_run_sqa.generator_run_type,
            enum=self.config.generator_run_type_enum,
        )
        generator_run.db_id = generator_run_sqa.id
        return generator_run

    def generation_strategy_from_sqa(
        self,
        gs_sqa: SQAGenerationStrategy,
        experiment: Experiment | None = None,
        reduced_state: bool = False,
    ) -> GenerationStrategy:
        """Convert SQALchemy generation strategy to Ax `GenerationStrategy`."""
        if len(gs_sqa.generator_runs) and experiment is None:
            raise SQADecodeError(
                "Cannot decode a generation strategy with a non-zero number of "
                "generator runs without an experiment."
            )
        if gs_sqa.nodes is None:  # Backward compat. for pre-nodes strategies.
            nodes = object_from_json(
                gs_sqa.steps,
                decoder_registry=self.config.json_decoder_registry,
                class_decoder_registry=self.config.json_class_decoder_registry,
            )
        else:
            nodes = object_from_json(
                gs_sqa.nodes,
                decoder_registry=self.config.json_decoder_registry,
                class_decoder_registry=self.config.json_class_decoder_registry,
            )
        gs = GenerationStrategy(name=gs_sqa.name, nodes=nodes)
        curr_node_name = gs_sqa.curr_node_name
        for node in gs._nodes:
            if node.name == curr_node_name:
                gs._curr = node
                break
        immutable_ss_and_oc = (
            experiment.immutable_search_space_and_opt_config if experiment else False
        )

        gs._generator_runs = [
            self.generator_run_from_sqa(
                generator_run_sqa=gr,
                reduced_state=reduced_state,
                immutable_search_space_and_opt_config=immutable_ss_and_oc,
            )
            for gr in gs_sqa.generator_runs[:-1]
        ]
        # This check is necessary to prevent an index error
        # on `gs_sqa.generator_runs[-1]`
        if gs_sqa.generator_runs:
            # Only fully load the last of the generator runs, load the rest with
            # reduced state.  This is necessary for stateful models.  The only
            # stateful models available in open source ax is currently SOBOL.
            try:
                gs._generator_runs.append(
                    self.generator_run_from_sqa(
                        generator_run_sqa=gs_sqa.generator_runs[-1],
                        reduced_state=False,
                        # We set immutable_search_space_and_opt_config to reduced_state
                        # to ensure that metrics and parameters are not loaded as part
                        # of the generator run, as they are not required for a
                        # generator run to be "fully" loaded.
                        immutable_search_space_and_opt_config=reduced_state,
                    )
                )
            except JSONDecodeError:
                if not reduced_state:
                    raise

                logger.exception(
                    "Failed to decode the last generator run because of the following "
                    "error.  Loading with reduced state:"
                )
                # If the last generator run is not fully loadable, load it with
                # reduced state.
                gs._generator_runs.append(
                    self.generator_run_from_sqa(
                        generator_run_sqa=gs_sqa.generator_runs[-1],
                        reduced_state=True,
                        immutable_search_space_and_opt_config=immutable_ss_and_oc,
                    )
                )
        gs._experiment = experiment
        gs.db_id = gs_sqa.id
        return gs

    def runner_from_sqa(self, runner_sqa: SQARunner) -> Runner:
        """Convert SQLAlchemy Runner to Ax Runner."""
        if runner_sqa.runner_type not in self.config.reverse_runner_registry:
            raise SQADecodeError(
                f"Cannot decode SQARunner because {runner_sqa.runner_type} "
                f"is an invalid type. "
            )
        runner_class = self.config.reverse_runner_registry[runner_sqa.runner_type]

        args = runner_class.deserialize_init_args(
            args=dict(runner_sqa.properties or {}),
            decoder_registry=self.config.json_decoder_registry,
            class_decoder_registry=self.config.json_class_decoder_registry,
        )
        # pyre-ignore[45]: Cannot instantiate abstract class `Runner`.
        runner = runner_class(**args)
        runner.db_id = runner_sqa.id
        return runner

    def trial_from_sqa(
        self,
        trial_sqa: SQATrial,
        experiment: Experiment,
        reduced_state: bool = False,
    ) -> BaseTrial:
        """Convert SQLAlchemy Trial to Ax Trial.

        Args:
            trial_sqa: `SQATrial` to decode.
            reduced_state: Whether to load trial's generator run(s) with a slightly
            reduced state (without model state, search space, and optimization config).

        """
        immutable_ss_and_oc = experiment.immutable_search_space_and_opt_config
        if trial_sqa.is_batch:
            trial = BatchTrial(
                experiment=experiment,
                ttl_seconds=trial_sqa.ttl_seconds,
                index=trial_sqa.index,
            )
            generator_runs = [
                self.generator_run_from_sqa(
                    generator_run_sqa=generator_run_sqa,
                    reduced_state=reduced_state,
                    immutable_search_space_and_opt_config=immutable_ss_and_oc,
                )
                for generator_run_sqa in trial_sqa.generator_runs
            ]
            trial._generator_runs = generator_runs
            trial._abandoned_arms_metadata = {
                abandoned_arm_sqa.name: self.abandoned_arm_from_sqa(
                    abandoned_arm_sqa=abandoned_arm_sqa
                )
                for abandoned_arm_sqa in trial_sqa.abandoned_arms
            }
            trial._refresh_arms_by_name()  # Trigger cache build

            # Trial.arms_by_name only returns arms with weights
            trial.should_add_status_quo_arm = (
                trial.status_quo is not None
                and trial.status_quo.name in trial.arms_by_name
            )

        else:
            trial = Trial(
                experiment=experiment,
                ttl_seconds=trial_sqa.ttl_seconds,
                index=trial_sqa.index,
            )
            if trial_sqa.generator_runs:
                if len(trial_sqa.generator_runs) != 1:
                    raise SQADecodeError(
                        "Cannot decode SQATrial to Trial because trial is not batched "
                        "but has more than one generator run."
                    )
                trial._generator_run = self.generator_run_from_sqa(
                    generator_run_sqa=trial_sqa.generator_runs[0],
                    reduced_state=reduced_state,
                    immutable_search_space_and_opt_config=immutable_ss_and_oc,
                )
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
        trial._status_reason = trial_sqa.abandoned_reason or trial_sqa.failed_reason
        # pyre-fixme[9]: _run_metadata has type `Dict[str, Any]`; used as
        #  `Optional[Dict[str, Any]]`.
        # pyre-fixme[8]: Attribute has type `Dict[str, typing.Any]`; used as
        #  `Optional[typing.Dict[Variable[_KT], Variable[_VT]]]`.
        trial._run_metadata = (
            dict(trial_sqa.run_metadata) if trial_sqa.run_metadata is not None else None
        )
        # pyre-fixme[9]: _run_metadata has type `Dict[str, Any]`; used as
        #  `Optional[Dict[str, Any]]`.
        # pyre-fixme[8]: Attribute has type `Dict[str, typing.Any]`; used as
        #  `Optional[typing.Dict[Variable[_KT], Variable[_VT]]]`.
        trial._stop_metadata = (
            dict(trial_sqa.stop_metadata)
            if trial_sqa.stop_metadata is not None
            else None
        )
        trial._num_arms_created = trial_sqa.num_arms_created
        trial._properties = dict(trial_sqa.properties or {})
        trial.db_id = trial_sqa.id
        return trial

    def data_from_sqa(self, data_sqa: SQAData) -> Data:
        """Convert SQLAlchemy Data to Ax Data."""
        # Override df from deserialize_init_args with `data_json`.
        # NOTE: Need dtype=False, otherwise infers arm_names like
        # "4_1" should be int 41.
        df = pd.read_json(StringIO(data_sqa.data_json), dtype=False)
        if "metric_signature" not in df.columns:
            df["metric_signature"] = df["metric_name"]

        dat = Data(df=df)
        dat.db_id = data_sqa.id
        return dat

    def analysis_card_from_sqa(
        self,
        analysis_card_sqa: SQAAnalysisCard,
    ) -> AnalysisCardBase:
        """Convert SQLAlchemy AnalysisCard to Ax AnalysisCard."""
        children = analysis_card_sqa.children

        if len(children) > 0:
            # Decode children and collect index
            index_to_child_card = {
                child.order: self.analysis_card_from_sqa(analysis_card_sqa=child)
                for child in children
            }

            # Sort children by index
            children = [card for _order, card in sorted(index_to_child_card.items())]

            # Convert None value of title to empty string to ensure compatibility with
            # AnalysisCardGroup constructor. Subtitle can be None.
            title = (
                analysis_card_sqa.title if analysis_card_sqa.title is not None else ""
            )
            subtitle = analysis_card_sqa.subtitle

            return AnalysisCardGroup(
                name=analysis_card_sqa.name,
                title=title,
                subtitle=subtitle,
                children=children,
                timestamp=analysis_card_sqa.timestamp,
            )

        title = none_throws(analysis_card_sqa.title)
        subtitle = none_throws(analysis_card_sqa.subtitle)
        blob = none_throws(analysis_card_sqa.blob)
        blob_annotation = analysis_card_sqa.blob_annotation

        if blob_annotation == "error":
            return ErrorAnalysisCard(
                name=analysis_card_sqa.name,
                title=title,
                subtitle=subtitle,
                df=read_json(analysis_card_sqa.dataframe_json),
                blob=blob,
                timestamp=analysis_card_sqa.timestamp,
            )
        if blob_annotation == "plotly":
            return PlotlyAnalysisCard(
                name=analysis_card_sqa.name,
                title=title,
                subtitle=subtitle,
                df=read_json(analysis_card_sqa.dataframe_json),
                blob=blob,
                timestamp=analysis_card_sqa.timestamp,
            )
        if blob_annotation == "markdown":
            return MarkdownAnalysisCard(
                name=analysis_card_sqa.name,
                title=title,
                subtitle=subtitle,
                df=read_json(analysis_card_sqa.dataframe_json),
                blob=blob,
                timestamp=analysis_card_sqa.timestamp,
            )
        if blob_annotation == "healthcheck":
            return HealthcheckAnalysisCard(
                name=analysis_card_sqa.name,
                title=title,
                subtitle=subtitle,
                df=read_json(analysis_card_sqa.dataframe_json),
                blob=blob,
                timestamp=analysis_card_sqa.timestamp,
            )
        return AnalysisCard(
            name=analysis_card_sqa.name,
            title=title,
            subtitle=subtitle,
            df=read_json(analysis_card_sqa.dataframe_json),
            blob=blob,
            timestamp=analysis_card_sqa.timestamp,
        )

    def _metric_from_sqa_util(self, metric_sqa: SQAMetric) -> Metric:
        """Convert SQLAlchemy Metric to Ax Metric"""
        if metric_sqa.metric_type not in self.config.reverse_metric_registry:
            raise SQADecodeError(
                f"Cannot decode SQAMetric because {metric_sqa.metric_type} "
                f"is an invalid type. "
            )
        metric_class = self.config.reverse_metric_registry[metric_sqa.metric_type]

        args = dict(
            object_from_json(
                metric_sqa.properties,
                decoder_registry=self.config.json_decoder_registry,
                class_decoder_registry=self.config.json_class_decoder_registry,
            )
            or {}
        )

        args["name"] = metric_sqa.name
        args["lower_is_better"] = metric_sqa.lower_is_better

        if metric_class is Metric and metric_sqa.signature != metric_sqa.name:
            args["signature_override"] = metric_sqa.signature

        args = metric_class.deserialize_init_args(args=args)
        metric = metric_class(**args)
        metric.db_id = metric_sqa.id

        return metric

    def _objective_from_sqa(self, metric: Metric, metric_sqa: SQAMetric) -> Objective:
        if metric_sqa.minimize is None:
            raise SQADecodeError(
                "Cannot decode SQAMetric to Objective because minimize is None."
            )
        if metric_sqa.scalarized_objective_weight is not None:
            raise SQADecodeError(
                f"The metric {metric.name} corresponding to regular objective does not "
                "have weight attribute"
            )
        # Resolve any conflicts between ``lower_is_better`` and ``minimize``.
        minimize = metric_sqa.minimize
        if metric.lower_is_better is not None and metric.lower_is_better != minimize:
            logger.warning(
                f"Metric {metric.name} has {metric.lower_is_better=} but objective "
                f"specifies {minimize=}. Overwriting ``lower_is_better`` to match "
                f"the optimization direction {minimize=}."
            )
            metric.lower_is_better = minimize
        return Objective(metric=metric, minimize=minimize)

    def _multi_objective_from_sqa(self, parent_metric_sqa: SQAMetric) -> Objective:
        metrics_sqa_children = self._get_and_process_children_metrics(
            parent_metric_sqa=parent_metric_sqa,
            children_attribute_name="scalarized_objective_children_metrics",
            fallback_function=_get_scalarized_objective_children_metrics,
            metric_type_name="MultiObjective",
        )

        # Extracting metric and weight for each child
        objectives = [
            self._objective_from_sqa(
                metric=self._metric_from_sqa_util(parent_metric_sqa),
                metric_sqa=parent_metric_sqa,
            )
            for parent_metric_sqa in metrics_sqa_children
        ]

        multi_objective = MultiObjective(objectives=objectives)
        multi_objective.db_id = parent_metric_sqa.id
        return multi_objective

    def _scalarized_objective_from_sqa(self, parent_metric_sqa: SQAMetric) -> Objective:
        if parent_metric_sqa.minimize is None:
            raise SQADecodeError(
                "Cannot decode SQAMetric to Scalarized Objective "
                "because minimize is None."
            )

        metrics_sqa_children = self._get_and_process_children_metrics(
            parent_metric_sqa=parent_metric_sqa,
            children_attribute_name="scalarized_objective_children_metrics",
            fallback_function=_get_scalarized_objective_children_metrics,
            metric_type_name="Scalarized Objective",
        )

        # Extracting metric and weight for each child
        metrics, weights = zip(
            *[
                (
                    self._metric_from_sqa_util(child),
                    child.scalarized_objective_weight,
                )
                for child in metrics_sqa_children
            ]
        )
        scalarized_objective = ScalarizedObjective(
            metrics=list(metrics),
            weights=list(weights),
            minimize=none_throws(parent_metric_sqa.minimize),
        )
        scalarized_objective.db_id = parent_metric_sqa.id
        return scalarized_objective

    def _outcome_constraint_from_sqa(
        self, metric: Metric, metric_sqa: SQAMetric
    ) -> OutcomeConstraint:
        if (
            metric_sqa.bound is None
            or metric_sqa.op is None
            or metric_sqa.relative is None
        ):
            raise SQADecodeError(
                "Cannot decode SQAMetric to OutcomeConstraint because "
                "bound, op, or relative is None."
            )
        return OutcomeConstraint(
            metric=metric,
            bound=float(none_throws(metric_sqa.bound)),
            op=none_throws(metric_sqa.op),
            relative=none_throws(metric_sqa.relative),
        )

    def _scalarized_outcome_constraint_from_sqa(
        self, metric: Metric, metric_sqa: SQAMetric
    ) -> ScalarizedOutcomeConstraint:
        if (
            metric_sqa.bound is None
            or metric_sqa.op is None
            or metric_sqa.relative is None
        ):
            raise SQADecodeError(
                "Cannot decode SQAMetric to Scalarized OutcomeConstraint because "
                "bound, op, or relative is None."
            )

        metrics_sqa_children = self._get_and_process_children_metrics(
            parent_metric_sqa=metric_sqa,
            children_attribute_name="scalarized_outcome_constraint_children_metrics",
            fallback_function=_get_scalarized_outcome_constraint_children_metrics,
            metric_type_name="Scalarized OutcomeConstraint",
        )

        # Extracting metric and weight for each child
        metrics, weights = zip(
            *[
                (
                    self._metric_from_sqa_util(child),
                    child.scalarized_outcome_constraint_weight,
                )
                for child in metrics_sqa_children
            ]
        )
        scalarized_outcome_constraint = ScalarizedOutcomeConstraint(
            metrics=list(metrics),
            weights=list(weights),
            bound=float(none_throws(metric_sqa.bound)),
            op=none_throws(metric_sqa.op),
            relative=none_throws(metric_sqa.relative),
        )
        scalarized_outcome_constraint.db_id = metric_sqa.id
        return scalarized_outcome_constraint

    def _objective_threshold_from_sqa(
        self, metric: Metric, metric_sqa: SQAMetric
    ) -> ObjectiveThreshold:
        if metric_sqa.bound is None:
            raise SQADecodeError(
                "Cannot decode SQAMetric to ObjectiveThreshold because bound is None."
            )

        if metric_sqa.relative is None:
            logger.warning(
                f"When decoding SQAMetric {metric.name} to ObjectiveThreshold, "
                f"'relative' is set to None. Overwriting 'relative' to False"
                " to prevent downstream experiment errors.  "
            )
        relative = metric_sqa.relative if metric_sqa.relative is not None else False

        ot = ObjectiveThreshold(
            metric=metric,
            bound=float(none_throws(metric_sqa.bound)),
            relative=relative,
            op=metric_sqa.op,
        )
        # ObjectiveThreshold constructor clones the passed-in metric, which means
        # the db id gets lost and so we need to reset it
        ot.metric._db_id = metric.db_id
        return ot

    def _get_and_process_children_metrics(
        self,
        parent_metric_sqa: SQAMetric,
        children_attribute_name: str,
        fallback_function: Callable[[int, "Decoder"], list[SQAMetric]],
        metric_type_name: str,
    ) -> list[SQAMetric]:
        """Helper method to get children metrics and apply skip_runners_and_metrics.

        This method consolidates the common pattern of:
        1. Trying to access children metrics directly from the parent
        2. Falling back to database query if DetachedInstanceError occurs
        3. Checking if children is None and raising appropriate error
        4. Applying skip_runners_and_metrics logic to children.
            This step requires setting skip_runners_and_metrics in
            `_set_sqa_metric_to_base_type` ahead of time.

        Args:
            parent_metric_sqa: The parent metric SQA object.
            children_attribute_name: Name of the attribute containing children metrics.
            fallback_function: Function to call if DetachedInstanceError occurs.
            metric_type_name: Name of the metric type for error messages.

        Returns:
            List of processed children metrics.
        """
        try:
            children_metrics_sqa = getattr(parent_metric_sqa, children_attribute_name)
        except DetachedInstanceError:
            children_metrics_sqa = fallback_function(parent_metric_sqa.id, self)

        if not children_metrics_sqa:
            raise SQADecodeError(
                f"Cannot decode SQAMetric to {metric_type_name} "
                "because the parent metric has no children metrics."
            )

        if parent_metric_sqa.properties and parent_metric_sqa.properties.get(
            "skip_runners_and_metrics"
        ):
            for child_metric in children_metrics_sqa:
                child_metric.metric_type = self.config.metric_registry[Metric]

        return children_metrics_sqa


def _get_scalarized_objective_children_metrics(
    metric_id: int, decoder: Decoder
) -> list[SQAMetric]:
    """Given a metric db id, fetch its scalarized objective children metrics."""
    metric_sqa_class = cast(
        type[SQAMetric],
        decoder.config.class_to_sqa_class[Metric],
    )
    with session_scope() as session:
        query = session.query(metric_sqa_class).filter_by(
            scalarized_objective_id=metric_id
        )
        metrics_sqa = query.all()
    return metrics_sqa


def _get_scalarized_outcome_constraint_children_metrics(
    metric_id: int, decoder: Decoder
) -> list[SQAMetric]:
    """Given a metric db id, fetch its scalarized outcome constraint
    children metrics."""
    metric_sqa_class = cast(
        type[SQAMetric],
        decoder.config.class_to_sqa_class[Metric],
    )
    with session_scope() as session:
        query = session.query(metric_sqa_class).filter_by(
            scalarized_outcome_constraint_id=metric_id
        )
        metrics_sqa = query.all()
    return metrics_sqa


def auxiliary_experiment_from_name(
    experiment_name: str,
    config: SQAConfig,
    is_active: bool,
    reduced_state: bool = False,
) -> AuxiliaryExperiment:
    """
    Load an ``AuxiliaryExperiment`` by name.

    Args:
        experiment_name: Name of the auxiliary experiment.
        config: The SQAConfig object used to load the experiment.

    Returns:
        An AuxiliaryExperiment object constructed from the given experiment.
    """

    from ax.storage.sqa_store.load import load_experiment

    experiment = load_experiment(
        experiment_name,
        config=config,
        skip_runners_and_metrics=True,
        load_auxiliary_experiments=False,
        reduced_state=reduced_state,
    )
    return AuxiliaryExperiment(experiment, is_active=is_active)
