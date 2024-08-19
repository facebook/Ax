#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from ax.analysis.analysis import AnalysisCardLevel

from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import LifecycleStage
from ax.core.parameter import ParameterType
from ax.core.types import (
    ComparisonOp,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.storage.sqa_store.db import (
    Base,
    LONG_STRING_FIELD_LENGTH,
    LONGTEXT_BYTES,
    NAME_OR_TYPE_FIELD_LENGTH,
)
from ax.storage.sqa_store.json import (
    JSONEncodedDict,
    JSONEncodedList,
    JSONEncodedLongTextDict,
    JSONEncodedObject,
    JSONEncodedTextDict,
)
from ax.storage.sqa_store.sqa_enum import IntEnum, StringEnum
from ax.storage.sqa_store.timestamp import IntTimestamp
from ax.storage.utils import DataType, DomainType, MetricIntent, ParameterConstraintType
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import backref, relationship

ONLY_ONE_FIELDS = ["experiment_id", "generator_run_id"]


ONLY_ONE_METRIC_FIELDS = ["scalarized_objective_id", "scalarized_outcome_constraint_id"]


class SQAParameter(Base):
    __tablename__: str = "parameter_v2"

    domain_type: Column[DomainType] = Column(IntEnum(DomainType), nullable=False)
    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    id: Column[int] = Column(Integer, primary_key=True)
    generator_run_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    parameter_type: Column[ParameterType] = Column(
        IntEnum(ParameterType), nullable=False
    )
    is_fidelity: Column[Optional[bool]] = Column(Boolean)
    target_value: Column[Optional[TParamValue]] = Column(JSONEncodedObject)

    # Attributes for Range Parameters
    digits: Column[Optional[int]] = Column(Integer)
    log_scale: Column[Optional[bool]] = Column(Boolean)
    lower: Column[Optional[Decimal]] = Column(Float)
    upper: Column[Optional[Decimal]] = Column(Float)

    # Attributes for Choice Parameters
    choice_values: Column[Optional[list[TParamValue]]] = Column(JSONEncodedList)
    is_ordered: Column[Optional[bool]] = Column(Boolean)
    is_task: Column[Optional[bool]] = Column(Boolean)
    dependents: Column[Optional[dict[TParamValue, list[str]]]] = Column(
        JSONEncodedObject
    )

    # Attributes for Fixed Parameters
    fixed_value: Column[Optional[TParamValue]] = Column(JSONEncodedObject)


class SQAParameterConstraint(Base):
    __tablename__: str = "parameter_constraint_v2"

    bound: Column[Decimal] = Column(Float, nullable=False)
    constraint_dict: Column[dict[str, float]] = Column(JSONEncodedDict, nullable=False)
    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    id: Column[int] = Column(Integer, primary_key=True)
    generator_run_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    type: Column[IntEnum] = Column(IntEnum(ParameterConstraintType), nullable=False)


class SQAMetric(Base):
    __tablename__: str = "metric_v2"

    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    generator_run_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    id: Column[int] = Column(Integer, primary_key=True)
    lower_is_better: Column[Optional[bool]] = Column(Boolean)
    intent: Column[MetricIntent] = Column(StringEnum(MetricIntent), nullable=False)
    metric_type: Column[int] = Column(Integer, nullable=False)
    name: Column[str] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    properties: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedTextDict, default={}
    )

    # Attributes for Objectives
    minimize: Column[Optional[bool]] = Column(Boolean)

    # Attributes for Outcome Constraints
    op: Column[Optional[ComparisonOp]] = Column(IntEnum(ComparisonOp))
    bound: Column[Optional[Decimal]] = Column(Float)
    relative: Column[Optional[bool]] = Column(Boolean)

    # Multi-type Experiment attributes
    trial_type: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    canonical_name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    scalarized_objective_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("metric_v2.id")
    )

    # Relationship containing SQAMetric(s) only defined for the parent metric
    # of Multi/Scalarized Objective contains all children of the parent metric
    # join_depth argument: used for loading self-referential relationships
    # https://docs.sqlalchemy.org/en/13/orm/self_referential.html#configuring-self-referential-eager-loading
    scalarized_objective_children_metrics: list[SQAMetric] = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_objective_id],
    )

    # Attribute only defined for the children of Scalarized Objective
    scalarized_objective_weight: Column[Optional[Decimal]] = Column(Float)
    scalarized_outcome_constraint_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("metric_v2.id")
    )
    scalarized_outcome_constraint_children_metrics: list[SQAMetric] = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_outcome_constraint_id],
    )
    scalarized_outcome_constraint_weight: Column[Optional[Decimal]] = Column(Float)


class SQAArm(Base):
    __tablename__: str = "arm_v2"

    generator_run_id: Column[int] = Column(
        Integer, ForeignKey("generator_run_v2.id"), nullable=False
    )
    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    parameters: Column[TParameterization] = Column(JSONEncodedTextDict, nullable=False)
    weight: Column[Decimal] = Column(Float, nullable=False, default=1.0)


class SQAAbandonedArm(Base):
    __tablename__: str = "abandoned_arm_v2"

    abandoned_reason: Column[Optional[str]] = Column(String(LONG_STRING_FIELD_LENGTH))
    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    time_abandoned: Column[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Column[int] = Column(Integer, ForeignKey("trial_v2.id"), nullable=False)


class SQAGeneratorRun(Base):
    __tablename__: str = "generator_run_v2"

    best_arm_name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    best_arm_parameters: Column[Optional[TParameterization]] = Column(
        JSONEncodedTextDict
    )
    best_arm_predictions: Column[Optional[TModelPredictArm]] = Column(JSONEncodedList)
    generator_run_type: Column[Optional[int]] = Column(Integer)
    id: Column[int] = Column(Integer, primary_key=True)
    index: Column[Optional[int]] = Column(Integer)
    model_predictions: Column[Optional[TModelPredict]] = Column(JSONEncodedList)
    time_created: Column[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Column[Optional[int]] = Column(Integer, ForeignKey("trial_v2.id"))
    weight: Column[Optional[Decimal]] = Column(Float)
    fit_time: Column[Optional[Decimal]] = Column(Float)
    gen_time: Column[Optional[Decimal]] = Column(Float)
    model_key: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    model_kwargs: Column[Optional[dict[str, Any]]] = Column(JSONEncodedTextDict)
    bridge_kwargs: Column[Optional[dict[str, Any]]] = Column(JSONEncodedTextDict)
    gen_metadata: Column[Optional[dict[str, Any]]] = Column(JSONEncodedTextDict)
    model_state_after_gen: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedTextDict
    )
    generation_strategy_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    generation_step_index: Column[Optional[int]] = Column(Integer)
    candidate_metadata_by_arm_signature: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedTextDict
    )
    generation_node_name: Column[Optional[str]] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH)
    )

    # relationships
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    arms: list[SQAArm] = relationship(
        "SQAArm",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAArm.id,
    )
    metrics: list[SQAMetric] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: list[SQAParameter] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: list[SQAParameterConstraint] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )


class SQARunner(Base):
    __tablename__: str = "runner"

    id: Column[int] = Column(Integer, primary_key=True)
    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    properties: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedLongTextDict, default={}
    )
    runner_type: Column[int] = Column(Integer, nullable=False)
    trial_id: Column[Optional[int]] = Column(Integer, ForeignKey("trial_v2.id"))

    # Multi-type Experiment attributes
    trial_type: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))


class SQAData(Base):
    __tablename__: str = "data_v2"

    id: Column[int] = Column(Integer, primary_key=True)
    data_json: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    description: Column[Optional[str]] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    time_created: Column[int] = Column(BigInteger, nullable=False)
    trial_index: Column[Optional[int]] = Column(Integer)
    generation_strategy_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    structure_metadata_json: Column[Optional[str]] = Column(
        Text(LONGTEXT_BYTES), nullable=True
    )


class SQAGenerationStrategy(Base):
    __tablename__: str = "generation_strategy"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    steps: Column[list[dict[str, Any]]] = Column(JSONEncodedList, nullable=False)
    curr_index: Column[int] = Column(Integer, nullable=False)
    experiment_id: Column[Optional[int]] = Column(
        Integer, ForeignKey("experiment_v2.id")
    )
    nodes: Column[list[dict[str, Any]]] = Column(JSONEncodedList, nullable=True)
    curr_node_name: Column[Optional[str]] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=True
    )

    generator_runs: list[SQAGeneratorRun] = relationship(
        "SQAGeneratorRun",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAGeneratorRun.id,
    )


class SQATrial(Base):
    __tablename__: str = "trial_v2"

    abandoned_reason: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    failed_reason: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    deployed_name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    experiment_id: Column[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )
    id: Column[int] = Column(Integer, primary_key=True)
    index: Column[int] = Column(Integer, index=True, nullable=False)
    is_batch: Column[bool] = Column("is_batched", Boolean, nullable=False, default=True)
    lifecycle_stage: Column[Optional[LifecycleStage]] = Column(
        IntEnum(LifecycleStage), nullable=True
    )
    num_arms_created: Column[int] = Column(Integer, nullable=False, default=0)
    optimize_for_power: Column[Optional[bool]] = Column(Boolean)
    ttl_seconds: Column[Optional[int]] = Column(Integer)
    run_metadata: Column[Optional[dict[str, Any]]] = Column(JSONEncodedLongTextDict)
    stop_metadata: Column[Optional[dict[str, Any]]] = Column(JSONEncodedTextDict)
    status: Column[TrialStatus] = Column(
        IntEnum(TrialStatus), nullable=False, default=TrialStatus.CANDIDATE
    )
    status_quo_name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    time_completed: Column[Optional[datetime]] = Column(IntTimestamp)
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    time_staged: Column[Optional[datetime]] = Column(IntTimestamp)
    time_run_started: Column[Optional[datetime]] = Column(IntTimestamp)
    trial_type: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    generation_step_index: Column[Optional[int]] = Column(Integer)
    properties: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedTextDict, default={}
    )

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    abandoned_arms: list[SQAAbandonedArm] = relationship(
        "SQAAbandonedArm", cascade="all, delete-orphan", lazy="selectin"
    )
    generator_runs: list[SQAGeneratorRun] = relationship(
        "SQAGeneratorRun", cascade="all, delete-orphan", lazy="selectin"
    )
    runner: SQARunner = relationship(
        "SQARunner", uselist=False, cascade="all, delete-orphan", lazy=False
    )


class SQAAnalysisCard(Base):
    __tablename__: str = "analysis_card"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    title: Column[str] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    subtitle: Column[str] = Column(Text, nullable=False)
    level: Column[AnalysisCardLevel] = Column(
        IntEnum(AnalysisCardLevel), nullable=False
    )
    dataframe_json: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    blob: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    blob_annotation: Column[str] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False
    )
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    experiment_id: Column[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )


class SQAExperiment(Base):
    __tablename__: str = "experiment_v2"

    description: Column[Optional[str]] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_type: Column[Optional[int]] = Column(Integer)
    id: Column[int] = Column(Integer, primary_key=True)
    is_test: Column[bool] = Column(Boolean, nullable=False, default=False)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    properties: Column[Optional[dict[str, Any]]] = Column(
        JSONEncodedTextDict, default={}
    )
    status_quo_name: Column[Optional[str]] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    status_quo_parameters: Column[Optional[TParameterization]] = Column(
        JSONEncodedTextDict
    )
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    default_trial_type: Column[Optional[str]] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH)
    )
    default_data_type: Column[DataType] = Column(IntEnum(DataType), nullable=True)
    # pyre-fixme[8]: Incompatible attribute type [8]: Attribute
    # `auxiliary_experiments_by_purpose` declared in class `SQAExperiment` has
    # type `Optional[Dict[str, List[str]]]` but is used as type `Column[typing.Any]`
    auxiliary_experiments_by_purpose: Optional[dict[str, list[str]]] = Column(
        JSONEncodedTextDict, nullable=True, default={}
    )

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    data: list[SQAData] = relationship(
        "SQAData", cascade="all, delete-orphan", lazy="selectin"
    )
    metrics: list[SQAMetric] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: list[SQAParameter] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: list[SQAParameterConstraint] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )
    runners: list[SQARunner] = relationship(
        "SQARunner", cascade="all, delete-orphan", lazy=False
    )
    trials: list[SQATrial] = relationship(
        "SQATrial", cascade="all, delete-orphan", lazy="selectin"
    )
    generation_strategy: Optional[SQAGenerationStrategy] = relationship(
        "SQAGenerationStrategy",
        backref=backref("experiment", lazy=True),
        uselist=False,
        lazy=True,
    )
    analysis_cards: list[SQAAnalysisCard] = relationship(
        "SQAAnalysisCard", cascade="all, delete-orphan", lazy="selectin"
    )
