#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, List

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
    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: Column[int] = Column(Integer, primary_key=True)
    generator_run_id: Column[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    parameter_type: Column[ParameterType] = Column(
        IntEnum(ParameterType), nullable=False
    )
    is_fidelity: Column[bool | None] = Column(Boolean)
    target_value: Column[TParamValue | None] = Column(JSONEncodedObject)

    # Attributes for Range Parameters
    digits: Column[int | None] = Column(Integer)
    log_scale: Column[bool | None] = Column(Boolean)
    lower: Column[Decimal | None] = Column(Float)
    upper: Column[Decimal | None] = Column(Float)

    # Attributes for Choice Parameters
    choice_values: Column[List[TParamValue] | None] = Column(JSONEncodedList)
    is_ordered: Column[bool | None] = Column(Boolean)
    is_task: Column[bool | None] = Column(Boolean)
    dependents: Column[dict[TParamValue, List[str]] | None] = Column(JSONEncodedObject)

    # Attributes for Fixed Parameters
    fixed_value: Column[TParamValue | None] = Column(JSONEncodedObject)


class SQAParameterConstraint(Base):
    __tablename__: str = "parameter_constraint_v2"

    bound: Column[Decimal] = Column(Float, nullable=False)
    constraint_dict: Column[dict[str, float]] = Column(JSONEncodedDict, nullable=False)
    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: Column[int] = Column(Integer, primary_key=True)
    generator_run_id: Column[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    type: Column[IntEnum] = Column(IntEnum(ParameterConstraintType), nullable=False)


class SQAMetric(Base):
    __tablename__: str = "metric_v2"

    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    generator_run_id: Column[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    id: Column[int] = Column(Integer, primary_key=True)
    lower_is_better: Column[bool | None] = Column(Boolean)
    intent: Column[MetricIntent] = Column(StringEnum(MetricIntent), nullable=False)
    metric_type: Column[int] = Column(Integer, nullable=False)
    name: Column[str] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    properties: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})

    # Attributes for Objectives
    minimize: Column[bool | None] = Column(Boolean)

    # Attributes for Outcome Constraints
    op: Column[ComparisonOp | None] = Column(IntEnum(ComparisonOp))
    bound: Column[Decimal | None] = Column(Float)
    relative: Column[bool | None] = Column(Boolean)

    # Multi-type Experiment attributes
    trial_type: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    canonical_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    scalarized_objective_id: Column[int | None] = Column(
        Integer, ForeignKey("metric_v2.id")
    )

    # Relationship containing SQAMetric(s) only defined for the parent metric
    # of Multi/Scalarized Objective contains all children of the parent metric
    # join_depth argument: used for loading self-referential relationships
    # https://docs.sqlalchemy.org/en/13/orm/self_referential.html#configuring-self-referential-eager-loading
    scalarized_objective_children_metrics: List["SQAMetric"] = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_objective_id],
    )

    # Attribute only defined for the children of Scalarized Objective
    scalarized_objective_weight: Column[Decimal | None] = Column(Float)
    scalarized_outcome_constraint_id: Column[int | None] = Column(
        Integer, ForeignKey("metric_v2.id")
    )
    scalarized_outcome_constraint_children_metrics: List["SQAMetric"] = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_outcome_constraint_id],
    )
    scalarized_outcome_constraint_weight: Column[Decimal | None] = Column(Float)


class SQAArm(Base):
    __tablename__: str = "arm_v2"

    generator_run_id: Column[int] = Column(
        Integer, ForeignKey("generator_run_v2.id"), nullable=False
    )
    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    parameters: Column[TParameterization] = Column(JSONEncodedTextDict, nullable=False)
    weight: Column[Decimal] = Column(Float, nullable=False, default=1.0)


class SQAAbandonedArm(Base):
    __tablename__: str = "abandoned_arm_v2"

    abandoned_reason: Column[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    time_abandoned: Column[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Column[int] = Column(Integer, ForeignKey("trial_v2.id"), nullable=False)


class SQAGeneratorRun(Base):
    __tablename__: str = "generator_run_v2"

    best_arm_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    best_arm_parameters: Column[TParameterization | None] = Column(JSONEncodedTextDict)
    best_arm_predictions: Column[TModelPredictArm | None] = Column(JSONEncodedList)
    generator_run_type: Column[int | None] = Column(Integer)
    id: Column[int] = Column(Integer, primary_key=True)
    index: Column[int | None] = Column(Integer)
    model_predictions: Column[TModelPredict | None] = Column(JSONEncodedList)
    time_created: Column[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Column[int | None] = Column(Integer, ForeignKey("trial_v2.id"))
    weight: Column[Decimal | None] = Column(Float)
    fit_time: Column[Decimal | None] = Column(Float)
    gen_time: Column[Decimal | None] = Column(Float)
    model_key: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    model_kwargs: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    bridge_kwargs: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    gen_metadata: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    model_state_after_gen: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    generation_strategy_id: Column[int | None] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    generation_step_index: Column[int | None] = Column(Integer)
    candidate_metadata_by_arm_signature: Column[dict[str, Any] | None] = Column(
        JSONEncodedTextDict
    )
    generation_node_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))

    # relationships
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    arms: List[SQAArm] = relationship(
        "SQAArm",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAArm.id,
    )
    metrics: List[SQAMetric] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: List[SQAParameter] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: List[SQAParameterConstraint] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )


class SQARunner(Base):
    __tablename__: str = "runner"

    id: Column[int] = Column(Integer, primary_key=True)
    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    properties: Column[dict[str, Any] | None] = Column(
        JSONEncodedLongTextDict, default={}
    )
    runner_type: Column[int] = Column(Integer, nullable=False)
    trial_id: Column[int | None] = Column(Integer, ForeignKey("trial_v2.id"))

    # Multi-type Experiment attributes
    trial_type: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))


class SQAData(Base):
    __tablename__: str = "data_v2"

    id: Column[int] = Column(Integer, primary_key=True)
    data_json: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    description: Column[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    time_created: Column[int] = Column(BigInteger, nullable=False)
    trial_index: Column[int | None] = Column(Integer)
    generation_strategy_id: Column[int | None] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    structure_metadata_json: Column[str | None] = Column(
        Text(LONGTEXT_BYTES), nullable=True
    )


class SQAGenerationStrategy(Base):
    __tablename__: str = "generation_strategy"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    steps: Column[List[dict[str, Any]]] = Column(JSONEncodedList, nullable=False)
    curr_index: Column[int | None] = Column(Integer, nullable=True)
    experiment_id: Column[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    nodes: Column[List[dict[str, Any]]] = Column(JSONEncodedList, nullable=True)
    curr_node_name: Column[str | None] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=True
    )

    generator_runs: List[SQAGeneratorRun] = relationship(
        "SQAGeneratorRun",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAGeneratorRun.id,
    )


class SQATrial(Base):
    __tablename__: str = "trial_v2"

    abandoned_reason: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    failed_reason: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    deployed_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    experiment_id: Column[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )
    id: Column[int] = Column(Integer, primary_key=True)
    index: Column[int] = Column(Integer, index=True, nullable=False)
    is_batch: Column[bool] = Column("is_batched", Boolean, nullable=False, default=True)
    lifecycle_stage: Column[LifecycleStage | None] = Column(
        IntEnum(LifecycleStage), nullable=True
    )
    num_arms_created: Column[int] = Column(Integer, nullable=False, default=0)
    optimize_for_power: Column[bool | None] = Column(Boolean)
    ttl_seconds: Column[int | None] = Column(Integer)
    run_metadata: Column[dict[str, Any] | None] = Column(JSONEncodedLongTextDict)
    stop_metadata: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    status: Column[TrialStatus] = Column(
        IntEnum(TrialStatus), nullable=False, default=TrialStatus.CANDIDATE
    )
    status_quo_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    time_completed: Column[datetime | None] = Column(IntTimestamp)
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    time_staged: Column[datetime | None] = Column(IntTimestamp)
    time_run_started: Column[datetime | None] = Column(IntTimestamp)
    trial_type: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    generation_step_index: Column[int | None] = Column(Integer)
    properties: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    abandoned_arms: List[SQAAbandonedArm] = relationship(
        "SQAAbandonedArm", cascade="all, delete-orphan", lazy="selectin"
    )
    generator_runs: List[SQAGeneratorRun] = relationship(
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
    level: Column[int] = Column(Integer, nullable=False)
    dataframe_json: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    blob: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    blob_annotation: Column[str] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False
    )
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    experiment_id: Column[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )
    attributes: Column[str] = Column(Text(LONGTEXT_BYTES), nullable=False)


class SQAExperiment(Base):
    __tablename__: str = "experiment_v2"

    description: Column[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_type: Column[int | None] = Column(Integer)
    id: Column[int] = Column(Integer, primary_key=True)
    is_test: Column[bool] = Column(Boolean, nullable=False, default=False)
    name: Column[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    properties: Column[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})
    status_quo_name: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    status_quo_parameters: Column[TParameterization | None] = Column(
        JSONEncodedTextDict
    )
    time_created: Column[datetime] = Column(IntTimestamp, nullable=False)
    default_trial_type: Column[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    default_data_type: Column[DataType] = Column(IntEnum(DataType), nullable=True)
    # pyre-fixme[8]: Incompatible attribute type [8]: Attribute
    # `auxiliary_experiments_by_purpose` declared in class `SQAExperiment` has
    # type `Optional[Dict[str, List[str]]]` but is used as type `Column[typing.Any]`
    auxiliary_experiments_by_purpose: dict[str, List[str]] | None = Column(
        JSONEncodedTextDict, nullable=True, default={}
    )

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    data: List[SQAData] = relationship(
        "SQAData", cascade="all, delete-orphan", lazy="selectin"
    )
    metrics: List[SQAMetric] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: List[SQAParameter] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: List[SQAParameterConstraint] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )
    runners: List[SQARunner] = relationship(
        "SQARunner", cascade="all, delete-orphan", lazy=False
    )
    trials: List[SQATrial] = relationship(
        "SQATrial", cascade="all, delete-orphan", lazy="selectin"
    )
    generation_strategy: SQAGenerationStrategy | None = relationship(
        "SQAGenerationStrategy",
        backref=backref("experiment", lazy=True),
        uselist=False,
        lazy=True,
    )
    analysis_cards: List[SQAAnalysisCard] = relationship(
        "SQAAnalysisCard", cascade="all, delete-orphan", lazy="selectin"
    )
