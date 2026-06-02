#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[8]
#
# `Mapped[T] = Column(...)` is the SA 1.4-compatible bridging form. SA 2.0 stubs
# see `Column[T]` on the RHS as incompatible with `Mapped[T]` LHS, hence the
# file-level [8] suppression. Runtime works under both SA versions, and pyre
# still resolves `obj.attr` access correctly via the Mapped[T] descriptor at
# callsites.
#
# **CONTRACT FOR FUTURE EDITS**: when adding a column, the `Mapped[T]` annotation
# MUST match the `Column(...)` runtime nullability:
#   - `Mapped[T]`        → `Column(..., nullable=False)`  (or `primary_key=True`)
#   - `Mapped[T | None]` → `Column(...)` (default) or `Column(..., nullable=True)`
# Mapped[T] no longer drives nullability automatically here (unlike SA 2.0's
# `mapped_column`), so a mismatched declaration silently creates a column with
# the WRONG nullability and the annotation will lie. Audit explicitly.

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from ax.core.evaluations_to_data import DataType
from ax.core.experiment_status import ExperimentStatus
from ax.core.parameter import ParameterType
from ax.core.trial_status import TrialStatus
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
from ax.storage.utils import DomainType, MetricIntent, ParameterConstraintType
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

# Mapped is SA 2.0-only. Under SA 2.0, it MUST be importable at runtime because
# the declarative metaclass evaluates class annotations (via typing.get_type_hints
# semantics) at class-definition time. Under SA 1.4 it doesn't exist — but SA 1.4
# also doesn't introspect mapped annotations, so silently skipping the import is
# safe (annotations stay as strings due to `from __future__ import annotations`).
try:
    from sqlalchemy.orm import Mapped
except ImportError:
    pass

ONLY_ONE_FIELDS = ["experiment_id", "generator_run_id"]


ONLY_ONE_METRIC_FIELDS = ["scalarized_objective_id", "scalarized_outcome_constraint_id"]


class SQAParameter(Base):
    __tablename__: str = "parameter_v2"

    domain_type: Mapped[DomainType] = Column(IntEnum(DomainType), nullable=False)
    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: Mapped[int] = Column(Integer, primary_key=True)
    generator_run_id: Mapped[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    name: Mapped[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    parameter_type: Mapped[ParameterType] = Column(
        IntEnum(ParameterType), nullable=False
    )
    is_fidelity: Mapped[bool | None] = Column(Boolean)
    target_value: Mapped[TParamValue | None] = Column(JSONEncodedObject)
    backfill_value: Mapped[TParamValue | None] = Column(JSONEncodedObject)
    default_value: Mapped[TParamValue | None] = Column(JSONEncodedObject)

    # Attributes for Range Parameters
    digits: Mapped[int | None] = Column(Integer)
    step_size: Mapped[float | None] = Column(Float)
    log_scale: Mapped[bool | None] = Column(Boolean)
    lower: Mapped[Decimal | None] = Column(Float)
    upper: Mapped[Decimal | None] = Column(Float)

    # Attributes for Choice Parameters
    choice_values: Mapped[list[TParamValue] | None] = Column(JSONEncodedList)
    is_ordered: Mapped[bool | None] = Column(Boolean)
    is_task: Mapped[bool | None] = Column(Boolean)
    dependents: Mapped[dict[TParamValue, list[str]] | None] = Column(JSONEncodedObject)

    # Attributes for Fixed Parameters
    fixed_value: Mapped[TParamValue | None] = Column(JSONEncodedObject)

    # Attribute for Derived Parameters
    expression_str: Mapped[str | None] = Column(String(LONGTEXT_BYTES))


class SQAParameterConstraint(Base):
    __tablename__: str = "parameter_constraint_v2"

    bound: Mapped[Decimal] = Column(Float, nullable=False)
    constraint_dict: Mapped[dict[str, float]] = Column(JSONEncodedDict, nullable=False)
    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: Mapped[int] = Column(Integer, primary_key=True)
    generator_run_id: Mapped[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    type: Mapped[IntEnum] = Column(IntEnum(ParameterConstraintType), nullable=False)


class SQAMetric(Base):
    __tablename__: str = "metric_v2"

    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    generator_run_id: Mapped[int | None] = Column(
        Integer, ForeignKey("generator_run_v2.id")
    )
    id: Mapped[int] = Column(Integer, primary_key=True)
    lower_is_better: Mapped[bool | None] = Column(Boolean)
    intent: Mapped[MetricIntent] = Column(StringEnum(MetricIntent), nullable=False)
    metric_type: Mapped[int] = Column(Integer, nullable=False)
    name: Mapped[str] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    properties: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})
    signature: Mapped[str] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)

    # Attributes for Objectives
    minimize: Mapped[bool | None] = Column(Boolean)

    # Attributes for Outcome Constraints
    op: Mapped[ComparisonOp | None] = Column(IntEnum(ComparisonOp))
    bound: Mapped[Decimal | None] = Column(Float)
    relative: Mapped[bool | None] = Column(Boolean)

    # Multi-type Experiment attributes
    trial_type: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    canonical_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    scalarized_objective_id: Mapped[int | None] = Column(
        Integer, ForeignKey("metric_v2.id")
    )

    # Relationship containing SQAMetric(s) only defined for the parent metric
    # of Multi/Scalarized Objective contains all children of the parent metric
    # join_depth argument: used for loading self-referential relationships
    # https://docs.sqlalchemy.org/en/13/orm/self_referential.html#configuring-self-referential-eager-loading
    scalarized_objective_children_metrics: Mapped[list[SQAMetric]] = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_objective_id],
    )

    # Attribute only defined for the children of Scalarized Objective
    scalarized_objective_weight: Mapped[Decimal | None] = Column(Float)
    scalarized_outcome_constraint_id: Mapped[int | None] = Column(
        Integer, ForeignKey("metric_v2.id")
    )
    scalarized_outcome_constraint_children_metrics: Mapped[list[SQAMetric]] = (
        relationship(
            "SQAMetric",
            cascade="all, delete-orphan",
            lazy=True,
            foreign_keys=[scalarized_outcome_constraint_id],
        )
    )
    scalarized_outcome_constraint_weight: Mapped[Decimal | None] = Column(Float)


class SQAArm(Base):
    __tablename__: str = "arm_v2"

    generator_run_id: Mapped[int] = Column(
        Integer, ForeignKey("generator_run_v2.id"), nullable=False
    )
    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    parameters: Mapped[TParameterization] = Column(JSONEncodedTextDict, nullable=False)
    weight: Mapped[Decimal] = Column(Float, nullable=False, default=1.0)


class SQAAbandonedArm(Base):
    __tablename__: str = "abandoned_arm_v2"

    abandoned_reason: Mapped[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    time_abandoned: Mapped[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Mapped[int] = Column(Integer, ForeignKey("trial_v2.id"), nullable=False)


class SQAGeneratorRun(Base):
    __tablename__: str = "generator_run_v2"

    best_arm_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    best_arm_parameters: Mapped[TParameterization | None] = Column(JSONEncodedTextDict)
    best_arm_predictions: Mapped[TModelPredictArm | None] = Column(JSONEncodedList)
    generator_run_type: Mapped[int | None] = Column(Integer)
    id: Mapped[int] = Column(Integer, primary_key=True)
    index: Mapped[int | None] = Column(Integer)
    model_predictions: Mapped[TModelPredict | None] = Column(JSONEncodedList)
    time_created: Mapped[datetime] = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: Mapped[int | None] = Column(Integer, ForeignKey("trial_v2.id"))
    weight: Mapped[Decimal | None] = Column(Float)
    fit_time: Mapped[Decimal | None] = Column(Float)
    gen_time: Mapped[Decimal | None] = Column(Float)
    model_key: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    model_kwargs: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    bridge_kwargs: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    gen_metadata: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    model_state_after_gen: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    generation_strategy_id: Mapped[int | None] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    candidate_metadata_by_arm_signature: Mapped[dict[str, Any] | None] = Column(
        JSONEncodedTextDict
    )
    generation_node_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    suggested_experiment_status: Mapped[ExperimentStatus | None] = Column(
        IntEnum(ExperimentStatus), nullable=True
    )

    # relationships
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    arms: Mapped[list[SQAArm]] = relationship(
        "SQAArm",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAArm.id,
    )
    metrics: Mapped[list[SQAMetric]] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: Mapped[list[SQAParameter]] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: Mapped[list[SQAParameterConstraint]] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )


class SQARunner(Base):
    __tablename__: str = "runner"

    id: Mapped[int] = Column(Integer, primary_key=True)
    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    properties: Mapped[dict[str, Any] | None] = Column(
        JSONEncodedLongTextDict, default={}
    )
    runner_type: Mapped[int] = Column(Integer, nullable=False)
    trial_id: Mapped[int | None] = Column(Integer, ForeignKey("trial_v2.id"))

    # Multi-type Experiment attributes
    trial_type: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))


class SQAData(Base):
    __tablename__: str = "data_v2"

    id: Mapped[int] = Column(Integer, primary_key=True)
    data_json: Mapped[str] = Column(Text(LONGTEXT_BYTES), nullable=False)
    description: Mapped[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    time_created: Mapped[int] = Column(BigInteger, nullable=False)
    trial_index: Mapped[int | None] = Column(Integer)
    generation_strategy_id: Mapped[int | None] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    structure_metadata_json: Mapped[str | None] = Column(
        Text(LONGTEXT_BYTES), nullable=True
    )


class SQAGenerationStrategy(Base):
    __tablename__: str = "generation_strategy"

    id: Mapped[int] = Column(Integer, primary_key=True)
    name: Mapped[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    steps: Mapped[list[dict[str, Any]]] = Column(JSONEncodedList, nullable=False)
    curr_index: Mapped[int | None] = Column(Integer, nullable=True)
    experiment_id: Mapped[int | None] = Column(Integer, ForeignKey("experiment_v2.id"))
    nodes: Mapped[list[dict[str, Any]] | None] = Column(JSONEncodedList, nullable=True)
    curr_node_name: Mapped[str | None] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=True
    )

    generator_runs: Mapped[list[SQAGeneratorRun]] = relationship(
        "SQAGeneratorRun",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAGeneratorRun.id,
    )


class SQATrial(Base):
    __tablename__: str = "trial_v2"

    abandoned_reason: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    failed_reason: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    deployed_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    experiment_id: Mapped[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )
    id: Mapped[int] = Column(Integer, primary_key=True)
    index: Mapped[int] = Column(Integer, index=True, nullable=False)
    is_batch: Mapped[bool] = Column("is_batched", Boolean, nullable=False, default=True)
    num_arms_created: Mapped[int] = Column(Integer, nullable=False, default=0)
    ttl_seconds: Mapped[int | None] = Column(Integer)
    run_metadata: Mapped[dict[str, Any] | None] = Column(JSONEncodedLongTextDict)
    stop_metadata: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    status: Mapped[TrialStatus] = Column(
        IntEnum(TrialStatus), nullable=False, default=TrialStatus.CANDIDATE
    )
    status_quo_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    time_completed: Mapped[datetime | None] = Column(IntTimestamp)
    time_created: Mapped[datetime] = Column(IntTimestamp, nullable=False)
    time_staged: Mapped[datetime | None] = Column(IntTimestamp)
    time_run_started: Mapped[datetime | None] = Column(IntTimestamp)
    trial_type: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    properties: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    abandoned_arms: Mapped[list[SQAAbandonedArm]] = relationship(
        "SQAAbandonedArm", cascade="all, delete-orphan", lazy="selectin"
    )
    generator_runs: Mapped[list[SQAGeneratorRun]] = relationship(
        "SQAGeneratorRun", cascade="all, delete-orphan", lazy="selectin"
    )
    runner: Mapped[SQARunner] = relationship(
        "SQARunner", uselist=False, cascade="all, delete-orphan", lazy=False
    )


class SQAAuxiliaryExperiment(Base):
    __tablename__: str = "auxiliary_experiments"

    source_experiment_id: Mapped[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), primary_key=True
    )
    target_experiment_id: Mapped[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), primary_key=True
    )
    purpose: Mapped[str] = Column(String(LONG_STRING_FIELD_LENGTH), primary_key=True)
    is_active: Mapped[bool] = Column(Boolean, nullable=False)
    properties: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict)
    time: Mapped[datetime] = Column(IntTimestamp, nullable=False, default=datetime.now)
    source_experiment: Mapped[SQAExperiment] = relationship(
        "SQAExperiment",
        foreign_keys=[source_experiment_id],
        lazy="selectin",
        viewonly=True,
        innerjoin=True,
    )
    target_experiment: Mapped[SQAExperiment] = relationship(
        "SQAExperiment",
        foreign_keys=[target_experiment_id],
        lazy="selectin",
        viewonly=True,
        innerjoin=True,
    )


class SQAExperiment(Base):
    __tablename__: str = "experiment_v2"

    description: Mapped[str | None] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_type: Mapped[int | None] = Column(Integer)
    id: Mapped[int] = Column(Integer, primary_key=True)
    is_test: Mapped[bool] = Column(Boolean, nullable=False, default=False)
    name: Mapped[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    properties: Mapped[dict[str, Any] | None] = Column(JSONEncodedTextDict, default={})
    status_quo_name: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    status_quo_parameters: Mapped[TParameterization | None] = Column(
        JSONEncodedTextDict
    )
    time_created: Mapped[datetime] = Column(IntTimestamp, nullable=False)
    status: Mapped[ExperimentStatus | None] = Column(
        IntEnum(ExperimentStatus), nullable=True
    )
    default_trial_type: Mapped[str | None] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    default_data_type: Mapped[DataType | None] = Column(
        IntEnum(DataType), nullable=True
    )
    auxiliary_experiments_by_purpose: Mapped[dict[str, list[dict[str, Any]]] | None] = (
        Column(JSONEncodedTextDict, nullable=True, default={})
    )

    # relationships
    # Trials and experiments are mutable, so the children relationships need
    # cascade="all, delete-orphan", which means if we remove or replace
    # a child, the old one will be deleted.
    # Use selectin loading for collections to prevent idle timeout errors
    # (https://docs.sqlalchemy.org/en/13/orm/loading_relationships.html#selectin-eager-loading)
    data: Mapped[list[SQAData]] = relationship(
        "SQAData", cascade="all, delete-orphan", lazy="selectin"
    )
    metrics: Mapped[list[SQAMetric]] = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin"
    )
    parameters: Mapped[list[SQAParameter]] = relationship(
        "SQAParameter", cascade="all, delete-orphan", lazy="selectin"
    )
    parameter_constraints: Mapped[list[SQAParameterConstraint]] = relationship(
        "SQAParameterConstraint", cascade="all, delete-orphan", lazy="selectin"
    )
    runners: Mapped[list[SQARunner]] = relationship(
        "SQARunner", cascade="all, delete-orphan", lazy=False
    )
    trials: Mapped[list[SQATrial]] = relationship(
        "SQATrial", cascade="all, delete-orphan", lazy="selectin"
    )
    generation_strategy: Mapped[SQAGenerationStrategy | None] = relationship(
        "SQAGenerationStrategy",
        backref=backref("experiment", lazy=True),
        uselist=False,
        lazy=True,
    )
    auxiliary_experiments: Mapped[list[SQAAuxiliaryExperiment]] = relationship(
        "SQAAuxiliaryExperiment",
        cascade="all, delete-orphan",
        lazy="selectin",
        foreign_keys=[SQAAuxiliaryExperiment.target_experiment_id],
    )

    analysis_cards: Mapped[list[SQAAnalysisCard]] = relationship(
        "SQAAnalysisCard", cascade="all, delete-orphan", lazy="selectin"
    )


class SQAAnalysisCard(Base):
    __tablename__: str = "analysis_card_v2"

    id: Mapped[int] = Column(Integer, primary_key=True)

    experiment_id: Mapped[int] = Column(
        Integer, ForeignKey("experiment_v2.id"), nullable=False
    )
    name: Mapped[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    timestamp: Mapped[datetime] = Column(IntTimestamp, nullable=False)

    parent_id: Mapped[int | None] = Column(
        Integer,
        ForeignKey("analysis_card_v2.id"),
        nullable=True,
    )
    order: Mapped[int | None] = Column(Integer, nullable=True)

    title: Mapped[str | None] = Column(String(LONG_STRING_FIELD_LENGTH), nullable=True)
    subtitle: Mapped[str | None] = Column(Text, nullable=True)
    dataframe_json: Mapped[str | None] = Column(Text(LONGTEXT_BYTES), nullable=True)
    blob: Mapped[str | None] = Column(Text(LONGTEXT_BYTES), nullable=True)
    blob_annotation: Mapped[str | None] = Column(
        String(NAME_OR_TYPE_FIELD_LENGTH), nullable=True
    )
    parent: Mapped[SQAAnalysisCard | None] = relationship(
        "SQAAnalysisCard",
        back_populates="children",
        remote_side=[id],
        lazy="selectin",
    )
    children: Mapped[list[SQAAnalysisCard]] = relationship(
        "SQAAnalysisCard",
        cascade="all, delete-orphan",
        back_populates="parent",
        lazy="selectin",
    )
