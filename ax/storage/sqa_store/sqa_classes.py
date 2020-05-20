#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime
from typing import Any, Dict, List, Optional

from ax.core.base_trial import TrialStatus
from ax.core.parameter import ParameterType
from ax.core.types import (
    ComparisonOp,
    TModelPredict,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.storage.sqa_store.db import (
    LONG_STRING_FIELD_LENGTH,
    LONGTEXT_BYTES,
    NAME_OR_TYPE_FIELD_LENGTH,
    Base,
)
from ax.storage.sqa_store.json import (
    JSONEncodedDict,
    JSONEncodedList,
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


ONLY_ONE_FIELDS = ["experiment_id", "generator_run_id"]


ONLY_ONE_METRIC_FIELDS = ["scalarized_objective_id"]


class SQAParameter(Base):
    __tablename__: str = "parameter_v2"

    domain_type: DomainType = Column(IntEnum(DomainType), nullable=False)
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: int = Column(Integer, primary_key=True)
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    parameter_type: ParameterType = Column(IntEnum(ParameterType), nullable=False)
    is_fidelity: Optional[bool] = Column(Boolean)
    target_value: Optional[TParamValue] = Column(JSONEncodedObject)

    # Attributes for Range Parameters
    digits: Optional[int] = Column(Integer)
    log_scale: Optional[bool] = Column(Boolean)
    lower: Optional[float] = Column(Float)
    upper: Optional[float] = Column(Float)

    # Attributes for Choice Parameters
    choice_values: Optional[List[TParamValue]] = Column(JSONEncodedList)
    is_ordered: Optional[bool] = Column(Boolean)
    is_task: Optional[bool] = Column(Boolean)

    # Attributes for Fixed Parameters
    fixed_value: Optional[TParamValue] = Column(JSONEncodedObject)

    immutable_fields = ["name"]
    unique_id = "name"


class SQAParameterConstraint(Base):
    __tablename__: str = "parameter_constraint_v2"

    bound: float = Column(Float, nullable=False)
    constraint_dict: Dict[str, float] = Column(JSONEncodedDict, nullable=False)
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    id: int = Column(Integer, primary_key=True)
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    type: IntEnum = Column(IntEnum(ParameterConstraintType), nullable=False)

    # ParameterConstraints should never be updated; since they don't have
    # a field that can be used for a UID, if anything changes,
    # we should just throw them out and recreate them
    immutable_fields = ["type", "constraint_dict", "bound"]


class SQAMetric(Base):
    __tablename__: str = "metric_v2"

    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    id: int = Column(Integer, primary_key=True)
    lower_is_better: Optional[bool] = Column(Boolean)
    intent: MetricIntent = Column(StringEnum(MetricIntent), nullable=False)
    metric_type: int = Column(Integer, nullable=False)
    name: str = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})

    # Attributes for Objectives
    minimize: Optional[bool] = Column(Boolean)

    # Attributes for Outcome Constraints
    op: Optional[ComparisonOp] = Column(IntEnum(ComparisonOp))
    bound: Optional[float] = Column(Float)
    relative: Optional[bool] = Column(Boolean)

    # Multi-type Experiment attributes
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    canonical_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))

    immutable_fields = ["name"]
    unique_id = "name"

    scalarized_objective_id = Column(Integer, ForeignKey("metric_v2.id"))

    # Relationship containing SQAMetric(s) only defined for the parent metric
    # of Multi/Scalarized Objective contains all children of the parent metric
    # join_depth argument: used for loading self-referential relationships
    # https://docs.sqlalchemy.org/en/13/orm/self_referential.html#configuring-self-referential-eager-loading
    scalarized_objective_children_metrics = relationship(
        "SQAMetric", cascade="all, delete-orphan", lazy="selectin", join_depth=4
    )

    # Attribute only defined for the children of Scalarized Objective
    scalarized_objective_weight: Optional[float] = Column(Float)


class SQAArm(Base):
    __tablename__: str = "arm_v2"

    generator_run_id: int = Column(Integer, ForeignKey("generator_run_v2.id"))
    id: int = Column(Integer, primary_key=True)
    name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    parameters: TParameterization = Column(JSONEncodedTextDict, nullable=False)
    weight: float = Column(Float, nullable=False, default=1.0)

    immutable_fields = ["parameters"]
    unique_id = "name"


class SQAAbandonedArm(Base):
    __tablename__: str = "abandoned_arm_v2"

    abandoned_reason: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    time_abandoned: datetime = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    trial_id: int = Column(Integer, ForeignKey("trial_v2.id"))

    immutable_fields = ["name"]
    unique_id = "name"


class SQAGeneratorRun(Base):
    __tablename__: str = "generator_run_v2"

    best_arm_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    best_arm_parameters: Optional[TParameterization] = Column(JSONEncodedTextDict)
    best_arm_predictions: Optional[TModelPredictArm] = Column(JSONEncodedList)
    generator_run_type: Optional[int] = Column(Integer)
    id: int = Column(Integer, primary_key=True)
    index: Optional[int] = Column(Integer)
    model_predictions: Optional[TModelPredict] = Column(JSONEncodedList)
    time_created: datetime = Column(IntTimestamp, nullable=False, default=datetime.now)
    trial_id: Optional[int] = Column(Integer, ForeignKey("trial_v2.id"))
    weight: Optional[float] = Column(Float)
    fit_time: Optional[float] = Column(Float)
    gen_time: Optional[float] = Column(Float)
    model_key: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    model_kwargs: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    bridge_kwargs: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    gen_metadata: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    model_state_after_gen: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    generation_strategy_id: Optional[int] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    generation_step_index: Optional[int] = Column(Integer)
    candidate_metadata_by_arm_signature: Optional[Dict[str, Any]] = Column(
        JSONEncodedTextDict
    )

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

    ignore_during_update_fields = ["time_created"]
    unique_id = "index"


class SQARunner(Base):
    __tablename__: str = "runner"

    id: int = Column(Integer, primary_key=True)
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})
    runner_type: int = Column(Integer, nullable=False)
    trial_id: Optional[int] = Column(Integer, ForeignKey("trial_v2.id"))

    # Multi-type Experiment attributes
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))


class SQAData(Base):
    __tablename__: str = "data_v2"

    id: int = Column(Integer, primary_key=True)
    data_json: str = Column(Text(LONGTEXT_BYTES), nullable=False)
    description: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_id: int = Column(Integer, ForeignKey("experiment_v2.id"))
    time_created: int = Column(BigInteger, nullable=False)
    trial_index: Optional[int] = Column(Integer)
    generation_strategy_id: Optional[int] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )

    unique_id = "time_created"


class SQAGenerationStrategy(Base):
    __tablename__: str = "generation_strategy"

    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    steps: List[Dict[str, Any]] = Column(JSONEncodedList, nullable=False)
    curr_index: int = Column(Integer, nullable=False)
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))

    generator_runs: List[SQAGeneratorRun] = relationship(
        "SQAGeneratorRun",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAGeneratorRun.id,
    )

    data: SQAData = relationship(
        "SQAData", cascade="all, delete-orphan", lazy=False, uselist=False
    )


class SQATrial(Base):
    __tablename__: str = "trial_v2"

    abandoned_reason: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    deployed_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    experiment_id: int = Column(Integer, ForeignKey("experiment_v2.id"))
    id: int = Column(Integer, primary_key=True)
    index: int = Column(Integer, index=True, nullable=False)
    is_batch: bool = Column("is_batched", Boolean, nullable=False, default=True)
    num_arms_created: int = Column(Integer, nullable=False, default=0)
    optimize_for_power: Optional[bool] = Column(Boolean)
    ttl_seconds: Optional[int] = Column(Boolean)
    run_metadata: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    status: TrialStatus = Column(
        IntEnum(TrialStatus), nullable=False, default=TrialStatus.CANDIDATE
    )
    status_quo_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    time_completed: Optional[datetime] = Column(IntTimestamp)
    time_created: datetime = Column(IntTimestamp, nullable=False)
    time_staged: Optional[datetime] = Column(IntTimestamp)
    time_run_started: Optional[datetime] = Column(IntTimestamp)
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    generation_step_index: Optional[int] = Column(Integer)
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})

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

    unique_id = "index"
    ignore_during_update_fields = ["time_created"]
    immutable_fields = ["is_batch"]


class SQAExperiment(Base):
    __tablename__: str = "experiment_v2"

    description: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    experiment_type: Optional[int] = Column(Integer)
    id: int = Column(Integer, primary_key=True)
    is_test: bool = Column(Boolean, nullable=False, default=False)
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})
    status_quo_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    status_quo_parameters: Optional[TParameterization] = Column(JSONEncodedTextDict)
    time_created: datetime = Column(IntTimestamp, nullable=False)
    default_trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))

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
    generation_strategy: Optional[SQAGenerationStrategy] = relationship(
        "SQAGenerationStrategy",
        backref=backref("experiment", lazy=False),
        uselist=False,
        lazy="selectin",
    )

    immutable_fields = ["name"]
    ignore_during_update_fields = ["time_created"]
