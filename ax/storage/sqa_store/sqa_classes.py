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

    # pyre-fixme[8]: Attribute has type `DomainType`; used as `Column[typing.Any]`.
    domain_type: DomainType = Column(IntEnum(DomainType), nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    # pyre-fixme[8]: Attribute has type `ParameterType`; used as `Column[typing.Any]`.
    parameter_type: ParameterType = Column(IntEnum(ParameterType), nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    is_fidelity: Optional[bool] = Column(Boolean)
    # pyre-fixme[8]: Attribute has type `Union[None, bool, float, int, str]`; used
    #  as `Column[typing.Any]`.
    target_value: Optional[TParamValue] = Column(JSONEncodedObject)

    # Attributes for Range Parameters
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    digits: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    log_scale: Optional[bool] = Column(Boolean)
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    lower: Optional[float] = Column(Float)
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    upper: Optional[float] = Column(Float)

    # Attributes for Choice Parameters
    # pyre-fixme[8]: Attribute has type `Optional[List[typing.Union[None, bool,
    #  float, int, str]]]`; used as `Column[typing.Any]`.
    choice_values: Optional[List[TParamValue]] = Column(JSONEncodedList)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    is_ordered: Optional[bool] = Column(Boolean)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    is_task: Optional[bool] = Column(Boolean)

    # Attributes for Fixed Parameters
    # pyre-fixme[8]: Attribute has type `Union[None, bool, float, int, str]`; used
    #  as `Column[typing.Any]`.
    fixed_value: Optional[TParamValue] = Column(JSONEncodedObject)


class SQAParameterConstraint(Base):
    __tablename__: str = "parameter_constraint_v2"

    # pyre-fixme[8]: Attribute has type `float`; used as `Column[decimal.Decimal]`.
    bound: float = Column(Float, nullable=False)
    # pyre-fixme[8]: Attribute has type `Dict[str, float]`; used as
    #  `Column[typing.Any]`.
    constraint_dict: Dict[str, float] = Column(JSONEncodedDict, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    # pyre-fixme[8]: Attribute has type `IntEnum`; used as `Column[typing.Any]`.
    type: IntEnum = Column(IntEnum(ParameterConstraintType), nullable=False)


class SQAMetric(Base):
    __tablename__: str = "metric_v2"

    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generator_run_id: Optional[int] = Column(Integer, ForeignKey("generator_run_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    lower_is_better: Optional[bool] = Column(Boolean)
    # pyre-fixme[8]: Attribute has type `MetricIntent`; used as `Column[typing.Any]`.
    intent: MetricIntent = Column(StringEnum(MetricIntent), nullable=False)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    metric_type: int = Column(Integer, nullable=False)
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    name: str = Column(String(LONG_STRING_FIELD_LENGTH), nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})

    # Attributes for Objectives
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    minimize: Optional[bool] = Column(Boolean)

    # Attributes for Outcome Constraints
    # pyre-fixme[8]: Attribute has type `Optional[ComparisonOp]`; used as
    #  `Column[typing.Any]`.
    op: Optional[ComparisonOp] = Column(IntEnum(ComparisonOp))
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    bound: Optional[float] = Column(Float)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    relative: Optional[bool] = Column(Boolean)

    # Multi-type Experiment attributes
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    canonical_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))

    scalarized_objective_id = Column(Integer, ForeignKey("metric_v2.id"))

    # Relationship containing SQAMetric(s) only defined for the parent metric
    # of Multi/Scalarized Objective contains all children of the parent metric
    # join_depth argument: used for loading self-referential relationships
    # https://docs.sqlalchemy.org/en/13/orm/self_referential.html#configuring-self-referential-eager-loading
    scalarized_objective_children_metrics = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_objective_id],
    )

    # Attribute only defined for the children of Scalarized Objective
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    scalarized_objective_weight: Optional[float] = Column(Float)

    scalarized_outcome_constraint_id = Column(Integer, ForeignKey("metric_v2.id"))
    scalarized_outcome_constraint_children_metrics = relationship(
        "SQAMetric",
        cascade="all, delete-orphan",
        lazy=True,
        foreign_keys=[scalarized_outcome_constraint_id],
    )
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    scalarized_outcome_constraint_weight: Optional[float] = Column(Float)


class SQAArm(Base):
    __tablename__: str = "arm_v2"

    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    generator_run_id: int = Column(Integer, ForeignKey("generator_run_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Dict[str, typing.Union[None, bool, float,
    #  int, str]]`; used as `Column[typing.Any]`.
    parameters: TParameterization = Column(JSONEncodedTextDict, nullable=False)
    # pyre-fixme[8]: Attribute has type `float`; used as `Column[decimal.Decimal]`.
    weight: float = Column(Float, nullable=False, default=1.0)


class SQAAbandonedArm(Base):
    __tablename__: str = "abandoned_arm_v2"

    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    abandoned_reason: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Column[typing.Any]`.
    time_abandoned: datetime = Column(
        IntTimestamp, nullable=False, default=datetime.now
    )
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    trial_id: int = Column(Integer, ForeignKey("trial_v2.id"))


class SQAGeneratorRun(Base):
    __tablename__: str = "generator_run_v2"

    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    best_arm_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Union[None, bool,
    #  float, int, str]]]`; used as `Column[typing.Any]`.
    best_arm_parameters: Optional[TParameterization] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[typing.Tuple[Dict[str, float],
    #  Optional[Dict[str, Dict[str, float]]]]]`; used as `Column[typing.Any]`.
    best_arm_predictions: Optional[TModelPredictArm] = Column(JSONEncodedList)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generator_run_type: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    index: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[typing.Tuple[Dict[str,
    #  List[float]], Dict[str, Dict[str, List[float]]]]]`; used as
    #  `Column[typing.Any]`.
    model_predictions: Optional[TModelPredict] = Column(JSONEncodedList)
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Column[typing.Any]`.
    time_created: datetime = Column(IntTimestamp, nullable=False, default=datetime.now)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    trial_id: Optional[int] = Column(Integer, ForeignKey("trial_v2.id"))
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    weight: Optional[float] = Column(Float)
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    fit_time: Optional[float] = Column(Float)
    # pyre-fixme[8]: Attribute has type `Optional[float]`; used as
    #  `Column[decimal.Decimal]`.
    gen_time: Optional[float] = Column(Float)
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    model_key: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    model_kwargs: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    bridge_kwargs: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    gen_metadata: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    model_state_after_gen: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generation_strategy_id: Optional[int] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generation_step_index: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
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


class SQARunner(Base):
    __tablename__: str = "runner"

    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    runner_type: int = Column(Integer, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    trial_id: Optional[int] = Column(Integer, ForeignKey("trial_v2.id"))

    # Multi-type Experiment attributes
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))


class SQAData(Base):
    __tablename__: str = "data_v2"

    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    data_json: str = Column(Text(LONGTEXT_BYTES), nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    description: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    experiment_id: int = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    time_created: int = Column(BigInteger, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    trial_index: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generation_strategy_id: Optional[int] = Column(
        Integer, ForeignKey("generation_strategy.id")
    )
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    structure_metadata_json: str = Column(Text(LONGTEXT_BYTES), nullable=True)


class SQAGenerationStrategy(Base):
    __tablename__: str = "generation_strategy"

    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    # pyre-fixme[8]: Attribute has type `List[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    steps: List[Dict[str, Any]] = Column(JSONEncodedList, nullable=False)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    curr_index: int = Column(Integer, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_id: Optional[int] = Column(Integer, ForeignKey("experiment_v2.id"))

    generator_runs: List[SQAGeneratorRun] = relationship(
        "SQAGeneratorRun",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by=lambda: SQAGeneratorRun.id,
    )


class SQATrial(Base):
    __tablename__: str = "trial_v2"

    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    abandoned_reason: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    deployed_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    experiment_id: int = Column(Integer, ForeignKey("experiment_v2.id"))
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    index: int = Column(Integer, index=True, nullable=False)
    # pyre-fixme[8]: Attribute has type `bool`; used as `Column[bool]`.
    is_batch: bool = Column("is_batched", Boolean, nullable=False, default=True)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    num_arms_created: int = Column(Integer, nullable=False, default=0)
    # pyre-fixme[8]: Attribute has type `Optional[bool]`; used as `Column[bool]`.
    optimize_for_power: Optional[bool] = Column(Boolean)
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    ttl_seconds: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    run_metadata: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    stop_metadata: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `TrialStatus`; used as `Column[typing.Any]`.
    status: TrialStatus = Column(
        IntEnum(TrialStatus), nullable=False, default=TrialStatus.CANDIDATE
    )
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    status_quo_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[datetime]`; used as
    #  `Column[typing.Any]`.
    time_completed: Optional[datetime] = Column(IntTimestamp)
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Column[typing.Any]`.
    time_created: datetime = Column(IntTimestamp, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[datetime]`; used as
    #  `Column[typing.Any]`.
    time_staged: Optional[datetime] = Column(IntTimestamp)
    # pyre-fixme[8]: Attribute has type `Optional[datetime]`; used as
    #  `Column[typing.Any]`.
    time_run_started: Optional[datetime] = Column(IntTimestamp)
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    generation_step_index: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
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


class SQAExperiment(Base):
    __tablename__: str = "experiment_v2"

    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    description: Optional[str] = Column(String(LONG_STRING_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[int]`; used as `Column[int]`.
    experiment_type: Optional[int] = Column(Integer)
    # pyre-fixme[8]: Attribute has type `int`; used as `Column[int]`.
    id: int = Column(Integer, primary_key=True)
    # pyre-fixme[8]: Attribute has type `bool`; used as `Column[bool]`.
    is_test: bool = Column(Boolean, nullable=False, default=False)
    # pyre-fixme[8]: Attribute has type `str`; used as `Column[str]`.
    name: str = Column(String(NAME_OR_TYPE_FIELD_LENGTH), nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Any]]`; used as
    #  `Column[typing.Any]`.
    properties: Optional[Dict[str, Any]] = Column(JSONEncodedTextDict, default={})
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    status_quo_name: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `Optional[Dict[str, typing.Union[None, bool,
    #  float, int, str]]]`; used as `Column[typing.Any]`.
    status_quo_parameters: Optional[TParameterization] = Column(JSONEncodedTextDict)
    # pyre-fixme[8]: Attribute has type `datetime`; used as `Column[typing.Any]`.
    time_created: datetime = Column(IntTimestamp, nullable=False)
    # pyre-fixme[8]: Attribute has type `Optional[str]`; used as `Column[str]`.
    default_trial_type: Optional[str] = Column(String(NAME_OR_TYPE_FIELD_LENGTH))
    # pyre-fixme[8]: Attribute has type `DataType`; used as `Column[typing.Any]`.
    default_data_type: DataType = Column(IntEnum(DataType), nullable=True)

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
        backref=backref("experiment", lazy=True),
        uselist=False,
        lazy=True,
    )
