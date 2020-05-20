#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

from ax.storage.sqa_store.db import SQABase
from ax.storage.sqa_store.sqa_classes import (
    ONLY_ONE_FIELDS,
    ONLY_ONE_METRIC_FIELDS,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
)
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.orm.mapper import Mapper


def consistency_exactly_one(instance: SQABase, exactly_one_fields: List[str]) -> Any:
    """Ensure that exactly one of `exactly_one_fields` has a value set."""
    values = [getattr(instance, field) is not None for field in exactly_one_fields]
    if sum(values) != 1:
        raise ValueError(
            f"{instance.__class__.__name__} must have exactly one of the following "
            f"fields set: {', '.join(exactly_one_fields)}."
        )


@event.listens_for(SQAParameter, "before_insert")
@event.listens_for(SQAParameter, "before_update")
def validate_parameter(mapper: Mapper, connection: Connection, target: SQABase) -> None:
    consistency_exactly_one(target, ONLY_ONE_FIELDS)


@event.listens_for(SQAParameterConstraint, "before_insert")
@event.listens_for(SQAParameterConstraint, "before_update")
def validate_parameter_constraint(
    mapper: Mapper, connection: Connection, target: SQABase
) -> None:
    consistency_exactly_one(target, ONLY_ONE_FIELDS)


@event.listens_for(SQAMetric, "before_insert")
@event.listens_for(SQAMetric, "before_update")
def validate_metric(mapper: Mapper, connection: Connection, target: SQABase) -> None:
    consistency_exactly_one(target, ONLY_ONE_FIELDS + ONLY_ONE_METRIC_FIELDS)


@event.listens_for(SQARunner, "before_insert")
@event.listens_for(SQARunner, "before_update")
def validate_runner(mapper: Mapper, connection: Connection, target: SQABase) -> None:
    consistency_exactly_one(target, ["experiment_id", "trial_id"])
