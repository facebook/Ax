#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Callable, List, TypeVar

from ax.storage.sqa_store.db import SQABase
from ax.storage.sqa_store.reduced_state import GR_LARGE_MODEL_ATTRS
from ax.storage.sqa_store.sqa_classes import (
    ONLY_ONE_FIELDS,
    ONLY_ONE_METRIC_FIELDS,
    SQAMetric,
    SQAParameter,
    SQAParameterConstraint,
    SQARunner,
)
from ax.utils.common.logger import get_logger
from sqlalchemy import event
from sqlalchemy.engine import Connection
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.base import NO_VALUE
from sqlalchemy.orm.mapper import Mapper

T = TypeVar("T")


logger: Logger = get_logger(__name__)


def listens_for_multiple(
    targets: List[InstrumentedAttribute],
    identifier: str,
    *args: Any,
    **kwargs: Any
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
) -> Callable:
    """Analogue of SQLAlchemy `listen_for`, but applies the same listening handler
    function to multiple instrumented attributes.
    """

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def wrapper(fn: Callable):
        for target in targets:
            event.listen(target, identifier, fn, *args, **kwargs)
        return fn

    return wrapper


# pyre-fixme[3]: Return annotation cannot be `Any`.
def consistency_exactly_one(instance: SQABase, exactly_one_fields: List[str]) -> Any:
    """Ensure that exactly one of `exactly_one_fields` has a value set."""
    values = [getattr(instance, field) is not None for field in exactly_one_fields]
    if sum(values) != 1:
        raise ValueError(
            f"{instance.__class__.__name__} must have exactly one of the following "
            f"fields set: {', '.join(exactly_one_fields)}."
        )


@listens_for_multiple(
    targets=GR_LARGE_MODEL_ATTRS,
    identifier="set",
    # `retval=True` instruct the operation ('set' on attributes in `targets`) to use
    # the return value of decorated function to set the attribute.
    retval=True,
    # `propagate=True` ensures that targets with subclasses of SQA classes used by
    # default Ax OSS encoder inherit the event listeners.
    propagate=True,
)
def do_not_set_existing_value_to_null(
    instance: SQABase, new_value: T, old_value: T, initiator_event: event.Events
) -> T:
    no_value = [None, NO_VALUE]
    if new_value in no_value and old_value not in no_value:
        logger.debug(
            f"New value for attribute is `None` or has no value, but old value "
            f"was set, so keeping the old value ({old_value})."
        )
        return old_value
    return new_value


@event.listens_for(
    SQAParameter,
    "before_insert",
)
@event.listens_for(
    SQAParameter,
    "before_update",
)
# pyre-fixme[11]: Annotation `Mapper` is not defined as a type.
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
