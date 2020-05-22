#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, List, Optional, TypeVar

import numpy as np
from ax.exceptions.storage import ImmutabilityError
from ax.storage.sqa_store.utils import is_foreign_key_field
from ax.utils.common.equality import datetime_equals, equality_typechecker
from ax.utils.common.typeutils import numpy_type_to_python_type
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, scoped_session, sessionmaker


# some constants for database fields
HASH_FIELD_LENGTH: int = 32
NAME_OR_TYPE_FIELD_LENGTH: int = 100
LONG_STRING_FIELD_LENGTH: int = 255
JSON_FIELD_LENGTH: int = 4096

# by default, Text gets mapped to a TEXT field in MySQL is 2^16 - 1
# we use have MEDIUMTEXT and LONGTEXT in the MySQL db; in this case, use
# Text(MEDIUMTEXT_BYTES) or Text(LONGTEXT_BYTES). This is preferable to
# using MEDIUMTEXT and LONGTEXT directly because those are incompatible with
# SQLite that is used in unit tests.
MEDIUMTEXT_BYTES: int = 2 ** 24 - 1
LONGTEXT_BYTES: int = 2 ** 32 - 1

# global database variables
Ax_PROD_TIER: str = "xdb.adaptive_experiment"
SESSION_FACTORY: Optional[Session] = None

# set this to false to prevent SQLAlchemy for automatically expiring objects
# on commit, which essentially makes them unusable outside of a session
# see e.g. https://stackoverflow.com/a/50272761
EXPIRE_ON_COMMIT = False

T = TypeVar("T")


class SQABase:
    """Metaclass for SQLAlchemy classes corresponding to core Ax classes."""

    @property
    def attributes(self):
        """Return a list of the column attributes and relationship fields on this
        SQABase instance. Used for iterating over the fields to determine equality,
        perform updates, etc.
        """
        mapper = inspect(self).mapper
        attrs = [c.key for c in mapper.column_attrs]

        # exclude backpopulated relationships; those will be accounted for on the
        # owning class
        relationships = [c.key for c in mapper.relationships if not c.back_populates]
        return attrs + relationships

    @staticmethod
    def list_equals(l1: List[T], l2: List[T]) -> bool:
        """Compare equality of two lists.

        Assumptions:
            -- The lists do not contain duplicates

        Checking equality is then the same as checking that the lists are the same
        length, and that one is a subset of the other.
        """
        if len(l1) != len(l2):
            return False
        for x in l1:
            for y in l2:
                # pyre-fixme[16]: `Dict` has no attribute `__ne__`.
                if type(x) != type(y):
                    equal = False
                if isinstance(x, SQABase):
                    equal = x.equals(y)
                elif isinstance(x, (int, float, str, bool, dict, Enum)):
                    equal = x == y
                else:
                    raise ValueError(
                        f"Calling list_equals on unsupported types: "
                        f"{type(x) and {type(y)}}"
                    )
                if equal:
                    break
            else:
                return False
        return True

    @staticmethod
    def list_update(l1: List[T], l2: List[T]) -> List[T]:
        """Given an existing list (`l1`) and an new version (`l2`):
           -- update the existing items in `l1` that have matching items in `l2`
           -- delete existing items in `l1` that don't have matching items in `l2`
           -- add items in `l2` that don't exist in `l1`

        e.g. list_update([1,2,3], [1,5]) => [1,5]
             list_update([Arm(name="0_0")], [Arm(name="0_0"), Arm(name="0_1")]) =>
                [Arm(name="0_0"), Arm(name="0_1")]
                where Arm(name="0_0") has been updated, not replaced, so that we
                don't delete/recreate the DB row
        """
        if not l1 and not l2:
            return l1

        types = {type(x) for x in l1 + l2}
        primitive_types = {int, float, str, bool, dict, Enum}
        if types.issubset(primitive_types):
            # No need to do a special update here; just return the new list
            return l2

        if len(types) > 1:
            raise ValueError(
                "Cannot call `list_update` on lists that contain a mix of "
                f"primitive and no primitive types: ({l1} and {l2})."
            )

        type_ = list(types)[0]

        if not issubclass(type_, SQABase):
            raise ValueError(f"Calling list_update on unsupported type {type_}.")

        unique_id = getattr(type_, "unique_id", None)
        if unique_id is None:
            return SQABase.list_update_without_unique_id(l1, l2)

        l1_dict = {getattr(x, unique_id): x for x in l1}
        l2_dict = {getattr(x, unique_id): x for x in l2}
        if len(l1_dict) != len(l1) or len(l2_dict) != len(l2):
            # If unique_ids aren't actually unique (could happen if all values
            # are None), act as if there are no unique ids at all
            return SQABase.list_update_without_unique_id(l1, l2)  # pragma: no cover

        new_list = []
        for key, new_val in l2_dict.items():
            # For each item in the new list, try to find a match in the old list.
            if key in l1_dict:
                # If there is a matching item in the old list, update it.
                old_val = l1_dict[key]
                # pyre-fixme[16]: `Variable[T]` has no attribute `update`.
                old_val.update(new_val)
                new_list.append(old_val)
            else:
                # If there is no matching item, append the new item.
                new_list.append(new_val)
        return new_list

    @staticmethod
    def list_update_without_unique_id(l1: List[T], l2: List[T]) -> List[T]:
        """Merge a new list (`l2`) into an existing list (`l1`)
        This method works for lists whose items do not have a unique_id field.
        If the lists are equal, return the old one. Else, return the new one.
        """
        if SQABase.list_equals(l1, l2):
            return l1
        return l2

    @equality_typechecker
    def equals(self, other: "SQABase") -> bool:
        """Check if `other` equals `self.`"""
        for field in self.attributes:
            if field in ["id", "_sa_instance_state"] or is_foreign_key_field(field):
                # We don't want to perform equality checks on foreign key fields,
                # since our equality checks are used to determine whether or not
                # to a new object is the same as an existing one. The new object
                # will always have None for its foreign key fields, because it
                # hasn't been inserted into the database yet.
                continue
            if not self.fields_equal(other, field):
                return False
        return True

    def update(self, other: "SQABase") -> None:
        """Merge `other` into `self.`"""
        ignore_during_update_fields = set(
            getattr(self, "ignore_during_update_fields", [])
            + ["id", "_sa_instance_state"]
        )
        immutable_fields = set(getattr(self, "immutable_fields", []))
        for field in self.attributes:
            if field in immutable_fields:
                if self.fields_equal(other, field):
                    continue
                raise ImmutabilityError(
                    f"Cannot change `{field}` of {self.__class__.__name__}."
                )
            if (
                field in ignore_during_update_fields
                # We don't want to update foreign key fields, e.g. experiment_id.
                # The new object will always have a value of None for this field,
                # but we don't want to overwrite the value on the existing object.
                or is_foreign_key_field(field)
            ):
                continue
            self.update_field(other, field)

    def update_field(self, other: "SQABase", field: str) -> None:
        """Update `field` on `self` to be equal to `field` on `other`."""
        self_val = getattr(self, field)
        other_val = getattr(other, field)
        if isinstance(self_val, list) and isinstance(other_val, list):
            other_val = SQABase.list_update(self_val, other_val)
        elif isinstance(self_val, SQABase) and isinstance(other_val, SQABase):
            self_val.update(other_val)
            other_val = self_val
        elif self.fields_equal(other, field):
            return
        setattr(self, field, other_val)

    def fields_equal(self, other: "SQABase", field: str) -> bool:
        """Check if `field` on `self` is equal to `field` on `other`."""
        self_val = getattr(self, field)
        other_val = getattr(other, field)
        self_val = numpy_type_to_python_type(self_val)
        other_val = numpy_type_to_python_type(other_val)
        if type(self_val) != type(other_val):
            return False
        if isinstance(self_val, list):
            return SQABase.list_equals(self_val, other_val)
        elif isinstance(self_val, SQABase):
            return self_val.equals(other_val)
        elif isinstance(self_val, datetime):
            return datetime_equals(self_val, other_val)
        elif isinstance(self_val, float):
            return np.isclose(self_val, other_val)
        else:
            return self_val == other_val


Base = declarative_base(cls=SQABase)


def create_mysql_engine_from_creator(
    creator: Callable, echo: bool = False, pool_recycle: int = 10, **kwargs: Any
) -> Engine:
    """Create a SQLAlchemy engine with the MySQL dialect given a creator function.

    Args:
        creator:  a callable which returns a DBAPI connection.
        echo: if True, set engine to be verbose.
        pool_recycle: number of seconds after which to recycle
            connections. -1 means no timeout. Default is 10 seconds.
        **kwargs: keyword args passed to `create_engine`

    Returns:
        Engine: SQLAlchemy engine with connection to MySQL DB.

    """
    return create_engine(
        "mysql://", creator=creator, pool_recycle=pool_recycle, echo=echo, **kwargs
    )


def create_mysql_engine_from_url(
    url: str, echo: bool = False, pool_recycle: int = 10, **kwargs: Any
) -> Engine:
    """Create a SQLAlchemy engine with the MySQL dialect given a database url.

    Args:
        url: a database url that can include username, password, hostname, database name
            as well as optional keyword arguments for additional configuration.
            e.g. `dialect+driver://username:password@host:port/database`.
        echo: if True, set engine to be verbose.
        pool_recycle: number of seconds after which to recycle
            connections. -1 means no timeout. Default is 10 seconds.
        **kwargs: keyword args passed to `create_engine`

    Returns:
        Engine: SQLAlchemy engine with connection to MySQL DB.

    """
    return create_engine(
        name_or_url=url, pool_recycle=pool_recycle, echo=echo, **kwargs
    )


def create_test_engine(path: Optional[str] = None, echo: bool = True) -> Engine:
    """Creates a SQLAlchemy engine object for use in unit tests.

    Args:
        path: if None, use in-memory SQLite; else
            attempt to create a SQLite DB in the path provided.
        echo: if True, set engine to be verbose.

    Returns:
        Engine: an instance of SQLAlchemy engine.

    """
    if path is None:
        db_path = "sqlite://"
    else:
        db_path = "sqlite:///{path}".format(path=path)
    return create_engine(db_path, echo=echo)


def init_engine_and_session_factory(
    url: Optional[str] = None,
    creator: Optional[Callable] = None,
    echo: bool = False,
    force_init: bool = False,
    **kwargs: Any,
) -> None:
    """Initialize the global engine and SESSION_FACTORY for SQLAlchemy.

    The initialization needs to only happen once. Note that it is possible to
    re-initialize the engine by setting the `force_init` flag to True, but this
    should only be used if you are absolutely certain that you know what you
    are doing.

    Args:
        url: a database url that can include username, password, hostname, database name
            as well as optional keyword arguments for additional configuration.
            e.g. `dialect+driver://username:password@host:port/database`.
            Either this argument or `creator` argument must be specified.
        creator: a callable which returns a DBAPI connection.
            Either this argument or `url` argument must be specified.
        echo: if True, logging for engine is enabled.
        force_init: if True, allows re-initializing engine
            and session factory.
        **kwargs: keyword arguments passed to `create_mysql_engine_from_creator`

    """
    global SESSION_FACTORY

    if SESSION_FACTORY is not None:
        if force_init:
            # pyre-fixme[16]: `Optional` has no attribute `bind`.
            SESSION_FACTORY.bind.dispose()
        else:
            return  # pragma: no cover
    if url is not None:
        engine = create_mysql_engine_from_url(url=url, echo=echo, **kwargs)
    elif creator is not None:
        engine = create_mysql_engine_from_creator(creator=creator, echo=echo, **kwargs)
    else:
        raise ValueError("Must specify either `url` or `creator`.")  # pragma: no cover
    SESSION_FACTORY = scoped_session(
        sessionmaker(bind=engine, expire_on_commit=EXPIRE_ON_COMMIT)
    )


def init_test_engine_and_session_factory(
    tier_or_path: Optional[str] = None,
    echo: bool = False,
    force_init: bool = False,
    **kwargs: Any,
) -> None:
    """Initialize the global engine and SESSION_FACTORY for SQLAlchemy,
    using an in-memory SQLite database.

    The initialization needs to only happen once. Note that it is possible to
    re-initialize the engine by setting the `force_init` flag to True, but this
    should only be used if you are absolutely certain that you know what you
    are doing.

    Args:
        tier_or_path: the name of the DB tier.
        echo: if True, logging for engine is enabled.
        force_init: if True, allows re-initializing engine
            and session factory.
        **kwargs: keyword arguments passed to `create_mysql_engine_from_creator`

    """
    global SESSION_FACTORY

    if SESSION_FACTORY is not None:
        if force_init:
            # pyre-fixme[16]: `Optional` has no attribute `bind`.
            SESSION_FACTORY.bind.dispose()
        else:
            return
    engine = create_test_engine(path=tier_or_path, echo=echo)
    create_all_tables(engine)

    SESSION_FACTORY = scoped_session(
        sessionmaker(bind=engine, expire_on_commit=EXPIRE_ON_COMMIT)
    )


def create_all_tables(engine: Engine) -> None:
    """Create all tables that inherit from Base.

    Args:
        engine: a SQLAlchemy engine with a connection to a MySQL
            or SQLite DB.

    Note:
        In order for all tables to be correctly created, all modules that
        define a mapped class that inherits from `Base` must be imported.

    """
    if (
        engine.dialect.name == "mysql"
        and engine.dialect.default_schema_name == "adaptive_experiment"
    ):
        raise Exception("Cannot mutate tables in XDB. Use AOSC.")  # pragma: no cover
    # pyre-fixme[16]: `Base` has no attribute `metadata`.
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Fetch a SQLAlchemy session with a connection to a DB.

    Unless `init_engine_and_session_factory` is called first with custom
    args, this will automatically initialize a connection to
    `xdb.adaptive_experiment`.

    Returns:
        Session: an instance of a SQLAlchemy session.

    """
    global SESSION_FACTORY
    if SESSION_FACTORY is None:
        init_engine_and_session_factory()  # pragma: no cover
    assert SESSION_FACTORY is not None
    # pyre-fixme[29]: `Session` is not a function.
    return SESSION_FACTORY()


def get_engine() -> Engine:
    """Fetch a SQLAlchemy engine, if already initialized.

    If not initialized, need to either call `init_engine_and_session_factory` or
    `get_session` explicitly.

    Returns:
        Engine: an instance of a SQLAlchemy engine with a connection to a DB.

    """
    global SESSION_FACTORY
    if SESSION_FACTORY is None:
        raise ValueError("Engine must be initialized first.")  # pragma: no cover
    # pyre-fixme[16]: `Optional` has no attribute `bind`.
    return SESSION_FACTORY.bind


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:  # pragma: no cover
        session.rollback()  # pragma: no cover
        raise  # pragma: no cover
    finally:
        session.close()
