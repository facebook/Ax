#!/usr/bin/env python3

from abc import ABCMeta
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, List, Optional, TypeVar

from ae.lazarus.ae.exceptions.storage import ImmutabilityError
from ae.lazarus.ae.storage.sqa_store.utils import is_foreign_key_field
from ae.lazarus.ae.utils.common.equality import datetime_equals, equality_typechecker
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
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
AE_PROD_TIER: str = "xdb.adaptive_experiment"
SESSION_FACTORY: Optional[Session] = None

T = TypeVar("T")


class DeclarativeABCMeta(DeclarativeMeta, ABCMeta):
    pass


class SQABase(object):
    """Metaclass for SQLAlchemy classes corresponding to core AE classes."""

    @property
    def attributes(self):
        """Return a list of the column attributes and relationship fields on this
        SQABase instance. Used for iterating over the fields to determine equality,
        perform updates, etc.
        """
        mapper = inspect(self).mapper
        attrs = [c.key for c in mapper.column_attrs]

        # exclude backref relationships; those will be accounted for on the
        # backpopulated class
        relationships = [c.key for c in mapper.relationships if not c.backref]
        return attrs + relationships

    @staticmethod
    def list_equals(l1: List[T], l2: List[T]) -> bool:
        """Compare equality of two lists of SQABase instances.

        Assumptions:
            -- The contents of each list are types that implement `equals`
            -- The lists do not contain duplicates

        Checking equality is then the same as checking that the lists are the same
        length, and that one is a subset of the other.
        """
        if len(l1) != len(l2):
            return False  # pragma: no cover
        for x in l1:
            for y in l2:
                # pyre-fixme[20]: Argument `o` expected.
                if type(x) != type(y):
                    equal = False
                if isinstance(x, SQABase):
                    # pyre-fixme[6]: Expected `SQABase` for 1st param but got `T`.
                    equal = x.equals(y)
                elif isinstance(x, (int, float, str, bool, Enum)):
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
        """Merge a new list (`l2`) into an existing list (`l1`).

        Assumptions:
            -- The lists do not contain duplicates

        For each item in the new list:
        -- Check if there is a matching item in the old list.
        -- If so, update and retain the existing item.
        -- If not, add the item from the new list.
        -- If there are any items in the old list that don't have a matching
           item in the new list, remove them.

        Example:
        1) You have an experiment with metrics [m1, m2]
        2) You update the experiment to remove m1, change the properties of m2,
           and add add m3. The new list of metrics is [m2(new), m3]
        3) This will result in calling `list_update([m1, m2], [m2(new), m3])`
        4) We start by looking at m2(new), and we find a matching object (m2)
           in the original list, because the unique_id (name) of both objects
           is the same. So we call m2.update(m2(new)) to update m2 in place,
           and add it to our new list.
        5) Now we look at m3. We don't find a matching object in the original list,
           so we add m3 to our new list.
        6) At the end, our new list contains the original m2 object with updates
           applied, and the new m3 object.
        7) SQL Alchemy will perform an update on m2, an insert of m3, and because
           the new list does not contain m1, and because we have
           cascade='all, delete-orphan' specified on the relationship between
           xperiment and metrics, this child will be deleted.
        """
        new_list = []
        for y in l2:  # For each item in the new list
            for x in l1:  # Check if there is a matching item in the old list
                # pyre-fixme[20]: Argument `o` expected.
                if type(x) != type(y):
                    equal = False
                elif isinstance(x, SQABase):
                    # To determine if the item is matching, first check if
                    # the class has a unique id specified
                    unique_id = getattr(type(x), "unique_id", None)
                    if unique_id is None:
                        # If there is no unique id specified, we can't match
                        # up items and perform an update. Instead, we check if
                        # there is an exact match, and if there isn't, we
                        # throw away the old item and insert the new one.
                        # This will happen for ParameterConstraints
                        # pyre-fixme[6]: Expected `SQABase` for 1st param but got `T`.
                        equal = x.equals(y)
                    else:
                        x_unique_value = getattr(x, unique_id)
                        y_unique_value = getattr(y, unique_id)
                        if x_unique_value is None or y_unique_value is None:
                            # e.g. GeneratorRun's unique id is index, which is nullable.
                            # Of a GeneratorRun does not have an index set,
                            # we'll have to check if there is an exact match,
                            # and if not, we throw away the old item and insert
                            # the new one. This will happen for GeneratorRuns
                            # wrapped around Trial status quos.
                            # pyre-fixme[6]: Expected `SQABase` for 1st param but got...
                            equal = x.equals(y)
                        else:
                            equal = x_unique_value == y_unique_value
                            # If we've determined that two items are "matching",
                            # i.e. have the same unique value but possibly differ
                            # in other ways, we can recursively update.
                            if equal:
                                # pyre-fixme[6]: Expected `SQABase` for 1st param but...
                                x.update(y)
                elif isinstance(x, (int, float, str, bool, dict, Enum)):
                    # Right now, the only times we perform an update on a list
                    # that isn't a list of SQABase children is:
                    # -- when we update the list of values of a ChoiceParameter.
                    # -- when we update GeneratorRun model predictions
                    equal = x == y
                else:
                    raise ValueError(
                        f"Calling list_update on unsupported types: "
                        f"{type(x)} and {type(y)}"
                    )
                if equal:
                    # If we found a matching item (and potentially updated it)
                    # we add it to our new list and stop looking for a match.
                    # pyre-fixme[6]: Expected `T` for 1st param but got `Union[Dict[A...
                    new_list.append(x)
                    break
            else:
                # If we were unable to find a matching item, either because this
                # item is new, or because this class doesn't have a unique id
                # and an old item was changed, then we ignore the old item
                # and add the new one.
                new_list.append(y)
        return new_list

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
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if type(self_val) != type(other_val):
                return False
            if isinstance(self_val, list):
                equal = SQABase.list_equals(self_val, other_val)
            elif isinstance(self_val, SQABase):
                equal = self_val.equals(other_val)
            elif isinstance(self_val, datetime):
                equal = datetime_equals(self_val, other_val)
            else:
                equal = self_val == other_val
            if not equal:
                return False
        return True

    def update(self, other: "SQABase") -> None:
        """Merge `other` into `self.`"""
        immutable_fields = set(
            getattr(self, "immutable_fields", []) + ["id", "_sa_instance_state"]
        )
        self.validate_update(other)
        for field in self.attributes:
            if (
                field in immutable_fields
                # We don't want to update foreign key fields, e.g. experiment_id.
                # The new object will always have a value of None for this field,
                # but we don't want to overwrite the value on the existing object.
                or is_foreign_key_field(field)
            ):
                continue
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if isinstance(self_val, list) and isinstance(other_val, list):
                other_val = SQABase.list_update(self_val, other_val)
            elif isinstance(self_val, SQABase) and isinstance(other_val, SQABase):
                self_val.update(other_val)
                other_val = self_val
            setattr(self, field, other_val)

    def validate_update(self, other: "SQABase") -> None:
        """Validate that `self` can be updated to `other`."""
        immutable_fields = getattr(self, "immutable_fields", [])
        for field in immutable_fields:
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if type(self_val) != type(other_val):
                equal = False
            if isinstance(self_val, list):
                equal = SQABase.list_equals(self_val, other_val)
            elif isinstance(self_val, SQABase):
                equal = self_val.equals(  # pragma: no cover (no example of this yet)
                    other_val
                )
            elif isinstance(self_val, datetime):
                equal = datetime_equals(self_val, other_val)
            else:
                equal = self_val == other_val
            if not equal:
                raise ImmutabilityError(
                    f"Cannot change `{field}` of {self.__class__.__name__}."
                )


Base = declarative_base(metaclass=DeclarativeABCMeta, cls=SQABase)


def create_mysql_engine_from_creator(
    creator: Callable, echo: bool = False, pool_recycle: int = 10, **kwargs: Any
) -> Engine:
    """Create a SQLAlchemy engine with the MySQL dialect given a creator function.

    Args:
        creator (Callable):  a callable which returns a DBAPI connection.
        echo (bool, optional): if True, set engine to be verbose.
        pool_recycle (int, optional): number of seconds after which to recycle
            connections. -1 means no timeout. Default is 10 seconds.
        **kwargs: keyword args passed to `create_engine`

    Returns:
        Engine: SQLAlchemy engine with connection to MySQL DB.

    """
    return create_engine(
        "mysql://", creator=creator, pool_recycle=pool_recycle, echo=echo, **kwargs
    )


def create_test_engine(path: Optional[str] = None, echo: bool = True) -> Engine:
    """Creates a SQLAlchemy engine object for use in unit tests.

    Args:
        path (Optional[str], optional): if None, use in-memory SQLite; else
            attempt to create a SQLite DB in the path provided.
        echo (bool, optional): if True, set engine to be verbose.

    Returns:
        Engine: an instance of SQLAlchemy engine.

    """
    if path is None:
        db_path = "sqlite://"
    else:
        db_path = "sqlite:///{path}".format(path=path)
    return create_engine(db_path, echo=echo)


def init_engine_and_session_factory(
    test: bool = False,
    tier_or_path: Optional[str] = None,
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
        test (bool, optional): if True, use in-memory SQLite.
        tier_or_path (Optional[str], optional): the name of the DB tier for
            writing to a SQL database.
        creator (Optional[Callable]):  a callable which returns a DBAPI connection.
        echo (bool, optional): if True, logging for engine is enabled.
        force_init (bool, optional): if True, allows re-initializing engine
            and session factory.
        **kwargs: keyword arguments passed to `create_mysql_engine_from_creator`

    """
    global SESSION_FACTORY

    if SESSION_FACTORY is not None:
        if force_init:
            SESSION_FACTORY.bind.dispose()
        else:
            return
    if test:
        engine = create_test_engine(path=tier_or_path, echo=echo)
        create_all_tables(engine)
    elif creator is not None:
        engine = create_mysql_engine_from_creator(creator=creator, echo=echo, **kwargs)
    else:
        raise ValueError(
            "Must specify either test mode or a creator function."
        )  # pragma: no cover
    SESSION_FACTORY = scoped_session(sessionmaker(bind=engine))


def create_all_tables(engine: Engine) -> None:
    """Create all tables that inherit from Base.

    Args:
        engine (Engine): a SQLAlchemy engine with a connection to a MySQL
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
    return SESSION_FACTORY()


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
