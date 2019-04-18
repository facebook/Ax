#!/usr/bin/env python3

from typing import Dict, Optional, Type

from ax.core.runner import Runner
from ax.runners.synthetic import SyntheticRunner


"""
Mapping of Runner classes to ints.

All runners will be stored in the same table in the database. When
saving, we look up the runner subclass in RUNNER_REGISTRY, and store
the corresponding type field in the database. When loading, we look
up the type field in REVERSE_RUNNER_REGISTRY, and initialize the
corresponding runner subclass.
"""
RUNNER_REGISTRY: Dict[Type[Runner], int] = {SyntheticRunner: 0}

REVERSE_RUNNER_REGISTRY: Dict[int, Type[Runner]] = {
    v: k for k, v in RUNNER_REGISTRY.items()
}


def register_runner(runner_cls: Type[Runner], val: Optional[int] = None) -> None:
    """Add a custom runner class to the registries.
    If no int is specified, use a hash of the class name.
    """
    registered_val = val or abs(hash(runner_cls.__name__)) % (10 ** 5)
    RUNNER_REGISTRY[runner_cls] = registered_val
    REVERSE_RUNNER_REGISTRY[registered_val] = runner_cls
