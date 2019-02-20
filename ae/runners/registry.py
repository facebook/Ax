#!/usr/bin/env python3

from typing import Dict, Type

from ae.lazarus.ae.runners.synthetic import SyntheticRunner


class RunnerRegistry:
    """Class that contains dictionaries mapping runner classes to ints.

    All runners will be stored in the same table in the database. When
    saving, we look up the runner subclass in TYPE_TO_CLASS, and store
    the corresponding type field in the database. When loading, we look
    up the type field in CLASS_TO_TYPE, and initialize the corresponding
    runner subclass.

    Create a subclass that inherits from RunnerRegistry if you want
    to add support for additional custom runner subclasses.
    """

    CLASS_TO_TYPE: Dict[Type, int] = {SyntheticRunner: 0}

    TYPE_TO_CLASS: Dict[int, Type] = {v: k for k, v in CLASS_TO_TYPE.items()}
