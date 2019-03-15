#!/usr/bin/env python3

from typing import Dict, Optional, Type

from ae.lazarus.ae.runners.synthetic import SyntheticRunner


class RunnerRegistry:
    """Class that contains dictionaries mapping runner classes to ints.

    All runners will be stored in the same table in the database. When
    saving, we look up the runner subclass in `type_to_class`, and store
    the corresponding type field in the database. When loading, we look
    up the type field in `class_to_type`, and initialize the corresponding
    runner subclass.
    """

    def __init__(self, class_to_type: Optional[Dict[Type, int]] = None):
        self.class_to_type = class_to_type or {SyntheticRunner: 0}
        self.type_to_class = {v: k for k, v in self.class_to_type.items()}
