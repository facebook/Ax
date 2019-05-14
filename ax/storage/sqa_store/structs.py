#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, NamedTuple, Optional

from ax.storage.sqa_store.sqa_config import SQAConfig


class DBSettings(NamedTuple):
    """
    Defines behavior for loading/saving experiment to/from db.
    """

    config: Optional[SQAConfig] = SQAConfig()
    creator: Optional[Callable] = None
    url: Optional[str] = None
