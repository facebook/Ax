#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, NamedTuple, Optional

from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder


class DBSettings(NamedTuple):
    """
    Defines behavior for loading/saving experiment to/from db.
    """

    decoder: Decoder
    encoder: Encoder
    creator: Optional[Callable] = None
