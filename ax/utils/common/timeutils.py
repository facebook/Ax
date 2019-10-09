#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from datetime import datetime
from time import time

import pandas as pd


def _ts_to_pandas(ts: int) -> pd.Timestamp:  # pyre-ignore[11]
    """Convert int timestamp into pandas timestamp."""
    return pd.Timestamp(datetime.fromtimestamp(ts))


def _pandas_ts_to_int(ts: pd.Timestamp) -> int:  # pyre-ignore[11]
    """Convert int timestamp into pandas timestamp."""
    return ts.to_pydatetime().timestamp()


def current_timestamp_in_millis() -> int:
    """Grab current timestamp in milliseconds as an int."""
    return int(round(time() * 1000))
