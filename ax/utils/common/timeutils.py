#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta
from time import time
from typing import Generator

import pandas as pd


DS_FRMT = "%Y-%m-%d"  # Format to use for parsing DS strings.


def to_ds(ts: datetime) -> str:
    """Convert a `datetime` to a DS string."""
    return datetime.strftime(ts, DS_FRMT)


def to_ts(ds: str) -> datetime:
    """Convert a DS string to a `datetime`."""
    return datetime.strptime(ds, DS_FRMT)


def _ts_to_pandas(ts: int) -> pd.Timestamp:
    """Convert int timestamp into pandas timestamp."""
    # pyre-fixme[6]: Expected `Optional[str]` for 1st param but got `datetime`.
    return pd.Timestamp(datetime.fromtimestamp(ts))


def _pandas_ts_to_int(ts: pd.Timestamp) -> int:
    """Convert int timestamp into pandas timestamp."""
    # pyre-fixme[16]: `Timestamp` has no attribute `to_pydatetime`.
    return ts.to_pydatetime().timestamp()


def current_timestamp_in_millis() -> int:
    """Grab current timestamp in milliseconds as an int."""
    return int(round(time() * 1000))


def timestamps_in_range(
    start: datetime, end: datetime, delta: timedelta
) -> Generator[datetime, None, None]:
    """Generator of timestamps in range [start, end], at intervals
    delta.
    """
    curr = start
    while curr <= end:
        yield curr
        curr += delta
