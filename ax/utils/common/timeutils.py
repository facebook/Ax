#!/usr/bin/env python3

from datetime import datetime

import pandas as pd


def _ts_to_pandas(ts: int) -> pd.Timestamp:  # pyre-ignore[11]
    """Convert int timestamp into pandas timestamp."""
    return pd.Timestamp(datetime.fromtimestamp(ts))


def _pandas_ts_to_int(ts: pd.Timestamp) -> int:  # pyre-ignore[11]
    """Convert int timestamp into pandas timestamp."""
    return ts.to_pydatetime().timestamp()
