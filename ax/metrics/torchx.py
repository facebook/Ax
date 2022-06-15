# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, cast

import pandas as pd
from ax.core import Trial
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger = get_logger(__name__)

try:
    from ax.runners.torchx import TORCHX_TRACKER_BASE
    from torchx.runtime.tracking import FsspecResultTracker

    class TorchXMetric(Metric):
        """
        Fetches AppMetric (the observation returned by the trial job/app) via the
        ``torchx.tracking`` module. Assumes that the app used the tracker in the
        following manner:

        .. code-block:: python

        tracker = torchx.runtime.tracking.FsspecResultTracker(tracker_base)
        tracker[str(trial_index)] = {metric_name: value}

        # -- or --
        tracker[str(trial_index)] = {"metric_name/mean": mean_value,
                                    "metric_name/sem": sem_value}

        """

        def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> Data:

            tracker_base = trial.run_metadata[TORCHX_TRACKER_BASE]
            tracker = FsspecResultTracker(tracker_base)
            res = tracker[trial.index]

            if self.name in res:
                mean = res[self.name]
                sem = None
            else:
                mean = res.get(f"{self.name}/mean")
                sem = res.get(f"{self.name}/sem")

            if mean is None and sem is None:
                raise KeyError(
                    f"Observation for `{self.name}` not found in tracker at base "
                    f"`{tracker_base}`. Ensure that the trial job is writing the "
                    "results at the same tracker base."
                )

            df_dict = {
                "arm_name": not_none(cast(Trial, trial).arm).name,
                "trial_index": trial.index,
                "metric_name": self.name,
                "mean": mean,
                "sem": sem,
            }
            return Data(df=pd.DataFrame.from_records([df_dict]))

except ImportError:
    logger.warning(
        "torchx package not found. If you would like to use TorchXMetric, please "
        "install torchx."
    )
    pass
