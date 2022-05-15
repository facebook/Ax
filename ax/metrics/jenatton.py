# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric
from ax.utils.common.typeutils import not_none


class JenattonMetric(Metric):
    def __init__(
        self,
        name: str = "jenatton",
    ) -> None:
        super().__init__(name=name)

    @staticmethod
    def _f(
        x1: Optional[int] = None,
        x2: Optional[int] = None,
        x3: Optional[int] = None,
        x4: Optional[float] = None,
        x5: Optional[float] = None,
        x6: Optional[float] = None,
        x7: Optional[float] = None,
        r8: Optional[float] = None,
        r9: Optional[float] = None,
    ) -> float:
        if x1 == 0:
            if x2 == 0:
                return not_none(x4) ** 2 + 0.1 + not_none(r8)
            else:
                return not_none(x5) ** 2 + 0.2 + not_none(r8)
        else:
            if x3 == 0:
                return not_none(x6) ** 2 + 0.3 + not_none(r9)
            else:
                return not_none(x7) ** 2 + 0.4 + not_none(r9)

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> Data:
        # pyre-ignore [6]
        mean = [self._f(**arm.parameters) for _, arm in trial.arms_by_name.items()]
        df = pd.DataFrame(
            {
                "arm_name": [name for name, _ in trial.arms_by_name.items()],
                "metric_name": self.name,
                "mean": mean,
                "sem": 0,
                "trial_index": trial.index,
            }
        )

        return Data(df=df)
