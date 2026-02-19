# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from enum import IntEnum

import pandas as pd
from ax.core.analysis_card import AnalysisCard


class HealthcheckStatus(IntEnum):
    PASS = 0
    FAIL = 1
    WARNING = 2


class HealthcheckAnalysisCard(AnalysisCard):
    def get_status(self) -> HealthcheckStatus:
        return HealthcheckStatus(json.loads(self.blob)["status"])

    def is_passing(self) -> bool:
        return self.get_status() == HealthcheckStatus.PASS

    def get_aditional_attrs(self) -> dict[str, str | int | float | bool]:
        return json.loads(self.blob)


def create_healthcheck_analysis_card(
    name: str,
    title: str,
    subtitle: str,
    df: pd.DataFrame,
    status: HealthcheckStatus,
    **additional_attrs: str | int | float | bool,
) -> HealthcheckAnalysisCard:
    return HealthcheckAnalysisCard(
        name=name,
        title=title,
        subtitle=subtitle,
        df=df,
        blob=json.dumps(
            {
                "status": status,
                **additional_attrs,
            }
        ),
    )
