# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
from enum import IntEnum

import pandas as pd
from ax.analysis.analysis import ErrorAnalysisCard
from ax.core.analysis_card import AnalysisCard, AnalysisCardBase


class HealthcheckStatus(IntEnum):
    PASS = 0
    FAIL = 1
    WARNING = 2


# Healthchecks that provide valuable progress info even when passing
PRIORITY_HEALTHCHECKS: set[str] = {
    "BaselineImprovementAnalysis",
    "EarlyStoppingAnalysis",
}


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


# Status order for sorting: FAIL first, then WARNING, then PASS
_STATUS_SORT_ORDER: dict[HealthcheckStatus, int] = {
    HealthcheckStatus.FAIL: 1,
    HealthcheckStatus.WARNING: 2,
    HealthcheckStatus.PASS: 3,
}


def sort_healthcheck_cards(
    cards: list[AnalysisCardBase],
) -> list[AnalysisCardBase]:
    """
    Sort healthcheck cards by severity and priority.

    Order:
        1. ErrorAnalysisCard (errors during computation)
        2. FAIL status
        3. WARNING status
        4. PASS status with priority (BaselineImprovement, EarlyStopping, etc.)
        5. PASS status (rest)

    Args:
        cards: List of analysis cards (typically HealthcheckAnalysisCard or
            ErrorAnalysisCard instances).

    Returns:
        Sorted list of cards.
    """

    def sort_key(card: AnalysisCardBase) -> tuple[int, int, str]:
        if isinstance(card, ErrorAnalysisCard):
            return (0, 0, card.name)

        if isinstance(card, HealthcheckAnalysisCard):
            return (
                _STATUS_SORT_ORDER[card.get_status()],
                0 if card.name in PRIORITY_HEALTHCHECKS else 1,
                card.name,
            )

        # Fallback for type safety (unreachable in practice)
        return (4, 1, card.name)

    return sorted(cards, key=sort_key)
