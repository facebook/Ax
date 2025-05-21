# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable
from logging import Logger
from typing import Any, Sequence, Union

import pandas as pd
from ax.adapter import Adapter
from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.logger import get_logger
from pyre_extensions import override

logger: Logger = get_logger(__name__)


class MetricFetchingErrorsAnalysis(HealthcheckAnalysis):
    """
    Analysis to check if any metric fetch errors occurred.
    If any metric fetch errors have occurred, the analysis will display the
    trial index, metric name, timestamp, error message, and traceback.
    """

    def __init__(
        self,
        max_records: int = 10,
        add_traceback_paste_callable: Callable[
            [str, list[dict[str, Any]]], list[dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        r"""
        Args:
            max_records: Max number of records to display.
            add_traceback_paste_callable: A function that takes an experiment name and a
            list of metric fetch errors, and returns the modified list with a paste
            link added to each error's traceback if available. If no function is
            provided, no traceback information will be included in the health check.
        """
        self.max_records = max_records
        self.add_traceback_paste_callable = add_traceback_paste_callable

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[HealthcheckAnalysisCard]:
        if experiment is None:
            raise UserInputError("MetricFetchingErrorsAnalysis requires an Experiment.")

        df = pd.DataFrame()

        metric_fetch_errors = experiment._metric_fetching_errors

        if not metric_fetch_errors:
            return [
                self._create_healthcheck_analysis_card(
                    title="Metric Fetch Errors",
                    subtitle="No metric fetch errors found.",
                    level=AnalysisCardLevel.DEBUG + 1,
                    df=df,
                    status=HealthcheckStatus.PASS,
                    category=AnalysisCardCategory.DIAGNOSTIC,
                )
            ]

        metric_fetch_errors = sorted(
            metric_fetch_errors, key=lambda e: e["timestamp"], reverse=True
        )
        if self.max_records is not None:
            metric_fetch_errors = metric_fetch_errors[: self.max_records]

        metric_fetch_errors = [
            e for e in metric_fetch_errors if self._validate_fields(errors=e)
        ]

        if self.add_traceback_paste_callable is not None:
            metric_fetch_errors = self.add_traceback_paste_callable(
                experiment.name, metric_fetch_errors
            )
        else:
            for e in metric_fetch_errors:
                e["paste"] = "No traceback available"

        metric_fetch_errors_for_card = [
            {
                "trial_index": e["trial_index"],
                "metric_name": e["metric_name"],
                "timestamp": e["timestamp"],
                "reason": e["reason"],
                "traceback": e["paste"],
            }
            for e in metric_fetch_errors
        ]

        df = pd.DataFrame(
            columns=[
                "trial_index",
                "metric_name",
                "timestamp",
                "reason",
                "traceback",
            ]
        )
        df = pd.concat([df, pd.DataFrame.from_records(metric_fetch_errors_for_card)])

        subtitle_df_columns = {
            "trial_index": "Trial Index",
            "metric_name": "Metric Name",
            "timestamp": "Timestamp",
            "reason": "Error Message",
            "traceback": "Traceback",
        }
        subtitle = df.rename(columns=subtitle_df_columns)

        return [
            self._create_healthcheck_analysis_card(
                title="Metric Fetch Errors",
                subtitle=subtitle.to_markdown(index=False),
                level=AnalysisCardLevel.DEBUG + 1,
                df=df,
                status=HealthcheckStatus.WARNING,
                category=AnalysisCardCategory.DIAGNOSTIC,
            )
        ]

    @staticmethod
    def _validate_fields(errors: dict[str, Union[int, str]]) -> bool:
        required_fields = {
            "trial_index",
            "metric_name",
            "reason",
            "timestamp",
            "traceback",
        }

        if not required_fields.issubset(errors.keys()):
            missing_fields = required_fields - errors.keys()
            logger.warning(
                f"Metric fetch error {errors} is missing required fields "
                f"{missing_fields}. Discarding."
            )
            return False

        # maybe it was already modified to have a paste since dicts are
        # passed by reference
        optional_fields = {"paste"}
        if not set(errors.keys()).issubset(required_fields.union(optional_fields)):
            extra_fields = set(errors.keys()) - required_fields.union(optional_fields)
            logger.warning(
                f"Metric fetch error {errors} has extra fields "
                f"{extra_fields}. Discarding."
            )
            return False

        return True
