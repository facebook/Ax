# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import final, Iterable, Sequence

from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.utils import validate_experiment
from ax.core.analysis_card import AnalysisCard
from ax.core.experiment import Experiment
from ax.core.trial_status import NON_STALE_STATUSES, TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override


@final
class Summary(Analysis):
    """
    High-level summary of the Experiment with one row per arm. Any values missing at
    compute time will be represented as None. Columns where every value is None will
    be omitted by default.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - trial_status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
        - failure_reason: The reason for the failure, if applicable
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METADATA: Any metadata associated with the trial, as specified by the
            Experiment's runner.run_metadata_report_keys field
        - **METRIC_NAME: The observed mean of the metric specified, for each metric
        - **PARAMETER_NAME: The parameter value for the arm, for each parameter
     Args:
        trial_indices: If specified, only include these trial indices.
        trial_status: If specified, only include trials with this status.
        omit_empty_columns: If True, omit columns where every value is None.
    """

    def __init__(
        self,
        trial_indices: Iterable[int] | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        omit_empty_columns: bool = True,
    ) -> None:
        self.trial_indices = trial_indices
        self.trial_statuses: Sequence[TrialStatus] = (
            trial_statuses if trial_statuses is not None else list(NON_STALE_STATUSES)
        )
        self.omit_empty_columns = omit_empty_columns

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=False,
            require_data=True,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        if experiment is None:
            raise UserInputError("`Summary` analysis requires an `Experiment` input")

        # Determine if we should relativize based on:
        # (1) experiment has metrics and (2) experiment has status quo
        # (3) experiment data is not MapData (MapData doesn't support relativization
        # due to time-series step alignment complexities.)
        data = experiment.lookup_data(trial_indices=self.trial_indices)
        should_relativize = (
            len(experiment.metrics) > 0
            and experiment.status_quo is not None
            and not data.has_step_column
        )

        return self._create_analysis_card(
            title=(
                "Summary for "
                f"{experiment.name if experiment.has_name else 'Experiment'}"
            ),
            subtitle=(
                "High-level summary of the `Trial`-s in this `Experiment`"
                if not should_relativize
                else (
                    "High-level summary of the `Trial`-s in this `Experiment` "
                    "Metric results are relativized against status quo."
                )
            ),
            df=experiment.to_df(
                trial_indices=self.trial_indices,
                omit_empty_columns=self.omit_empty_columns,
                trial_statuses=self.trial_statuses,
                relativize=should_relativize,
            ),
        )
