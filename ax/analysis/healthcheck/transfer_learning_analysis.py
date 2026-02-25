# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import final, TYPE_CHECKING

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override

if TYPE_CHECKING:
    from ax.storage.sqa_store.sqa_config import SQAConfig


@final
class TransferLearningAnalysis(Analysis):
    def __init__(
        self,
        experiment_types: list[str] | None = None,
        overlap_threshold: float = 0.25,
        max_num_exps: int = 10,
        config: SQAConfig | None = None,
    ) -> None:
        self.experiment_types = experiment_types
        self.overlap_threshold = overlap_threshold
        self.max_num_exps = max_num_exps
        self.config = config

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        if experiment is None:
            raise UserInputError(
                "TransferLearningAnalysis requires a non-null experiment to compute "
                "overlap percentages. Please provide an experiment."
            )

        # Determine experiment types to query for.
        experiment_types = self.experiment_types
        if experiment_types is None:
            if experiment.experiment_type is None:
                return create_healthcheck_analysis_card(
                    name=self.__class__.__name__,
                    title="Transfer Learning Eligibility",
                    subtitle=(
                        "No experiment type set on this experiment. "
                        "Cannot search for transferable experiments."
                    ),
                    df=pd.DataFrame(),
                    status=HealthcheckStatus.PASS,
                )
            experiment_types = [experiment.experiment_type]

        # Lazy import to avoid circular dependency (sqa_store depends on
        # healthcheck_analysis).
        from ax.storage.sqa_store.load import identify_transferable_experiments

        transferable_experiments = identify_transferable_experiments(
            search_space=experiment.search_space,
            experiment_types=experiment_types,
            overlap_threshold=self.overlap_threshold,
            max_num_exps=self.max_num_exps,
            config=self.config,
        )

        # Filter out the target experiment itself from results.
        transferable_experiments = {
            name: metadata
            for name, metadata in transferable_experiments.items()
            if name != experiment.name
        }

        if not transferable_experiments:
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Transfer Learning Eligibility",
                subtitle="No eligible source experiments found for transfer learning.",
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        total_parameters = len(experiment.search_space.parameters)

        rows = []
        for exp_name, metadata in transferable_experiments.items():
            overlap_count = len(metadata.overlap_parameters)
            overlap_pct = (
                (overlap_count / total_parameters * 100)
                if total_parameters > 0
                else 0.0
            )
            rows.append(
                {
                    "Experiment": exp_name,
                    "Overlapping Parameters": overlap_count,
                    "Overlap (%)": round(overlap_pct, 1),
                    "Parameters": ", ".join(sorted(metadata.overlap_parameters)),
                }
            )

        # Sort by overlapping parameter count descending
        rows.sort(key=lambda r: r["Overlapping Parameters"], reverse=True)

        df = pd.DataFrame(rows)

        n = len(rows)
        exp_lines = "\n".join(
            f"- **{r['Experiment']}** ({r['Overlap (%)']:.1f}% parameter overlap)"
            for r in rows
        )
        subtitle = (
            "Transfer learning can improve optimization by leveraging data "
            "from similar past experiments. We found "
            f"**{n} eligible source experiment(s)** "
            "for transfer learning:\n\n"
            f"{exp_lines}\n\n"
            "Caution: Only use source experiments that are closely related "
            "to your current experiment. "
            "Using data from unrelated experiments can lead to negative "
            "transfer, which may hurt "
            "optimization performance. Review the overlapping parameters "
            "before enabling transfer learning."
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Transfer Learning Eligibility",
            subtitle=subtitle,
            df=df,
            status=HealthcheckStatus.WARNING,
        )
