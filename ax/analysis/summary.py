# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.analysis import Analysis, AnalysisCard, AnalysisCardLevel
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from pyre_extensions import none_throws


class Summary(Analysis):
    """
    High-level summary of the Experiment with one row per arm. Any values missing at
    compute time will be represented as None. Columns where every value is None will
    be omitted by default.

    The DataFrame computed will contain one row per arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
        - failure_reason: The reason for the failure, if applicable
        - generation_method: The model_key of the model that generated the arm
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METADATA: Any metadata associated with the trial, as specified by the
            Experiment's runner.run_metadata_report_keys field
        - **METRIC_NAME: The observed mean of the metric specified, for each metric
        - **PARAMETER_NAME: The value of said parameter for the arm, for each parameter
    """

    def __init__(self, omit_empty_columns: bool = True) -> None:
        self.omit_empty_columns = omit_empty_columns

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> AnalysisCard:
        if experiment is None:
            raise UserInputError("`Summary` analysis requires an `Experiment` input")

        records = []
        data_df = experiment.lookup_data().df
        for index, trial in experiment.trials.items():
            for arm in trial.arms:
                # Find the observed means for each metric, placing None if not found
                observed_means = {}
                for metric in experiment.metrics.keys():
                    try:
                        observed_means[metric] = data_df[
                            (data_df["arm_name"] == arm.name)
                            & (data_df["metric_name"] == metric)
                        ]["mean"].item()
                    except ValueError:
                        observed_means[metric] = None

                # Find the arm's associated generation method from the trial via the
                # GeneratorRuns if possible
                grs = [gr for gr in trial.generator_runs if arm in gr.arms]
                generation_method = grs[0]._model_key if len(grs) > 0 else None
                generation_node = grs[0]._generation_node_name if len(grs) > 0 else None

                # Find other metadata from the trial to include from the trial based
                # on the experiment's runner
                metadata = (
                    {
                        key: value
                        for key, value in trial.run_metadata.items()
                        if key
                        in none_throws(experiment.runner).run_metadata_report_keys
                    }
                    if experiment.runner is not None
                    else {}
                )

                # Construct the record
                record = {
                    "trial_index": index,
                    "arm_name": arm.name,
                    "generation_method": generation_method,
                    "generation_node": generation_node,
                    "status": trial.status.name,
                    "fail_reason": trial.run_metadata.get("fail_reason", None),
                    **metadata,
                    **arm.parameters,
                    **observed_means,
                }

                records.append(record)

        df = pd.DataFrame(records)

        if self.omit_empty_columns:
            df = df.loc[:, df.notnull().all()]

        return self._create_analysis_card(
            title=f"Summary for {experiment.name}",
            subtitle="High-level summary of the `Trial`-s in this `Experiment`",
            level=AnalysisCardLevel.MID,
            df=df,
        )
