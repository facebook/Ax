# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-safe

from typing import Sequence

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.utils import get_target_trial_index
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.utils.common.constants import Keys
from pyre_extensions import none_throws


def extract_relevant_adapter(
    experiment: Experiment | None,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
) -> Adapter:
    """
    Analysis.compute(...) takes in both a GenerationStrategy and an Adapter as optional
    arguments. To keep consistency we always want use the same logic for extracting the
    relevant adapter across all Analysis classes.

    * If an Adapter is provided, return it.
    * If a GenerationStrategy is provided, return the Adapter from the current
        GenerationNode. Additionally, if it has not been fit, fit it using the
        Experiment.
    * If both are provided prefer the Adapter.
    * If neither are provided, raise an Exception.
    """

    if generation_strategy is None and adapter is None:
        raise UserInputError(
            "Must provide either a GenerationStrategy or an Adapter to compute this "
            "Analysis."
        )

    if adapter is not None:
        return adapter

    generation_strategy = none_throws(generation_strategy)

    if (model := generation_strategy.model) is not None:
        return model

    if experiment is None:
        raise UserInputError(
            "Provided GenerationStrategy has no model, but no Experiment was provided "
            "to source data to fit the model."
        )

    generation_strategy.current_node._fit(experiment=experiment)

    return none_throws(generation_strategy.model)


def prepare_arm_data(
    experiment: Experiment,
    metric_names: Sequence[str],
    use_model_predictions: bool,
    adapter: Adapter | None = None,
    trial_index: int | None = None,
    additional_arms: Sequence[Arm] | None = None,
) -> pd.DataFrame:
    """
    Create a table with one entry per arm and columns for each requested metric. This
    is useful for all analyses where the atomic unit is an Arm (ex. ArmEffectsPlot,
    ScatterPlot, ParallelCoordinatesPlot).

    Args:
        experiment: The experiment to extract data from.
        metric_names: The names of the metrics to include in the table.
        use_model_predictions: Whether to use an Adapter to predict the effects of each
            arm. This is often more trustworthy (and leads to better reproducibility)
            than using the raw data, especially when model fit is good and in high-
            noise settings.
        adapter: The adapter to use to predict the effects of each arm if using model
            predictions.
        trial_index: If present, only use arms from the trial with the given index.
            Otherwise include all arms on the experiment. If trial_index=-1, do not
            include any arms from the experiment.
        additional_arms: If present, include these arms in the table. These arms will
            be marked as belonging to a trial with index -1.

    Returns a DataFrame with the following columns:
        - trial_index
        - arm_name
        - generation_node
        - METRIC_NAME_mean for each metric_name in metric_names
        - METRIC_NAME_sem for each metric_name in metric_names
    """

    # Ensure a valid combination of arguments is provided.
    if len(metric_names) < 1:
        raise UserInputError("Must provide at least one metric name.")

    missing_metrics = set(metric_names) - set(experiment.metrics.keys())
    if missing_metrics:
        raise UserInputError(
            f"Requested metrics {missing_metrics} are not present in the experiment."
        )

    if (
        trial_index is not None
        # -1 is allowed and signifies that no arms from the experiment should be
        # included.
        and trial_index != -1
        and trial_index not in experiment.trials.keys()
    ):
        raise UserInputError(f"Trial with index {trial_index} not found in experiment.")

    if use_model_predictions:
        if adapter is None:
            raise UserInputError(
                "Must provide an adapter to use model predictions for the analysis."
            )

        df = _prepare_modeled_arm_data(
            experiment=experiment,
            metric_names=metric_names,
            adapter=adapter,
            trial_index=trial_index,
            additional_arms=additional_arms,
        )
    else:
        if additional_arms is not None:
            raise UserInputError(
                "Cannot provide additional arms when use_model_predictions=False since"
                "there is no observed raw data for the additional arms that are not "
                "part of the Experiment."
            )

        df = _prepare_raw_arm_data(
            metric_names=metric_names,
            experiment=experiment,
            trial_index=trial_index,
        )

    # Add additional columns which do not require predicting or extracting data.
    # TODO[mpolson64]: Add a column for the arm feasibility.
    df["trial_status"] = df["trial_index"].apply(
        lambda trial_index: experiment.trials[trial_index].status.name
        if trial_index != -1
        else "Additional Arm"
    )
    df["generation_node"] = df.apply(
        lambda row: _extract_generation_node_name(
            trial=experiment.trials[row["trial_index"]],
            arm=experiment.arms_by_name[row["arm_name"]],
        )
        if row["trial_index"] in experiment.trials.keys()
        else Keys.UNKNOWN_GENERATION_NODE.value,
        axis=1,
    )

    return df


def _prepare_modeled_arm_data(
    experiment: Experiment,
    metric_names: Sequence[str],
    adapter: Adapter,
    trial_index: int | None = None,
    additional_arms: Sequence[Arm] | None = None,
) -> pd.DataFrame:
    """
    Compute the modeled (mean, sem) for each arm for each requested metric.

    Returns a DataFrame with the following columns:
        - trial_index
        - arm_name
        - METRIC_NAME_mean for each metric_name in metric_names
        - METRIC_NAME_sem for each metric_name in metric_names
    """
    # Extract the information necessary to construct each row of the DataFrame.
    trial_index_arm_pairs = [
        # Arms from the experiment (empty if trial_index=-1)
        *[
            (
                trial.index,
                arm,
            )
            for trial in experiment.trials.values()
            if (trial.index == trial_index) or (trial_index is None)
            for arm in trial.arms
        ],
        # Additional arms passed in by the user
        *[(-1, arm) for arm in additional_arms or []],
    ]

    # Batch predict for efficiency.
    predictions = adapter.predict(
        observation_features=[
            ObservationFeatures.from_arm(
                arm=arm,
                # Always predict as if the arm is a member of the target trial.
                trial_index=get_target_trial_index(experiment=experiment),
            )
            for _, arm in trial_index_arm_pairs
        ]
    )

    records = [
        {
            "trial_index": trial_index_arm_pairs[i][0],
            "arm_name": trial_index_arm_pairs[i][1].name
            if trial_index_arm_pairs[i][1].has_name
            else f"{Keys.UNNAMED_ARM.value}_{i}",
            **{
                f"{metric_name}_mean": predictions[0][metric_name][i]
                for metric_name in metric_names
            },
            **{
                f"{metric_name}_sem": predictions[1][metric_name][metric_name][i] ** 0.5
                for metric_name in metric_names
            },
        }
        for i in range(len(trial_index_arm_pairs))
    ]

    return pd.DataFrame.from_records(records)


def _prepare_raw_arm_data(
    metric_names: Sequence[str],
    experiment: Experiment,
    trial_index: int | None,
) -> pd.DataFrame:
    """
    Extract the raw (mean, sem) for each arm for each requested metric.

    Returns a DataFrame with the following columns:
        - trial_index
        - arm_name
        - METRIC_NAME_mean for each metric_name in metric_names
        - METRIC_NAME_sem for each metric_name in metric_names
    """
    # If trial_index is -1 do not extract any data.
    if trial_index == -1:
        return pd.DataFrame()
    # If trial_index is None, extract data from all trials.
    elif trial_index is not None:
        trials = [experiment.trials[trial_index]]
        data_df = experiment.lookup_data(trial_indices=[trial_index]).df
    else:
        trials = [*experiment.trials.values()]
        data_df = experiment.lookup_data().df

    records = []
    for trial in trials:
        for arm in trial.arms:
            # Extract (mean, sem) pairs for each metric when available, otherwise set to
            # None.
            means = {}
            sems = {}
            for metric_name in metric_names:
                mask = (
                    (data_df["trial_index"] == trial.index)
                    & (data_df["arm_name"] == arm.name)
                    & (data_df["metric_name"] == metric_name)
                )

                if not mask.any():
                    means[metric_name] = None
                    sems[metric_name] = None
                else:
                    means[metric_name] = data_df.loc[mask, "mean"].iloc[0]
                    sems[metric_name] = data_df.loc[mask, "sem"].iloc[0]

            records.append(
                {
                    "trial_index": trial.index,
                    "arm_name": arm.name,
                    **{
                        f"{metric_name}_mean": means[metric_name]
                        for metric_name in metric_names
                    },
                    **{
                        f"{metric_name}_sem": sems[metric_name]
                        for metric_name in metric_names
                    },
                }
            )

    return pd.DataFrame.from_records(records)


def _extract_generation_node_name(trial: BaseTrial, arm: Arm) -> str:
    """
    Extract the name of the GenerationNode that generated the given arm, assuming the
    arm was generated as part of the given trial.

    Args:
        trial: The trial to extract the GenerationNode from.
        arm: The arm to extract the GenerationNode from.

    Returns:
        The name of the GenerationNode that generated the given arm.
    """

    for gr in trial.generator_runs:
        if arm.signature in gr.arm_signatures:
            return gr._generation_node_name or Keys.UNKNOWN_GENERATION_NODE.value

    return Keys.UNKNOWN_GENERATION_NODE.value
