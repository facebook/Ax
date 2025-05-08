# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Sequence

import numpy as np

import pandas as pd
import torch
from ax.analysis.plotly.utils import is_predictive
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.core.types import ComparisonOp
from ax.core.utils import get_target_trial_index
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.registry import Generators
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import relativize
from botorch.utils.probability.utils import compute_log_prob_feas_from_bounds
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)

# Warn if p_feasible is less than this threshold.
# TODO: Move this constrant to best point utilities so that the logic for warning in
# analyses is also gates points from being chosen during best point selection.
POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD: float = 0.05


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
    trial_statuses: Sequence[TrialStatus] | None = None,
    additional_arms: Sequence[Arm] | None = None,
    relativize: bool = False,
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
        trial_statuses: If present, only include arms from trials with statuses in this
            collection. If not present, include all arms on the experiment.
        additional_arms: If present, include these arms in the table. These arms will
            be marked as belonging to a trial with index -1.
        relativize: Whether to relativize the effects of each arm against the status
            quo arm. If multiple status quo arms are present, relativize each arm
            against the status quo arm from the same trial.

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

    if trial_index is not None and trial_statuses is not None:
        raise UserInputError(
            "Cannot provide both trial_index and trial_statuses. Either provide a "
            "trial_index to filter on or a collection of trial_statuses to filter on."
        )

    # Compute the trial index of the target trial both to pass as a fixed feature
    # during prediction if using model predictions, and to relativize against the
    # status quo arm from the target trial if relativizing.
    target_trial_index = get_target_trial_index(experiment=experiment)

    if use_model_predictions:
        if adapter is None:
            raise UserInputError(
                "Must provide an adapter to use model predictions for the analysis."
            )

        if not is_predictive(adapter=adapter):
            logger.info(
                "Using Empirical Bayes to model effects because we were unable to find "
                " a suitable adapter."
            )
            data = (
                experiment.lookup_data(trial_indices=[trial_index])
                if trial_index is not None
                else experiment.lookup_data()
            )
            adapter = Generators.EMPIRICAL_BAYES_THOMPSON(
                experiment=experiment, data=data
            )

        df = _prepare_modeled_arm_data(
            experiment=experiment,
            metric_names=metric_names,
            adapter=adapter,
            trial_index=trial_index,
            trial_statuses=trial_statuses,
            additional_arms=additional_arms,
            target_trial_index=target_trial_index,
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
            trial_statuses=trial_statuses,
        )

    if relativize:
        if experiment.status_quo is None:
            raise UserInputError(
                "Cannot relativize data without a status quo arm on the experiment."
            )

        if experiment.status_quo.name is None:
            raise UserInputError(
                "Cannot relativize data without a named status quo arm on the "
                "experiment."
            )

        status_quo_name = experiment.status_quo.name

        if use_model_predictions:
            # Use the status quo arm from the target trial, prefer model predictions
            # over raw observations.
            mask = (df["arm_name"] == status_quo_name) & (
                df["trial_index"] == target_trial_index
            )

            # Use the raw status quo effects if no model predictions are available
            # (ex. if the status quo had None in its parameters).
            if not mask.any() or df[mask].isna().any().any():
                raw_df = _prepare_raw_arm_data(
                    metric_names=metric_names,
                    experiment=experiment,
                    trial_index=trial_index,
                    trial_statuses=trial_statuses,
                )

                # Fallback on the raw observations from the status quo arm on the
                # target trial.
                raw_mask = (raw_df["arm_name"] == status_quo_name) & (
                    raw_df["trial_index"] == target_trial_index
                )

                if not raw_mask.any():
                    raise UserInputError(
                        "Could not find a status quo arm on Trial "
                        f"{target_trial_index} to relativize against."
                    )

                status_quo_row = raw_df[raw_mask]
            else:
                status_quo_row = df[mask]

            status_quo_df = pd.concat(
                [status_quo_row] * len(df["trial_index"].unique())
            )
            status_quo_df["trial_index"] = df["trial_index"].unique()

        else:
            # If not using model predictions search for an appropriate status quo arm
            # using the following logic:
            # 1. Use the status quo arm from the same trial as the arm being
            #    relativized.
            # 2. Use the only status quo arm on the experiment.
            # 3. Raise an exception
            status_quo_mask = df["arm_name"] == status_quo_name
            status_quo_rows = []
            for trial_index in df["trial_index"].unique():
                trial_mask = (df["trial_index"] == trial_index) & status_quo_mask
                if trial_mask.any():
                    row = df[trial_mask].iloc[0]
                elif status_quo_mask.sum() == 1:
                    row = df[status_quo_mask].iloc[0]
                else:
                    raise UserInputError(
                        "Failed to relativize, no status quo arm found in the "
                        "experiment."
                    )

                row["trial_index"] = trial_index
                status_quo_rows.append(row)

            status_quo_df = pd.DataFrame(status_quo_rows)[
                [
                    "trial_index",
                    *[f"{name}_mean" for name in metric_names],
                    *[f"{name}_sem" for name in metric_names],
                ]
            ]

        df = _relativize_data(
            df=df,
            status_quo_df=status_quo_df,
        )

    # Add additional columns which do not require predicting or extracting data.
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

    df["p_feasible"] = _prepare_p_feasible(
        df=df,
        outcome_constraints=experiment.optimization_config.outcome_constraints
        if experiment.optimization_config is not None
        else [],
    )

    return df


def _prepare_modeled_arm_data(
    experiment: Experiment,
    metric_names: Sequence[str],
    adapter: Adapter,
    trial_index: int | None = None,
    trial_statuses: Sequence[TrialStatus] | None = None,
    additional_arms: Sequence[Arm] | None = None,
    target_trial_index: int | None = None,
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
            if ((trial.index == trial_index) or (trial_index is None))
            and ((trial_statuses is None) or (trial.status in trial_statuses))
            for arm in trial.arms
        ],
        # Additional arms passed in by the user
        *[(-1, arm) for arm in additional_arms or []],
    ]

    # Remove arms with missing parameters since we cannot predict for them.
    predictable_pairs = [
        (trial_index, arm)
        for trial_index, arm in trial_index_arm_pairs
        # Remove arms outside of the search space since we cannot predict for them.
        if experiment.search_space.check_membership(
            parameterization=arm.parameters,
            raise_error=False,
            check_all_parameters_present=True,
        )
    ]
    unpredictable_pairs = [
        (trial_index, arm)
        for trial_index, arm in trial_index_arm_pairs
        if not experiment.search_space.check_membership(
            parameterization=arm.parameters,
            raise_error=False,
            check_all_parameters_present=True,
        )
    ]

    # Batch predict for efficiency.
    predictions = adapter.predict(
        observation_features=[
            ObservationFeatures.from_arm(
                arm=arm,
                # Always predict as if the arm was run in the target trial.
                trial_index=target_trial_index,
            )
            for _, arm in predictable_pairs
        ]
    )

    records = [
        *[
            {
                "trial_index": predictable_pairs[i][0],
                "arm_name": predictable_pairs[i][1].name
                if predictable_pairs[i][1].has_name
                else f"{Keys.UNNAMED_ARM.value}_{i}",
                **{
                    f"{metric_name}_mean": predictions[0][metric_name][i]
                    for metric_name in metric_names
                },
                **{
                    f"{metric_name}_sem": predictions[1][metric_name][metric_name][i]
                    ** 0.5
                    for metric_name in metric_names
                },
            }
            for i in range(len(predictable_pairs))
        ],
        *[
            {
                "trial_index": unpredictable_pairs[i][0],
                "arm_name": unpredictable_pairs[i][1].name
                if unpredictable_pairs[i][1].has_name
                else f"{Keys.UNNAMED_ARM.value}_{i}",
                **{f"{metric_name}_mean": None for metric_name in metric_names},
                **{f"{metric_name}_sem": None for metric_name in metric_names},
            }
            for i in range(len(unpredictable_pairs))
        ],
    ]

    return pd.DataFrame.from_records(records)


def _prepare_raw_arm_data(
    metric_names: Sequence[str],
    experiment: Experiment,
    trial_index: int | None,
    trial_statuses: Sequence[TrialStatus] | None,
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
    else:
        trials = [
            trial
            for trial in experiment.trials.values()
            if (trial.index == trial_index or trial_index is None)
            and ((trial_statuses is None) or (trial.status in trial_statuses))
        ]
        data_df = experiment.lookup_data(
            trial_indices=[trial.index for trial in trials]
        ).df

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


def _prepare_p_feasible(
    df: pd.DataFrame,
    outcome_constraints: Sequence[OutcomeConstraint],
) -> pd.Series:
    """
    Compute the probability that each arm is feasible with respect to the given
    outcome constraints (assuming normally distributed observations). Calculated in a
    batch for efficiency.

    Ensure that the df and outcome constraints are either both relative or both
    absolute.

    Args:
        df: Result of _prepare_modeled_arm_data or _prepare_raw_arm_data.
        outcome_constraints: The outcome constraints to use to compute the probability
            of feasibility.

    Returns:
        A Series with one entry per row in df describing the probability that the arm
        is feasible with respect to the given outcome constraints.
    """
    if len(outcome_constraints) == 0:
        return pd.Series(np.ones(len(df)))

    # If an arm is missing data for a metric leave the mean as NaN.
    means = [
        df[f"{constraint.metric.name}_mean"].tolist()
        if f"{constraint.metric.name}_mean" in df.columns
        else np.nan * np.ones(len(df))
        for constraint in outcome_constraints
    ]

    # If an arm is missing data for a metric treat the sd as 0.
    sigmas = [
        (df[f"{constraint.metric.name}_sem"].fillna(0) ** 2).tolist()
        if f"{constraint.metric.name}_sem" in df.columns
        else [0] * len(df)
        for constraint in outcome_constraints
    ]

    con_lower_inds = [
        i
        for i in range(len(outcome_constraints))
        if outcome_constraints[i].op == ComparisonOp.GEQ
    ]
    con_upper_inds = [
        i
        for i in range(len(outcome_constraints))
        if outcome_constraints[i].op == ComparisonOp.LEQ
    ]

    con_lower = [
        constraint.bound
        for constraint in outcome_constraints
        if constraint.op == ComparisonOp.GEQ
    ]

    con_upper = [
        constraint.bound
        for constraint in outcome_constraints
        if constraint.op == ComparisonOp.LEQ
    ]

    log_prob_feas = compute_log_prob_feas_from_bounds(
        con_lower_inds=torch.tensor(con_lower_inds, dtype=torch.int),
        con_upper_inds=torch.tensor(con_upper_inds, dtype=torch.int),
        con_lower=torch.tensor(con_lower, dtype=torch.double),
        con_upper=torch.tensor(con_upper, dtype=torch.double),
        con_both_inds=torch.empty(0, dtype=torch.int),
        con_both=torch.empty(0),
        means=torch.tensor(means, dtype=torch.double).T,
        sigmas=torch.tensor(sigmas, dtype=torch.double).T,
    )

    return pd.Series(log_prob_feas.exp())


def _relativize_data(
    df: pd.DataFrame,
    status_quo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Relativize the data with respect to some status quo arm.

    Args:
        df: DataFrame containing the data to be relativized. Must include columns
            'METRIC_NAME_mean' and 'METRIC_NAME_sem' for each metric, as well as a
            column identifying the trial and a column identifying the status quo
            arm for each trial.
        status_quo_df: DataFrame containing the status quo data for each trial.

    Returns:
        A DataFrame with the same structure as the input, but with 'METRIC_NAME_mean'
        and 'METRIC_NAME_sem' columns relativized to the status quo arm for each metric
        within each trial.
    """
    metric_names = [name[:-5] for name in df.columns if name.endswith("_mean")]

    rel_df = df.copy()

    for trial_idx, trial_df in rel_df.groupby("trial_index"):
        status_quo_row = status_quo_df[status_quo_df["trial_index"] == trial_idx]

        for metric_name in metric_names:
            mean_col = f"{metric_name}_mean"
            sem_col = f"{metric_name}_sem"

            y_rel, y_se_rel = relativize(
                means_t=trial_df[mean_col],
                sems_t=trial_df[sem_col],
                mean_c=status_quo_row[mean_col].values[0],
                sem_c=status_quo_row[sem_col].values[0],
            )

            rel_df.loc[rel_df["trial_index"] == trial_idx, mean_col] = y_rel
            rel_df.loc[rel_df["trial_index"] == trial_idx, sem_col] = y_se_rel

    return rel_df
