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
from ax.adapter.base import Adapter
from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.trial_status import TrialStatus
from ax.core.types import ComparisonOp
from ax.core.utils import get_target_trial_index
from ax.exceptions.core import AxError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.stats.math_utils import relativize
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

    if (adapter := generation_strategy.adapter) is not None:
        return adapter

    if experiment is None:
        raise UserInputError(
            "Provided GenerationStrategy has no adapter, but no Experiment was "
            "provided to source data to fit the adapter."
        )

    generation_strategy.current_node._fit(experiment=experiment)
    adapter = generation_strategy.adapter

    if adapter is None:
        raise UserInputError(
            "Currently, Ax has not yet reached a GenerationNode that involves fitting "
            "a surrogate model, from which it can make predictions about points in the "
            "search space (current GenerationNode: "
            f"{generation_strategy.current_node}). This analysis will become available "
            "once that optimization state is reached, later in the course of the "
            "experiment. To generate this analysis on-demand when interacting with the "
            "Analysis directly, please provide an Adapter."
        )

    return adapter


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

        if not adapter.can_model_in_sample:
            logger.info(
                "Provided adapter is unable to model effects. "
                "Using Empirical Bayes as falback."
            )

            trial_indices = None  # This will indicate all trials to `lookup_data`
            if trial_index is not None:
                trial_indices = {trial_index}
                # always look up target trial data even if it's outside the filters
                if target_trial_index is not None:
                    trial_indices.add(target_trial_index)
            data = experiment.lookup_data(trial_indices=trial_indices)
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
            target_trial_index=target_trial_index,
        )
    raw_df = df
    has_relative_constraints = experiment.optimization_config is not None and any(
        oc.relative for oc in experiment.optimization_config.outcome_constraints
    )
    status_quo_df = None
    if relativize or has_relative_constraints:
        is_raw_data = not use_model_predictions
        status_quo_df = _get_status_quo_df(
            experiment=experiment,
            df=raw_df,
            metric_names=metric_names,
            is_raw_data=is_raw_data,
            trial_index=trial_index,
            trial_statuses=trial_statuses,
            target_trial_index=target_trial_index,
        )
        if relativize:
            df = relativize_data(
                experiment=experiment,
                df=df,
                metric_names=metric_names,
                is_raw_data=is_raw_data,
                trial_index=trial_index,
                trial_statuses=trial_statuses,
                target_trial_index=target_trial_index,
                status_quo_df=status_quo_df,
            )

    # Add additional columns which do not require predicting or extracting data.
    df["trial_status"] = df["trial_index"].apply(
        lambda trial_index: experiment.trials[trial_index].status.name
        if trial_index != -1
        else "Additional Arm"
    )
    df["fail_reason"] = df["trial_index"].apply(
        lambda trial_index: experiment.trials[trial_index].failed_reason
        if trial_index != -1
        and experiment.trials[trial_index].failed_reason is not None
        else None
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
    df["p_feasible_mean"] = _prepare_p_feasible(
        df=raw_df,
        status_quo_df=status_quo_df,
        outcome_constraints=experiment.optimization_config.outcome_constraints
        if experiment.optimization_config is not None
        else [],
    )
    df["p_feasible_sem"] = np.nan

    # Earlier we add target trial data to the df to support relativization easily
    # but if a specific trial, or trial statuses were requested, and the target trial
    # isn't in that subset, let's remove it so it isn't plotted
    if target_trial_index is not None:
        if trial_index is not None and trial_index != target_trial_index:
            df = df[df["trial_index"] != target_trial_index]
        elif (
            trial_statuses is not None
            and experiment.trials[target_trial_index].status not in trial_statuses
        ):
            df = df[df["trial_index"] != target_trial_index]

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

    # Filter trials by trial_index, target_trial_index, and trial_statuses
    # Determine what to pass to extract_relevant_trials:
    # - None: no filtering, get all trials (when trial_index is None)
    # - []: no trials (when trial_index=-1 and no target_trial_index)
    # - [indices]: specific trials (when trial_index is specified and not None)
    if trial_index is None:
        trial_indices_arg = None
    else:
        trial_indices_to_filter: set[int] = set()
        if trial_index != -1:
            trial_indices_to_filter.add(trial_index)
        if target_trial_index is not None:
            trial_indices_to_filter.add(target_trial_index)
        trial_indices_arg = list(trial_indices_to_filter)

    filtered_trials = experiment.extract_relevant_trials(
        trial_indices=trial_indices_arg,
        trial_statuses=trial_statuses,
    )

    # Exclude abandoned arms if the trial is of type BatchTrial
    # https://www.internalfb.com/code/fbsource/[a19525e3f9e6]/fbcode/ax/core/batch_trial.py?lines=51
    # Note: abandoned arms are expected to be excluded in all modeling and predictions.
    # If there is a use case to include abandoned arms, please modify this logic.
    trial_index_arm_pairs = [
        (trial.index, arm)
        for trial in filtered_trials
        for arm in (
            [
                arm
                for arm in trial.arms
                if arm.name
                not in [
                    abandoned_arm.name
                    for abandoned_arm in trial.abandoned_arms_metadata
                ]
            ]
            if isinstance(trial, BatchTrial)
            else trial.arms
        )
    ]
    # Add additional arms passed in by the user
    trial_index_arm_pairs += [(-1, arm) for arm in additional_arms or []]

    # Remove arms with missing parameters since we cannot predict for them.
    predictable_pairs = []
    unpredictable_pairs = []
    for trial_index, arm in trial_index_arm_pairs:
        if adapter.model_space.check_membership(
            parameterization=arm.parameters,
            raise_error=False,
            check_all_parameters_present=True,
        ):
            predictable_pairs.append((trial_index, arm))
        else:
            unpredictable_pairs.append((trial_index, arm))

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
    target_trial_index: int | None,
) -> pd.DataFrame:
    """
    Extract the raw (mean, sem) for each arm for each requested metric.

    Returns a DataFrame with the following columns:
        - trial_index
        - arm_name
        - METRIC_NAME_mean for each metric_name in metric_names
        - METRIC_NAME_sem for each metric_name in metric_names
    """
    # Trial index of -1 indicates candidate arms, we don't have observed data for
    # hypothetical arms so return an empty df.
    if trial_index == -1 and target_trial_index is None:
        return pd.DataFrame(
            columns=(
                [
                    "trial_index",
                    "arm_name",
                    *[f"{metric_name}_mean" for metric_name in metric_names],
                    *[f"{metric_name}_sem" for metric_name in metric_names],
                ]
            )
        )
    else:
        # Filter trials by trial_index, target_trial_index, and trial_statuses
        # Determine what to pass to extract_relevant_trials:
        # - None: no filtering, get all trials (when trial_index is None)
        # - []: no trials (when trial_index=-1 and no target_trial_index)
        # - [indices]: specific trials (when trial_index is specified and not None)
        if trial_index is None:
            trial_indices_arg = None
        else:
            trial_indices_to_filter: set[int] = set()
            if trial_index != -1:
                trial_indices_to_filter.add(trial_index)
            if target_trial_index is not None:
                trial_indices_to_filter.add(target_trial_index)
            trial_indices_arg = list(trial_indices_to_filter)

        trials = experiment.extract_relevant_trials(
            trial_indices=trial_indices_arg,
            trial_statuses=trial_statuses,
        )
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
    status_quo_df: pd.DataFrame | None,
    outcome_constraints: Sequence[OutcomeConstraint],
) -> pd.Series:
    """
    Compute the probability that each arm is feasible with respect to the given
    outcome constraints (assuming normally distributed observations). Calculated in a
    batch for efficiency.


    Args:
        df: Result of _prepare_modeled_arm_data or _prepare_raw_arm_data.
        outcome_constraints: The outcome constraints to use to compute the probability
            of feasibility.

    Returns:
        A Series with one entry per row in df describing the probability that the arm
        is feasible with respect to the given outcome constraints.
    """
    rel_df = None
    if any(c.relative for c in outcome_constraints):
        if status_quo_df is None:
            raise AxError("Must provide status quo data to relativize data.")
        rel_df = _relativize_df_with_sq(
            df=df, status_quo_df=status_quo_df, as_percent=True
        )
    if len(outcome_constraints) == 0:
        return pd.Series(np.ones(len(df)))

    # If an arm is missing data for a metric leave the mean as NaN.
    oc_names = []
    for oc in outcome_constraints:
        if isinstance(oc, ScalarizedOutcomeConstraint):
            # take the str representation of the scalarized outcome constraint
            oc_names.append(str(oc))
        else:
            oc_names.append(oc.metric.name)

    assert len(oc_names) == len(outcome_constraints)

    means = []
    sigmas = []
    for i, oc_name in enumerate(oc_names):
        df_constraint = none_throws(rel_df if outcome_constraints[i].relative else df)
        # TODO[T235432214]: currently we are leaving the mean as NaN if the constraint
        # is on ScalarizedOutcomeConstraint but we should be able to calculate it by
        # setting the mean to be weights * individual metrics and sem to be
        # sqrt(sum((weights * individual_sems)^2)), assuming independence.
        if f"{oc_name}_mean" in df_constraint.columns:
            means.append(df_constraint[f"{oc_name}_mean"].tolist())

        else:
            means.append(np.nan * np.ones(len(df_constraint)))
        sigmas.append(
            (df_constraint[f"{oc_name}_sem"].fillna(0)).tolist()
            if f"{oc_name}_sem" in df_constraint.columns
            else [0] * len(df)
        )

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


def _relativize_df_with_sq(
    df: pd.DataFrame,
    status_quo_df: pd.DataFrame,
    as_percent: bool = False,
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
                as_percent=as_percent,
            )

            rel_df.loc[rel_df["trial_index"] == trial_idx, mean_col] = y_rel
            rel_df.loc[rel_df["trial_index"] == trial_idx, sem_col] = y_se_rel

    return rel_df


def _get_sq_arm_name(experiment: Experiment) -> str:
    """
    Retrieve the name of the status quo arm from the given experiment.

    Args:
        experiment: An Ax experiment.
    Returns:
        The name of the status quo arm.
    Raises:
        UserInputError: If the experiment does not have a status quo arm or if the
        status quo arm is not named.

    """
    if experiment.status_quo is None:
        raise UserInputError(
            "Cannot relativize data without a status quo arm on the experiment."
        )

    if experiment.status_quo.name is None:
        raise UserInputError(
            "Cannot relativize data without a named status quo arm on the "
            "experiment."
        )

    return experiment.status_quo.name


def relativize_data(
    experiment: Experiment,
    df: pd.DataFrame,
    metric_names: Sequence[str],
    is_raw_data: bool,
    trial_index: int | None,
    trial_statuses: Sequence[TrialStatus] | None,
    target_trial_index: int | None,
    status_quo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Relativize the data in the given DataFrame with respect to the status quo arm.
    This method includes logic to select the appropriate status quo arm to use
    for relativization. If relativizing raw data, the status quo is taken from
    the inputted trial and falls back to the only status quo arm on the experiment
    if the former is not avaialble. If relativizing modeled data, the status quo is
    taken from the target trial and falls back to the raw status quo arm on the
    target trial if the former is not available.

    Args:
        experiment: An Ax experiment.
        df: The DataFrame containing the data to be relativized. Must include columns
            'METRIC_NAME_mean', 'METRIC_NAME_sem', trial_index, and arm_name.
        metric_names: The names of the metrics to relativize.
        is_raw_data: Whether the data is raw or modeled.
        trial_index: If present, only relativize data from the trial with the given
            index.
        trial_statuses: If present, only relativize data from trials with statuses in
            this collection.
        target_trial_index: The index of the target trial.
        status_quo_df: The status quo data for each trial.
    Returns:
        A new DataFrame with the same structure as the input, but with
            'METRIC_NAME_mean'
        and 'METRIC_NAME_sem' columns relativized to the status quo arm for
            each metric
        within each trial.
    Raises:
        UserInputError: If the experiment does not have a status quo arm, if the status
            quo arm is not named, or if no status quo arm is found in the experiment.
    """
    if status_quo_df is None:
        status_quo_df = _get_status_quo_df(
            experiment=experiment,
            df=df,
            metric_names=metric_names,
            is_raw_data=is_raw_data,
            trial_index=trial_index,
            trial_statuses=trial_statuses,
            target_trial_index=target_trial_index,
        )
    return _relativize_df_with_sq(df=df, status_quo_df=status_quo_df)


def _get_status_quo_df(
    experiment: Experiment,
    df: pd.DataFrame,
    metric_names: Sequence[str],
    is_raw_data: bool,
    trial_index: int | None,
    trial_statuses: Sequence[TrialStatus] | None,
    target_trial_index: int | None,
) -> pd.DataFrame:
    status_quo_name = _get_sq_arm_name(experiment=experiment)

    if not is_raw_data:
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
                target_trial_index=target_trial_index,
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

        status_quo_df = pd.concat([status_quo_row] * len(df["trial_index"].unique()))
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
        for trial_idx in df["trial_index"].unique():
            trial_mask = (df["trial_index"] == trial_idx) & status_quo_mask
            if trial_mask.any():
                row = df[trial_mask].iloc[0]
            elif status_quo_mask.sum() == 1:
                row = df[status_quo_mask].iloc[0]
            else:
                raise UserInputError(
                    "Failed to relativize, no status quo arm found in the "
                    "experiment."
                )

            row["trial_index"] = trial_idx
            status_quo_rows.append(row)

        status_quo_df = pd.DataFrame(status_quo_rows)[
            [
                "trial_index",
                *[f"{name}_mean" for name in metric_names],
                *[f"{name}_sem" for name in metric_names],
            ]
        ]
    return status_quo_df


def get_lower_is_better(experiment: Experiment, metric_name: str) -> bool | None:
    if metric_name == "p_feasible":
        return False
    return experiment.metrics[metric_name].lower_is_better


def validate_experiment(
    experiment: Experiment | None,
    require_trials: bool = False,
    require_data: bool = False,
) -> str | None:
    """
    Validate that the Analysis has been provided an Experiment, and optionally check if
    the Experiment has trials and data.

    Typically used in Analysis.validate_applicable_state(...), which is called before
    Analysis.compute(...) in Analysis.compute_result(...).
    """

    if experiment is None:
        return "Requires an Experiment."

    if require_trials:
        if len(experiment.trials) == 0:
            return "Experiment has no trials."

    if require_data:
        if experiment.lookup_data().df.empty:
            return "Experiment has no data."


def validate_experiment_has_trials(
    experiment: Experiment,
    trial_indices: Sequence[int] | None,
    trial_statuses: Sequence[TrialStatus] | None,
    required_metric_names: Sequence[str] | None,
) -> str | None:
    filtered_trials = experiment.extract_relevant_trials(
        trial_indices=trial_indices,
        trial_statuses=trial_statuses,
    )

    if len(filtered_trials) == 0:
        return f"Experiment has no trials in {trial_indices=} with {trial_statuses=}."

    if required_metric_names is not None:
        filtered_trial_indices = [trial.index for trial in filtered_trials]
        metric_names = (
            experiment.lookup_data(trial_indices=filtered_trial_indices)
            .df["metric_name"]
            .unique()
        )

        missing_metrics = {*required_metric_names} - {*metric_names}

        if len(missing_metrics) > 0:
            return (
                f"Experiment has no data for metrics {missing_metrics} in "
                f"{trial_indices=} with {trial_statuses=}."
            )


def validate_adapter_can_predict(
    experiment: Experiment | None,
    generation_strategy: GenerationStrategy | None,
    adapter: Adapter | None,
    required_metric_names: Sequence[str] | None,
) -> str | None:
    # If using model predictions ensure we have an Adapter which can predict
    try:
        adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if not adapter.can_predict:
            return (
                f"Adapter {adapter} does not support predictions, please "
                "use use_model_predictions=False, provide a suitable "
                "Adapter, or wait until GenerationStrategy reaches a "
                "GenerationNode with an adapter that is able to predict."
            )

        if required_metric_names is not None:
            experiment = none_throws(experiment)

            required_metric_signatures = [
                experiment.metrics[name].signature for name in required_metric_names
            ]

            missing_metric_signatures = {*required_metric_signatures} - {
                *adapter.metric_signatures
            }

            missing_metric_names = [
                experiment.signature_to_metric[signature].name
                for signature in missing_metric_signatures
            ]

            if len(missing_metric_names) > 0:
                return (
                    f"Adapter {adapter} does not support metrics "
                    f"{missing_metric_names}."
                )

    except UserInputError as e:
        return e.message

    return None
