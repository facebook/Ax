#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.search_space import SearchSpace
from ax.core.trial import BaseTrial, Trial
from ax.modelbridge import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.plot.contour import interact_contour_plotly
from ax.plot.slice import plot_slice_plotly
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger: Logger = get_logger(__name__)


def _get_objective_trace_plot(
    trials: Dict[int, BaseTrial],
    metric_name: str,
    model_transitions: List[int],
    optimization_direction: Optional[str] = None,
    # pyre-ignore[11]: Annotation `go.Figure` is not defined as a type.
) -> Optional[go.Figure]:
    best_objectives = np.array(
        [[checked_cast(Trial, t).objective_mean for t in trials.values()]]
    )
    return optimization_trace_single_method_plotly(
        y=best_objectives,
        title="Best objective found vs. # of iterations",
        ylabel=metric_name,
        model_transitions=model_transitions,
        optimization_direction=optimization_direction,
        plot_trial_points=True,
    )


def _get_objective_v_param_plot(
    search_space: SearchSpace,
    model: ModelBridge,
    metric_name: str,
    trials: Dict[int, BaseTrial],
) -> Optional[go.Figure]:
    range_params = list(search_space.range_parameters.keys())
    if len(range_params) == 1:
        # individual parameter slice plot
        output_slice_plot = plot_slice_plotly(
            model=not_none(model),
            param_name=range_params[0],
            metric_name=metric_name,
            generator_runs_dict={
                str(t.index): not_none(checked_cast(Trial, t).generator_run)
                for t in trials.values()
            },
        )
        return output_slice_plot
    if len(range_params) > 1:
        # contour plot
        output_contour_plot = interact_contour_plotly(
            model=not_none(model),
            metric_name=metric_name,
        )
        return output_contour_plot


def get_standard_plots(
    experiment: Experiment, generation_strategy: GenerationStrategy
) -> List[go.Figure]:
    """Extract standard plots for single-objective optimization.

    TODO: Describe specific logic here on what happens depending on search space
    and generation strategy attributes.
    """
    output_plot_list = []
    output_plot_list.append(
        _get_objective_trace_plot(
            trials=experiment.trials,
            metric_name=not_none(experiment.optimization_config).objective.metric.name,
            model_transitions=generation_strategy.model_transitions,
            optimization_direction=(
                "minimize"
                if not_none(experiment.optimization_config).objective.minimize
                else "maximize"
            ),
        )
    )

    # TODO: Replace this check with a check of whether the model
    # implements `predict`.
    if isinstance(generation_strategy.model, RandomModelBridge):
        return output_plot_list

    output_plot_list.append(
        _get_objective_v_param_plot(
            search_space=experiment.search_space,
            model=not_none(generation_strategy.model),
            metric_name=not_none(experiment.optimization_config).objective.metric.name,
            trials=experiment.trials,
        )
    )

    return [plot for plot in output_plot_list if plot is not None]


def exp_to_df(
    exp: Experiment,
    metrics: Optional[List[Metric]] = None,
    key_components: Optional[List[str]] = None,
    run_metadata_fields: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Transforms an experiment to a DataFrame. Only supports Experiment and
    SimpleExperiment.

    Transforms an Experiment into a dataframe with rows keyed by trial_index
    and arm_name, metrics pivoted into one row.

    Args:
        exp: An Experiment that may have pending trials.
        metrics: Override list of metrics to return. Return all metrics if None.
        key_components: fields that combine to make a unique key corresponding
            to rows, similar to the list of fields passed to a GROUP BY.
            Defaults to ['arm_name', 'trial_index'].
        run_metadata_fields: fields to extract from trial.run_metadata for trial
            in experiment.trials. If there are multiple arms per trial, these
            fields will be replicated across the arms of a trial.
        **kwargs: Custom named arguments, useful for passing complex
            objects from call-site to the `fetch_data` callback.

    Returns:
        DataFrame: A dataframe of inputs and metrics by trial and arm.
    """

    def prep_return(
        df: pd.DataFrame, drop_col: str, sort_by: List[str]
    ) -> pd.DataFrame:
        return not_none(not_none(df.drop(drop_col, axis=1)).sort_values(sort_by))

    key_components = key_components or ["trial_index", "arm_name"]

    # Accept Experiment and SimpleExperiment
    if isinstance(exp, MultiTypeExperiment):
        raise ValueError("Cannot transform MultiTypeExperiments to DataFrames.")

    results = exp.fetch_data(metrics, **kwargs).df
    if len(results.index) == 0:  # Handle empty case
        return results
    key_col = "-".join(key_components)
    key_vals = results[key_components[0]].astype("str")
    for key in key_components[1:]:
        key_vals = key_vals + results[key].astype("str")
    results[key_col] = key_vals

    metric_vals = results.pivot(
        index=key_col, columns="metric_name", values="mean"
    ).reset_index()
    metadata = results[key_components + [key_col]].drop_duplicates()
    metric_and_metadata = pd.merge(metric_vals, metadata, on=key_col)
    arm_names_and_params = pd.DataFrame(
        [{"arm_name": name, **arm.parameters} for name, arm in exp.arms_by_name.items()]
    )

    exp_df = pd.merge(metric_and_metadata, arm_names_and_params, on="arm_name")
    trials = exp.trials.items()
    trial_to_status = {index: trial.status.name for index, trial in trials}
    exp_df["trial_status"] = [trial_to_status[key] for key in exp_df.trial_index]

    if run_metadata_fields is None:
        return prep_return(exp_df, key_col, key_components)
    if not isinstance(run_metadata_fields, list):
        raise ValueError("run_metadata_fields must be List[str] or None.")

    for field in run_metadata_fields:
        trial_to_metadata_field = {
            index: (trial.run_metadata[field] if field in trial.run_metadata else None)
            for index, trial in trials
        }
        if any(trial_to_metadata_field.values()):  # field present for any trial
            if not all(trial_to_metadata_field.values()):  # not present for all trials
                logger.warning(
                    f"Field {field} missing for some trials' run_metadata. "
                    "Returning None when missing."
                )
            exp_df[field] = [trial_to_metadata_field[key] for key in exp_df.trial_index]
        else:
            logger.warning(
                f"Field {field} missing for all trials' run_metadata. "
                "Not appending column."
            )
    return prep_return(exp_df, key_col, key_components)
