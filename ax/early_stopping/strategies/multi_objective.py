# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from logging import Logger
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

import pandas as pd
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import align_partial_results
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger: Logger = get_logger(__name__)


class AbstractScaledParetoEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """
    This is an early stopping strategy for multi-objective optimization problems. It
    consists of rescaling the empirical Pareto frontier toward the reference point by a
    scalar or vector. Early stopping decisions are made whenever a trial is not weakly
    Pareto efficient w.r.t. the rescaled Pareto frontier at the current progression.
    """

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 300,
        min_progression: Optional[float] = 10,
        max_progression: Optional[float] = None,
        min_curves: Optional[int] = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
        normalize_progressions: bool = False,
        pareto_scaling_factor: float = 1 / 2,
        ref_point: Optional[Dict[str, float]] = None,
    ) -> None:
        """Construct a AbstractScaledParetoEarlyStoppingStrategy instance.

        Args:
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
                NOTE: In addition, ScaledParetoEarlyStoppingStrategy requires that at
                least `min_curves` trials have objectives that dominate the reference
                point. If this condition is not met, no trials will be early stopped at
                this particular time step.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
            pareto_scaling_factor: A scalar in [0, 1] to rescale the Pareto frontier.
            ref_point: Optional reference point for the Pareto frontier. If None, the
                reference point is inferred from the objective thresholds of the
                optimization config.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            trial_indices_to_ignore=trial_indices_to_ignore,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
        )
        self._ref_point = ref_point  # IDEA: this could be time-dependent.

    @abstractmethod
    def _scale_pareto_frontier(self, Y: np.ndarray) -> np.ndarray:
        """Rescales the objective values along each dimension.

        Args:
            Y: An (n x m)-dim array of values of objectives whose reference point is
                assumed to be the origin and that are to be maximized. This is done
                through appropriate scaling and centering of the raw objective values
                in `should_stop_trial_early`.

        Returns:
            An (n x m)-dim array of rescaled objective values.
        """
        pass

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:

        optimization_config = not_none(experiment.optimization_config)
        if not isinstance(optimization_config.objective, MultiObjective):
            raise ValueError(
                "Scaled Pareto early stopping strategy requires a MultiObjective."
            )

        metric_names, directions = self._all_objectives_and_directions(
            experiment=experiment
        )
        data = self._check_validity_and_get_data(
            experiment=experiment, metric_names=metric_names
        )
        if data is None:
            # don't stop any trials if we don't get data back
            return {}

        map_key = next(iter(data.map_keys))
        df = data.map_df

        # default checks on `min_progression` and `min_curves`; if not met, don't do
        # early stopping at all and return {}
        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=df, map_key=map_key
        ):
            return {}

        try:  # align metrics
            # IDEA: could incorporate sems into strategy
            metric_to_aligned_means, _ = align_partial_results(
                df=df,
                progr_key=map_key,
                metrics=metric_names,
            )
        except Exception as e:
            logger.warning(
                f"Encountered exception while aligning data: {e}. "
                "Not early stopping any trials."
            )
            return {}

        decisions = {
            trial_index: self._should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                metric_dict=metric_to_aligned_means,
                df_raw=df,
                map_key=map_key,
                directions=directions,
            )
            for trial_index in trial_indices
        }
        return {
            trial_index: reason
            for trial_index, (should_stop, reason) in decisions.items()
            if should_stop
        }

    def _should_stop_trial_early(
        self,
        trial_index: int,
        experiment: Experiment,
        metric_dict: Dict[str, pd.DataFrame],
        df_raw: pd.DataFrame,
        map_key: str,
        directions: Dict[str, bool],
    ) -> Tuple[bool, Optional[str]]:
        """Stop a trial if its performance doesn't reach a pre-specified threshold
        by `min_progression`.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            metric_dict: Dictionary mapping a metric to a Dataframe of partial results.
            map_key: Name of the column of the dataset that indicates progression.
            directions: Indicates whether an objective is being minimized.

        Returns:
            A tuple `(should_stop, reason)`, where `should_stop` is `True` iff the
            trial should be stopped, and `reason` is an (optional) string providing
            information on why the trial should or should not be stopped.
        """
        logger.info(f"Considering trial {trial_index} for early stopping.")

        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index,
            experiment=experiment,
            df=df_raw,
            map_key=map_key,
        )
        if not stopping_eligible:
            return False, reason

        # Find last progression of the trial for which *all* metrics are available
        trial_last_prog = np.inf
        for _, df in metric_dict.items():
            # dropna() here will exclude trials that have not made it to the
            # last progression of the trial under consideration, and therefore
            # can't be included in the comparison
            df_trial = not_none(df[trial_index].dropna())
            trial_last_prog = min(trial_last_prog, df_trial.index.max())

        (
            trial_data_at_last_prog,
            all_data_at_last_prog,
        ) = self._get_all_data_at_last_prog(
            metric_dict=metric_dict,
            trial_index=trial_index,
            trial_last_prog=trial_last_prog,
            map_key=map_key,
        )

        logger.info("Early stopping objectives at last progression are:\n")
        for metric_name, value in trial_data_at_last_prog.items():
            logger.info(f"\t{metric_name}: {value}.\n")

        ref_point = self._get_ref_point(
            experiment=experiment, metric_names=list(metric_dict.keys())
        )
        (
            trial_array,
            all_array,
            dir_array,
            ref_array,
        ) = self._consolidate_all_data_as_arrays(
            trial_data_at_last_prog=trial_data_at_last_prog,
            all_data_at_last_prog=all_data_at_last_prog,
            directions=directions,
            ref_point=ref_point,
        )

        # TODO: check how to deal with outcome_constraints
        # Checking that at least min_curves number of trials are strictly better than
        # the reference point at the last progression, otherwise don't stop the trial.
        centered_array = dir_array * (all_array - ref_array)
        if (centered_array > 0).all(axis=-1).sum() < self.min_curves:
            reason = (
                f"At least {self.min_curves} trials must be strictly better "
                f"than the reference point ({ref_point}). Not stopping trial "
                f"{trial_index}."
            )
            logger.info(reason)
            return False, reason

        # Scaling objective at the last progression toward the reference point
        scaled_array = self._scale_pareto_frontier(Y=centered_array)

        # Compute if the trial's objective values are not Pareto efficient compared
        # to the scaled data points and make early stopping decision based on this.
        centered_trial_array = dir_array * (trial_array - ref_array)
        not_efficient = (centered_trial_array < scaled_array).all(axis=-1).any()
        should_early_stop = not_efficient
        comp = ("not" if should_early_stop else "") + " Pareto efficient"
        reason = (
            f"Trial objective values are {comp} by the scaled Pareto frontier "
            f"at the progression {trial_last_prog}."
        )
        logger.info(
            f"Early stopping decision for {trial_index}: {should_early_stop}. "
            f"Reason: {reason}"
        )
        return should_early_stop, reason

    def _consolidate_all_data_as_arrays(
        self,
        trial_data_at_last_prog: Dict[str, float],
        all_data_at_last_prog: Dict[str, np.ndarray],
        directions: Dict[str, bool],
        ref_point: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converts the data into numpy arrays for numerical computations, ensuring that
        the metric dimensions match the reference point.

        Returns:
            4-element tuple containing numpy arrays for trial data (m),
            all data (n x m), the directions (m) and the reference point (m).
        """

        def asarray(
            x: Union[Dict[str, float], Dict[str, bool]]
        ) -> Dict[str, np.ndarray]:
            return {k: np.asarray(v) for k, v in x.items()}

        def stack(x: Dict[str, np.ndarray]) -> np.ndarray:
            return np.stack(tuple(x[metric] for metric in ref_point), axis=-1)

        ref_array = stack(asarray(ref_point))  # m
        all_data_array = stack(all_data_at_last_prog)  # (n x m)
        # converting directions from (0, 1) to (1, -1)
        dir_array = 1 - 2 * stack(asarray(directions))  # m
        trial_data_array = stack(asarray(trial_data_at_last_prog))
        return trial_data_array, all_data_array, dir_array, ref_array

    def _get_all_data_at_last_prog(
        self,
        metric_dict: Dict[str, pd.DataFrame],
        trial_index: int,
        trial_last_prog: float,
        map_key: str,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Get all objectives at the last progression for
        1) the trial under consideration for an early stopping decision, and
        2) all other trials.

        Returns:
            Two dictionaries mapping metric names to `n x m` arrays of objective values.
        """
        # Get all objectives all last progression, even non-Pareto optimal ones.
        # Note that we can compute if a trial is weakly efficient by a point without
        # computing the full Pareto frontier, which generally scales exponentially
        # with the number of objectives m. The computation here is O(nm).
        all_data_at_last_prog = {}
        trial_data_at_last_prog = {}
        for metric_name, df in metric_dict.items():
            # Get the data of the trial under consideration for early stopping
            df_trial = not_none(df[trial_index].dropna())
            trial_data_at_last_prog[metric_name] = df_trial.loc[trial_last_prog]

            # Get the data of all other trials for comparison
            all_data_at_last_prog[metric_name] = df.loc[trial_last_prog].to_numpy()

        return trial_data_at_last_prog, all_data_at_last_prog

    def _get_ref_point(
        self, experiment: Experiment, metric_names: List[str]
    ) -> Dict[str, float]:
        if self._ref_point is not None and set(self._ref_point) == set(metric_names):
            return self._ref_point

        optimization_config = not_none(experiment.optimization_config)

        if not isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            raise ValueError(
                "Cannot get reference point from single-objective optimization"
                "configs."
            )

        ref_point = {}
        for thresh in optimization_config.objective_thresholds:
            ref_point[thresh.metric.name] = thresh.bound

        if set(metric_names) != set(ref_point.keys()):
            raise ValueError(
                "Metric names do not match those in the optimization config. Please "
                "ensure that an objective threshold exists for each metric in the "
                "early strategy or pass a reference point to the early stopping "
                f"strategy.\n\t{set(metric_names)=}\n\t{set(ref_point.keys())=}."
            )
        return ref_point


class ScaledParetoEarlyStoppingStrategy(AbstractScaledParetoEarlyStoppingStrategy):
    """
    This is an early stopping strategy for multi-objective optimization problems. It
    consists of rescaling the empirical Pareto frontier toward the reference point by a
    scalar `pareto_scaling_factor`. Early stopping decisions are made whenever a trial
    is not weakly Pareto efficient w.r.t. the rescaled Pareto frontier at the current
    progression.
    """

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 300,
        min_progression: Optional[float] = 10,
        max_progression: Optional[float] = None,
        min_curves: Optional[int] = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
        normalize_progressions: bool = False,
        pareto_scaling_factor: float = 1 / 2,
        ref_point: Optional[Dict[str, float]] = None,
    ) -> None:
        """Construct a ScaledParetoEarlyStoppingStrategy instance.

        Args:
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
                NOTE: In addition, ScaledParetoEarlyStoppingStrategy requires that at
                least `min_curves` trials have objectives that dominate the reference
                point. If this condition is not met, no trials will be early stopped at
                this particular time step.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
            pareto_scaling_factor: A scalar in [0, 1] to rescale the Pareto frontier.
            ref_point: Optional reference point for the Pareto frontier. If None, the
                reference point is inferred from the objective thresholds of the
                optimization config.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            trial_indices_to_ignore=trial_indices_to_ignore,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
            ref_point=ref_point,
        )
        if pareto_scaling_factor < 0 or pareto_scaling_factor > 1:
            raise ValueError("pareto_scaling_factor must be between 0 and 1.")
        self._pareto_scaling_factor = pareto_scaling_factor

    def _scale_pareto_frontier(self, Y: np.ndarray) -> np.ndarray:
        """Rescales the objective values along each dimension so that the maximum value
        of the scaled values is equal to the corresponding percentile of the original
        value of each objective.

        Args:
            Y: An (n x m)-dim array of values of objectives whose reference point is
                assumed to be the origin and that are to be maximized. This is done
                through appropriate scaling and centering of the raw objective values
                in `should_stop_trial_early`.

        Returns:
            An (n x m)-dim array of rescaled objective values such that the maximum of
            each column is equal to the corresponding percentile of the original values.
        """
        return np.asarray(self._pareto_scaling_factor) * Y.clip(0)


class PercentileScaledParetoEarlyStoppingStrategy(
    AbstractScaledParetoEarlyStoppingStrategy
):
    """
    This strategy generalizes the single objective percentile early stopping
    strategy to multi-objective optimization problems. It rescales the empirical
    Pareto frontier toward the reference point by a vector so that the maximum value
    of the rescaled values is equal to the corresponding percentiles of the original
    objectives. Early stopping decisions are made whenever a trial is not weakly Pareto
    efficient w.r.t. the rescaled Pareto frontier at the current progression.
    """

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 300,
        min_progression: Optional[float] = 10,
        max_progression: Optional[float] = None,
        min_curves: Optional[int] = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
        normalize_progressions: bool = False,
        percentile_threshold: float = 50.0,
        ref_point: Optional[Dict[str, float]] = None,
    ) -> None:
        """Construct a ScaledParetoEarlyStoppingStrategy instance.

        Args:
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
                NOTE: In addition, ScaledParetoEarlyStoppingStrategy requires that at
                least `min_curves` trials have objectives that dominate the reference
                point. If this condition is not met, no trials will be early stopped at
                this particular time step.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
            pareto_scaling_factor: A scalar in [0, 1] to rescale the Pareto frontier.
            ref_point: Optional reference point for the Pareto frontier. If None, the
                reference point is inferred from the objective thresholds of the
                optimization config.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            trial_indices_to_ignore=trial_indices_to_ignore,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
            ref_point=ref_point,
        )
        if percentile_threshold < 0 or percentile_threshold > 100:
            raise ValueError("percentile must be between 0 and 100.")
        self._percentile_threshold = percentile_threshold

    def _scale_pareto_frontier(self, Y: np.ndarray) -> np.ndarray:
        """Rescales the objective values along each dimension so that the maximum value
        of the scaled values is equal to the corresponding percentile of the original
        value of each objective.

        Args:
            Y: An (n x m)-dim array of values of objectives whose reference point is
                assumed to be the origin and that are to be maximized. This is done
                through appropriate scaling and centering of the raw objective values
                in `should_stop_trial_early`.

        Returns:
            An (n x m)-dim array of rescaled objective values such that the maximum of
            each column is equal to the corresponding percentile of the original values.
        """
        Y = Y.clip(0)
        quantiles = np.percentile(Y, q=self._percentile_threshold, axis=0)  # m-dim
        # clipping maximum objective values to avoid NaNs
        alpha = quantiles / Y.max(axis=0).clip(1e-12)  # m-dim
        return Y * alpha
