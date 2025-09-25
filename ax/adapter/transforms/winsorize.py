#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.core.objective import MultiObjective, ScalarizedObjective
from ax.core.observation import ObservationData
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import AxOptimizationWarning, UserInputError
from ax.generators.types import TConfig, WinsorizationConfig
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


AUTO_WINS_QUANTILE = -1  # This shouldn't be in the [0, 1] range
DEFAULT_CUTOFFS: tuple[float, float] = (-float("inf"), float("inf"))


class Winsorize(Transform):
    """Clip the mean values for each metric to lay within the limits provided in
    the config. The config can contain either or both of two keys:
    - ``"winsorization_config"``, corresponding to either a single
        ``WinsorizationConfig``, which, if provided will be used for all metrics; or
        a mapping ``Dict[str, WinsorizationConfig]`` between each metric name and its
        ``WinsorizationConfig``.
    - ``"derelativize_with_raw_status_quo"``, indicating whether to use the raw
        status-quo value for any derelativization. Note this defaults to ``False``,
        which is unsupported and simply fails if derelativization is necessary. The
        user must specify ``derelativize_with_raw_status_quo = True`` in order for
        derelativization to succeed. Note that this must match the `use_raw_status_quo`
        value in the ``Derelativize`` config if used.
    For example,
    ``{"winsorization_config": WinsorizationConfig(lower_quantile_margin=0.3)}``
    will specify the same 30% winsorization from below for all metrics, whereas
    ```
    {
        "winsorization_config":
        {
            "metric_1": WinsorizationConfig(lower_quantile_margin=0.2),
            "metric_2": WinsorizationConfig(upper_quantile_margin=0.1),
        }
    }
    ```
    will winsorize 20% from below for metric_1 and 10% from above from metric_2.
    Additional metrics won't be winsorized.

    You can also determine the winsorization cutoffs automatically without having an
    ``OptimizationConfig`` by passing in AUTO_WINS_QUANTILE for the quantile you want
    to winsorize. For example, to automatically winsorize large values:
        ``"m1": WinsorizationConfig(upper_quantile_margin=AUTO_WINS_QUANTILE)``.
    This may be useful when fitting models in a notebook where there is no corresponding
    ``OptimizationConfig``.

    Additionally, you can pass in winsorization boundaries ``lower_boundary`` and
    ``upper_boundary``that specify a maximum allowable amount of winsorization. This
    is discouraged and will eventually be deprecated as we strongly encourage
    that users allow ``Winsorize`` to automatically infer these boundaries from
    the optimization config.
    """

    requires_data_for_initialization: bool = True

    cutoffs: dict[str, tuple[float, float]]

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        optimization_config = adapter._optimization_config if adapter else None
        if config is None and optimization_config is None:
            raise UserInputError(
                "Transform config for `Winsorize` transform must be specified and "
                "non-empty when using winsorization, or an adapter containing an "
                "optimization_config must be provided."
            )
        if config is None:
            config = {}

        # Get config settings.
        winsorization_config = config.get("winsorization_config", {})
        use_raw_sq = _get_and_validate_use_raw_sq(config=config)
        means_df = none_throws(experiment_data).observation_data["mean"]
        # Dropping NaNs here since the DF will have NaN for missing values.
        all_metric_values = {
            signature: column.dropna().values for signature, column in means_df.items()
        }
        self.cutoffs = {
            metric_signature: _get_cutoffs(
                metric_signature=metric_signature,
                metric_values=metric_values,
                winsorization_config=winsorization_config,
                adapter=adapter,
                optimization_config=optimization_config,
                use_raw_sq=use_raw_sq,
            )
            for metric_signature, metric_values in all_metric_values.items()
        }

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for idx, metric_signature in enumerate(obsd.metric_signatures):
                if metric_signature not in self.cutoffs:
                    raise ValueError(
                        f"Cannot winsorize unknown metric {metric_signature}"
                    )
                # Clip on the winsorization bounds.
                obsd.means[idx] = max(
                    obsd.means[idx], self.cutoffs[metric_signature][0]
                )
                obsd.means[idx] = min(
                    obsd.means[idx], self.cutoffs[metric_signature][1]
                )
        return observation_data

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        obs_data = experiment_data.observation_data
        # NOTE: Operating on metrics one by one rather than calling
        # obs_data["mean"].clip with dict valued bounds, since profiling
        # showed this to be faster. I suspect calling with dict values
        # operates over rows, which is less efficient than operating over columns.
        for m, (lower, upper) in self.cutoffs.items():
            # Observation data columns are multi-indexed, with first level index
            # being "mean" or "sem" and the second level being the metric name.
            # This here updates the "mean" column for the given metric in-place.
            obs_data["mean", m] = obs_data["mean", m].clip(lower=lower, upper=upper)
        return ExperimentData(
            arm_data=experiment_data.arm_data, observation_data=obs_data
        )


def _get_cutoffs(
    metric_signature: str,
    metric_values: list[float],
    winsorization_config: WinsorizationConfig | dict[str, WinsorizationConfig],
    adapter: Optional["adapter_module.base.Adapter"],
    optimization_config: OptimizationConfig | None,
    use_raw_sq: bool,
) -> tuple[float, float]:
    # (1) Use the same config for all metrics if one WinsorizationConfig was specified
    if isinstance(winsorization_config, WinsorizationConfig):
        return _quantiles_to_cutoffs(
            metric_signature=metric_signature,
            metric_values=metric_values,
            metric_config=winsorization_config,
        )

    # (2) If `winsorization_config` is a dict, use it if `metric_signature` is a key,
    # and the corresponding value is a WinsorizationConfig.
    if (
        isinstance(winsorization_config, dict)
        and metric_signature in winsorization_config
    ):
        metric_config = winsorization_config[metric_signature]
        if not isinstance(metric_config, WinsorizationConfig):
            raise UserInputError(
                "Expected winsorization config of type "
                f"`WinsorizationConfig` but got {metric_config} of type "
                f"{type(metric_config)} for metric {metric_signature}."
            )
        return _quantiles_to_cutoffs(
            metric_signature=metric_signature,
            metric_values=metric_values,
            metric_config=metric_config,
        )

    # (3) For constraints and objectives that don't have a pre-specified config we
    # choose the cutoffs automatically using the optimization config (if supplied).
    # We ignore ScalarizedOutcomeConstraint and ScalarizedObjective for now, and
    # derelativize relative constraints if possible.

    # When no optimization config is available, return defaults.
    if adapter is None or optimization_config is None:
        return DEFAULT_CUTOFFS
    relative_constraint_metrics = {
        metric.name
        for oc in optimization_config.all_constraints
        for metric in (
            oc.metrics if isinstance(oc, ScalarizedOutcomeConstraint) else [oc.metric]
        )
        if oc.relative
    }
    if metric_signature in relative_constraint_metrics:
        if not use_raw_sq:
            warnings.warn(
                "Automatic winsorization doesn't support relative outcome constraints "
                "or objective thresholds when `derelativize_with_raw_status_quo` is "
                "not set to `True`. Skipping winsorization for metric "
                f"{metric_signature}.",
                AxOptimizationWarning,
                stacklevel=3,
            )
            return DEFAULT_CUTOFFS
        optimization_config = derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config, adapter=adapter
        )

    # Non-objective metrics - obtain cutoffs from outcome_constraints.
    if metric_signature not in optimization_config.objective.metric_signatures:
        # Get all outcome constraints for `metric_signature`` that aren't scalarized.
        return _obtain_cutoffs_from_outcome_constraints(
            optimization_config=optimization_config,
            metric_signature=metric_signature,
            metric_values=metric_values,
        )

    # Make sure we winsorize from the correct direction if a `ScalarizedObjective`.
    if isinstance(optimization_config.objective, ScalarizedObjective):
        objective = assert_is_instance(
            optimization_config.objective, ScalarizedObjective
        )
        weight = [
            w for m, w in objective.metric_weights if m.signature == metric_signature
        ][0]
        # Winsorize from above if the weight is positive and minimize is `True` or the
        # weight is negative and minimize is `False`.
        return _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values,
            minimize=objective.minimize if weight >= 0 else not objective.minimize,
        )

    # Single-objective
    if not optimization_config.is_moo_problem:
        return _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values,
            minimize=optimization_config.objective.minimize,
        )

    # Multi-objective
    return _get_auto_winsorization_cutoffs_multi_objective(
        optimization_config=optimization_config,
        metric_signature=metric_signature,
        metric_values=metric_values,
    )


def _get_auto_winsorization_cutoffs_multi_objective(
    optimization_config: OptimizationConfig,
    metric_signature: str,
    metric_values: list[float],
) -> tuple[float, float]:
    # We approach a multi-objective metric the same as output constraints. It may be
    # worth investigating setting the winsorization cutoffs based on the Pareto
    # frontier in the future.
    optimization_config = assert_is_instance(
        optimization_config, MultiObjectiveOptimizationConfig
    )
    objective_threshold = _get_objective_threshold_from_moo_config(
        optimization_config=optimization_config, metric_signature=metric_signature
    )
    if objective_threshold:
        return _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=metric_values,
            outcome_constraints=objective_threshold,
        )
    else:
        warnings.warn(
            "Encountered a `MultiObjective` without objective thresholds. We will "
            "winsorize each objective separately. We strongly recommend specifying "
            "the objective thresholds when using multi-objective optimization.",
            AxOptimizationWarning,
            stacklevel=3,
        )
        objectives = assert_is_instance(optimization_config.objective, MultiObjective)
        minimize = [
            objective.minimize
            for objective in objectives.objectives
            if objective.metric.signature == metric_signature
        ][0]
        return _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values,
            minimize=minimize,
        )


def _obtain_cutoffs_from_outcome_constraints(
    optimization_config: OptimizationConfig,
    metric_signature: str,
    metric_values: list[float],
) -> tuple[float, float]:
    """Get all outcome constraints (non-scalarized) for a given metric."""
    # Check for scalarized outcome constraints for the given metric
    if any(
        isinstance(oc, ScalarizedOutcomeConstraint)
        and metric_signature in [metric.signature for metric in oc.metrics]
        for oc in optimization_config.outcome_constraints
    ):
        warnings.warn(
            "Automatic winsorization isn't supported for a "
            "`ScalarizedOutcomeConstraint`. Specify the winsorization settings "
            f"manually if you want to winsorize metric {metric_signature}.",
            AxOptimizationWarning,
            stacklevel=3,
        )
    outcome_constraints = _get_non_scalarized_outcome_constraints(
        optimization_config=optimization_config, metric_signature=metric_signature
    )
    if outcome_constraints:
        return _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=metric_values,
            outcome_constraints=outcome_constraints,
        )
    return DEFAULT_CUTOFFS


def _get_non_scalarized_outcome_constraints(
    optimization_config: OptimizationConfig, metric_signature: str
) -> list[OutcomeConstraint]:
    return [
        oc
        for oc in optimization_config.outcome_constraints
        if not isinstance(oc, ScalarizedOutcomeConstraint)
        and oc.metric.signature == metric_signature
    ]


def _get_objective_threshold_from_moo_config(
    optimization_config: MultiObjectiveOptimizationConfig, metric_signature: str
) -> list[ObjectiveThreshold]:
    """Get the non-relative objective threshold for a given metric."""
    return [
        ot
        for ot in optimization_config.objective_thresholds
        if ot.metric.signature == metric_signature
    ]


def _get_tukey_cutoffs(Y: npt.NDArray, lower: bool) -> float:
    """Compute winsorization cutoffs similarly to Tukey boxplots.

    See https://mathworld.wolfram.com/Box-and-WhiskerPlot.html for more details.
    """
    q1 = np.percentile(Y, q=25, method="lower")
    q3 = np.percentile(Y, q=75, method="higher")
    iqr = q3 - q1
    return q1 - 1.5 * iqr if lower else q3 + 1.5 * iqr


def _get_auto_winsorization_cutoffs_single_objective(
    metric_values: list[float], minimize: bool
) -> tuple[float, float]:
    """Automatic winsorization for a single objective.

    We use a heuristic similar to what is used for Tukey box-plots in order to determine
    what is an outlier. If we are minimizing we make sure that we winsorize large values
    and if we maximize we winsorize small values.
    """
    Y = np.array(metric_values)
    if minimize:
        return (DEFAULT_CUTOFFS[0], _get_tukey_cutoffs(Y, lower=False))
    else:
        return (_get_tukey_cutoffs(Y, lower=True), DEFAULT_CUTOFFS[1])


def _get_auto_winsorization_cutoffs_outcome_constraint(
    metric_values: list[float],
    outcome_constraints: list[ObjectiveThreshold] | list[OutcomeConstraint],
) -> tuple[float, float]:
    """Automatic winsorization to an outcome constraint.

    We need to be careful here so we don't make infeasible points feasible.
    While it is possible to winsorize from both ends, we only winsorize from the
    infeasible direction for now so the same method can be used for MOO. We rely on
    a heuristic similar to `_get_tukey_cutoffs`, but instead take the max of
    q3 and the bound when we have a LEQ constraint and the min of q1 and the bound
    with a GEQ constraint.
    """
    Y = np.array(metric_values)
    q1 = np.percentile(Y, q=25, method="lower")
    q3 = np.percentile(Y, q=75, method="higher")
    lower_cutoff, upper_cutoff = DEFAULT_CUTOFFS
    for oc in outcome_constraints:
        bnd = oc.bound
        if oc.op == ComparisonOp.LEQ:
            upper_cutoff = max(q3, bnd) + 1.5 * (max(q3, bnd) - q1)
        elif oc.op == ComparisonOp.GEQ:
            lower_cutoff = min(q1, bnd) - 1.5 * (q3 - min(q1, bnd))
        else:
            raise ValueError("Expected outcome_constraint to use operator LEQ or GEQ")
    return lower_cutoff, upper_cutoff


def _quantiles_to_cutoffs(
    metric_signature: str,
    metric_values: list[float],
    metric_config: WinsorizationConfig,
) -> tuple[float, float]:
    """Compute winsorization cutoffs from a config and values."""
    Y = np.array(metric_values)
    lower = metric_config.lower_quantile_margin or 0.0
    upper = metric_config.upper_quantile_margin or 0.0
    bnd_l = metric_config.lower_boundary
    bnd_u = metric_config.upper_boundary
    if (
        lower != AUTO_WINS_QUANTILE
        and upper != AUTO_WINS_QUANTILE
        and lower >= 1 - upper
    ):
        raise ValueError(
            f"Lower bound: {lower} was greater than the inverse of the "
            f"upper bound: {1 - upper} for metric {metric_signature}. Decrease "
            f"one or both of `lower_quantile_margin` and "
            "`upper_quantile_margin`."
        )
    if lower == AUTO_WINS_QUANTILE:
        cutoff_l = _get_tukey_cutoffs(Y=Y, lower=True)
    elif lower == 0.0:  # Use the default cutoff if there is no winsorization
        cutoff_l = DEFAULT_CUTOFFS[0]
    else:
        cutoff_l = np.percentile(Y, lower * 100, method="lower")

    if upper == AUTO_WINS_QUANTILE:
        cutoff_u = _get_tukey_cutoffs(Y=Y, lower=False)
    elif upper == 0.0:  # Use the default cutoff if there is no winsorization
        cutoff_u = DEFAULT_CUTOFFS[1]
    else:
        cutoff_u = np.percentile(Y, (1 - upper) * 100, method="higher")

    cutoff_l = min(cutoff_l, bnd_l) if bnd_l is not None else cutoff_l
    cutoff_u = max(cutoff_u, bnd_u) if bnd_u is not None else cutoff_u
    return (cutoff_l, cutoff_u)


def _get_and_validate_use_raw_sq(config: TConfig) -> bool:
    use_raw_sq = config.get("derelativize_with_raw_status_quo", False)
    if isinstance(use_raw_sq, bool):
        return use_raw_sq
    raise UserInputError(
        f"`derelativize_with_raw_status_quo` must be a boolean. Got {use_raw_sq}."
    )
