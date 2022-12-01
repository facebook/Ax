#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from logging import Logger
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from ax.core.objective import ScalarizedObjective
from ax.core.observation import Observation, ObservationData
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
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
    get_data,
)
from ax.models.types import TConfig, WinsorizationConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover

logger: Logger = get_logger(__name__)


OLD_KEYS = ["winsorization_lower", "winsorization_upper", "percentile_bounds"]
AUTO_WINS_QUANTILE = -1  # This shouldn't be in the [0, 1] range
DEFAULT_CUTOFFS: Tuple[float, float] = (-float("inf"), float("inf"))


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

    cutoffs: Dict[str, Tuple[float, float]]

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        if observations is None or len(observations) == 0:
            raise DataRequiredError("`Winsorize` transform requires non-empty data.")
        if config is not None and config.get("optimization_config") is not None:
            warnings.warn(
                "Winsorization received an out-of-date `transform_config`, containing "
                'the key `"optimization_config"`. Please update the config according '
                "to the docs of `ax.modelbridge.transforms.winsorize.Winsorize`.",
                DeprecationWarning,
            )
        optimization_config = modelbridge._optimization_config if modelbridge else None
        if config is None and optimization_config is None:
            raise ValueError(
                "Transform config for `Winsorize` transform must be specified and "
                "non-empty when using winsorization, or a modelbridge containing an "
                "optimization_config must be provided."
            )
        if config is None:
            config = {}
        observation_data = [obs.data for obs in observations]

        # Check for legacy config
        use_legacy = False
        old_present = set(OLD_KEYS).intersection(config.keys())
        if old_present:
            warnings.warn(
                "Winsorization received an out-of-date `transform_config`, containing "
                f"the following deprecated keys: {old_present}. Please update the "
                "config according to the docs of "
                "`ax.modelbridge.transforms.winsorize.Winsorize`.",
                DeprecationWarning,
            )
            use_legacy = True

        # Get config settings.
        winsorization_config = config.get("winsorization_config", {})
        use_raw_sq = _get_and_validate_use_raw_sq(config=config)
        self.cutoffs = {}
        all_metric_values = get_data(observation_data=observation_data)
        for metric_name, metric_values in all_metric_values.items():
            if use_legacy:
                self.cutoffs[metric_name] = _get_cutoffs_from_legacy_transform_config(
                    metric_name=metric_name,
                    metric_values=metric_values,
                    transform_config=config,
                )
            else:
                self.cutoffs[metric_name] = _get_cutoffs(
                    metric_name=metric_name,
                    metric_values=metric_values,
                    winsorization_config=winsorization_config,
                    modelbridge=modelbridge,
                    observations=observations,
                    optimization_config=optimization_config,
                    use_raw_sq=use_raw_sq,
                )

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for idx, metric_name in enumerate(obsd.metric_names):
                if metric_name not in self.cutoffs:  # pragma: no cover
                    raise ValueError(f"Cannot winsorize unknown metric {metric_name}")
                # Clip on the winsorization bounds.
                obsd.means[idx] = max(obsd.means[idx], self.cutoffs[metric_name][0])
                obsd.means[idx] = min(obsd.means[idx], self.cutoffs[metric_name][1])
        return observation_data


def _get_cutoffs(
    metric_name: str,
    metric_values: List[float],
    winsorization_config: Union[WinsorizationConfig, Dict[str, WinsorizationConfig]],
    modelbridge: Optional["modelbridge_module.base.ModelBridge"],
    observations: Optional[List[Observation]],
    optimization_config: Optional[OptimizationConfig],
    use_raw_sq: bool,
) -> Tuple[float, float]:
    # (1) Use the same config for all metrics if one WinsorizationConfig was specified
    if isinstance(winsorization_config, WinsorizationConfig):
        return _quantiles_to_cutoffs(
            metric_name=metric_name,
            metric_values=metric_values,
            metric_config=winsorization_config,
        )

    # (2) If `winsorization_config` is a dict, use it if `metric_name` is a key,
    # and the corresponding value is a WinsorizationConfig.
    if isinstance(winsorization_config, dict) and metric_name in winsorization_config:
        metric_config = winsorization_config[metric_name]
        if not isinstance(metric_config, WinsorizationConfig):
            raise UserInputError(  # pragma: no cover
                "Expected winsorization config of type "
                f"`WinsorizationConfig` but got {metric_config} of type "
                f"{type(metric_config)} for metric {metric_name}."
            )
        return _quantiles_to_cutoffs(
            metric_name=metric_name,
            metric_values=metric_values,
            metric_config=metric_config,
        )

    # (3) For constraints and objectives that don't have a pre-specified config we
    # choose the cutoffs automatically using the optimization config (if supplied).
    # We ignore ScalarizedOutcomeConstraint and ScalarizedObjective for now, and
    # derelativize relative constraints if possible.

    # When no optimization config is available, return defaults.
    if modelbridge is None or optimization_config is None:
        return DEFAULT_CUTOFFS
    if any(oc.relative for oc in optimization_config.all_constraints):
        if not use_raw_sq:
            raise UnsupportedError(
                "Automatic winsorization doesn't support relative outcome constraints "
                "or objective thresholds when `derelativize_with_raw_status_quo` is "
                "not set to `True`."
            )
        optimization_config = derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config,
            modelbridge=modelbridge,
            observations=observations,
        )

    # Non-objective metrics - obtain cutoffs from outcome_constraints.
    if metric_name not in optimization_config.objective.metric_names:
        # Get all outcome constraints for `metric_name`` that aren't scalarized.
        return _obtain_cutoffs_from_outcome_constraints(
            optimization_config=optimization_config,
            metric_name=metric_name,
            metric_values=metric_values,
        )

    # Don't winsorize a ScalarizedObjective
    if isinstance(optimization_config.objective, ScalarizedObjective):
        warnings.warn(
            "Automatic winsorization isn't supported for ScalarizedObjective. "
            "Specify the winsorization settings manually if you want to "
            f"winsorize metric {metric_name}."
        )
        return DEFAULT_CUTOFFS

    # Single-objective
    if not optimization_config.is_moo_problem:
        return _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values,
            minimize=optimization_config.objective.minimize,
        )

    # Multi-objective
    return _get_auto_winsorization_cutoffs_multi_objective(
        optimization_config=optimization_config,
        metric_name=metric_name,
        metric_values=metric_values,
    )


def _get_auto_winsorization_cutoffs_multi_objective(
    optimization_config: OptimizationConfig,
    metric_name: str,
    metric_values: List[float],
) -> Tuple[float, float]:
    # We approach a multi-objective metric the same as output constraints. It may be
    # worth investigating setting the winsorization cutoffs based on the Pareto
    # frontier in the future.
    optimization_config = checked_cast(
        MultiObjectiveOptimizationConfig, optimization_config
    )
    objective_threshold = _get_objective_threshold_from_moo_config(
        optimization_config=optimization_config, metric_name=metric_name
    )
    if objective_threshold:
        return _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=metric_values,
            outcome_constraints=objective_threshold,
        )
    warnings.warn(
        "Automatic winsorization isn't supported for an objective in `MultiObjective` "
        "without objective thresholds. Specify the winsorization settings manually if "
        f"you want to winsorize metric {metric_name}."
    )
    return DEFAULT_CUTOFFS  # Don't winsorize if there is no threshold


def _obtain_cutoffs_from_outcome_constraints(
    optimization_config: OptimizationConfig,
    metric_name: str,
    metric_values: List[float],
) -> Tuple[float, float]:
    """Get all outcome constraints (non-scalarized) for a given metric."""
    # Check for scalarized outcome constraints for the given metric
    if any(
        isinstance(oc, ScalarizedOutcomeConstraint)
        and metric_name in [metric.name for metric in oc.metrics]
        for oc in optimization_config.outcome_constraints
    ):
        warnings.warn(
            "Automatic winsorization isn't supported for a "
            "`ScalarizedOutcomeConstraint`. Specify the winsorization settings "
            f"manually if you want to winsorize metric {metric_name}."
        )
    outcome_constraints = _get_non_scalarized_outcome_constraints(
        optimization_config=optimization_config, metric_name=metric_name
    )
    if outcome_constraints:
        return _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=metric_values,
            outcome_constraints=outcome_constraints,
        )
    return DEFAULT_CUTOFFS


def _get_non_scalarized_outcome_constraints(
    optimization_config: OptimizationConfig, metric_name: str
) -> List[OutcomeConstraint]:
    return [
        oc
        for oc in optimization_config.outcome_constraints
        if not isinstance(oc, ScalarizedOutcomeConstraint)
        and oc.metric.name == metric_name
    ]


def _get_objective_threshold_from_moo_config(
    optimization_config: MultiObjectiveOptimizationConfig, metric_name: str
) -> List[ObjectiveThreshold]:
    """Get the non-relative objective threshold for a given metric."""
    return [
        ot
        for ot in optimization_config.objective_thresholds
        if ot.metric.name == metric_name
    ]


def _get_tukey_cutoffs(Y: np.ndarray, lower: bool) -> float:
    """Compute winsorization cutoffs similarly to Tukey boxplots.

    See https://mathworld.wolfram.com/Box-and-WhiskerPlot.html for more details.
    """
    q1 = np.percentile(Y, q=25, interpolation="lower")
    q3 = np.percentile(Y, q=75, interpolation="higher")
    iqr = q3 - q1
    return q1 - 1.5 * iqr if lower else q3 + 1.5 * iqr


def _get_auto_winsorization_cutoffs_single_objective(
    metric_values: List[float], minimize: bool
) -> Tuple[float, float]:
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
    metric_values: List[float],
    outcome_constraints: Union[List[ObjectiveThreshold], List[OutcomeConstraint]],
) -> Tuple[float, float]:
    """Automatic winsorization to an outcome constraint.

    We need to be careful here so we don't make infeasible points feasible.
    While it is possible to winsorize from both ends, we only winsorize from the
    infeasible direction for now so the same method can be used for MOO. We rely on
    a heuristic similar to `_get_tukey_cutoffs`, but instead take the max of
    q3 and the bound when we have a LEQ constraint and the min of q1 and the bound
    with a GEQ constraint.
    """
    Y = np.array(metric_values)
    q1 = np.percentile(Y, q=25, interpolation="lower")
    q3 = np.percentile(Y, q=75, interpolation="higher")
    lower_cutoff, upper_cutoff = DEFAULT_CUTOFFS
    for oc in outcome_constraints:
        bnd = oc.bound
        if oc.op == ComparisonOp.LEQ:
            upper_cutoff = max(q3, bnd) + 1.5 * (max(q3, bnd) - q1)
        elif oc.op == ComparisonOp.GEQ:
            lower_cutoff = min(q1, bnd) - 1.5 * (q3 - min(q1, bnd))
        else:
            raise ValueError(  # pragma: no cover
                "Exected outcome_constraint to use operator LEQ or GEQ"
            )
    return lower_cutoff, upper_cutoff


def _quantiles_to_cutoffs(
    metric_name: str,
    metric_values: List[float],
    metric_config: WinsorizationConfig,
) -> Tuple[float, float]:
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
        raise ValueError(  # pragma: no cover
            f"Lower bound: {lower} was greater than the inverse of the "
            f"upper bound: {1 - upper} for metric {metric_name}. Decrease "
            f"one or both of `lower_quantile_margin` and "
            "`upper_quantile_margin`."
        )
    if lower == AUTO_WINS_QUANTILE:
        cutoff_l = _get_tukey_cutoffs(Y=Y, lower=True)
    elif lower == 0.0:  # Use the default cutoff if there is no winsorization
        cutoff_l = DEFAULT_CUTOFFS[0]
    else:
        cutoff_l = np.percentile(Y, lower * 100, interpolation="lower")

    if upper == AUTO_WINS_QUANTILE:
        cutoff_u = _get_tukey_cutoffs(Y=Y, lower=False)
    elif upper == 0.0:  # Use the default cutoff if there is no winsorization
        cutoff_u = DEFAULT_CUTOFFS[1]
    else:
        cutoff_u = np.percentile(Y, (1 - upper) * 100, interpolation="higher")

    cutoff_l = min(cutoff_l, bnd_l) if bnd_l is not None else cutoff_l
    cutoff_u = max(cutoff_u, bnd_u) if bnd_u is not None else cutoff_u
    return (cutoff_l, cutoff_u)


def _get_cutoffs_from_legacy_transform_config(
    metric_name: str,
    metric_values: List[float],
    transform_config: TConfig,
) -> Tuple[float, float]:
    winsorization_config = WinsorizationConfig()
    if "winsorization_lower" in transform_config:
        winsorization_lower = transform_config["winsorization_lower"]
        if isinstance(winsorization_lower, dict):
            if metric_name in winsorization_lower:
                winsorization_config.lower_quantile_margin = winsorization_lower[
                    metric_name
                ]
        elif isinstance(winsorization_lower, (int, float)):
            winsorization_config.lower_quantile_margin = winsorization_lower
    if "winsorization_upper" in transform_config:
        winsorization_upper = transform_config["winsorization_upper"]
        if isinstance(winsorization_upper, dict):
            if metric_name in winsorization_upper:
                winsorization_config.upper_quantile_margin = winsorization_upper[
                    metric_name
                ]
        elif isinstance(winsorization_upper, (int, float)):
            winsorization_config.upper_quantile_margin = winsorization_upper
    if "percentile_bounds" in transform_config:
        percentile_bounds = transform_config["percentile_bounds"]
        output_percentile_bounds = (None, None)
        if isinstance(percentile_bounds, dict):
            if metric_name in percentile_bounds:
                output_percentile_bounds = percentile_bounds[metric_name]
        elif isinstance(percentile_bounds, tuple):  # pragma: no cover
            output_percentile_bounds = percentile_bounds  # pragma: no cover
        if len(output_percentile_bounds) != 2 or not all(
            isinstance(pb, (int, float)) or pb is None
            for pb in output_percentile_bounds
        ):
            raise ValueError(
                f"Expected percentile_bounds for metric {metric_name} to be "
                f"of the form (l, u), got {output_percentile_bounds}."
            )
        winsorization_config.lower_boundary = output_percentile_bounds[0]
        winsorization_config.upper_boundary = output_percentile_bounds[1]
    return _quantiles_to_cutoffs(
        metric_name=metric_name,
        metric_values=metric_values,
        metric_config=winsorization_config,
    )


def _get_and_validate_use_raw_sq(config: TConfig) -> bool:
    use_raw_sq = config.get("derelativize_with_raw_status_quo", False)
    if isinstance(use_raw_sq, bool):
        return use_raw_sq
    raise UserInputError(
        f"`derelativize_with_raw_status_quo` must be a boolean. Got {use_raw_sq}."
    )
