#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
DerivedMetric: A metric computed from other metrics.

``DerivedMetric`` is a base class for metrics whose values depend on other
metrics being fetched first.  The experiment's data-fetch loop uses
``isinstance(m, DerivedMetric)`` to guarantee that all base metric data is
attached to the cache before any derived metric's ``fetch_trial_data`` runs.

The base class implements a template-method ``fetch_trial_data`` that handles:

1. Resolving where each arm's input metric data lives
   (via ``_resolve_source_trial_indices``).
2. Looking up and validating cached base metric data.
3. Collecting per-arm input metric data into a
   ``{arm_name: DataFrame}`` dict, where each DataFrame contains
   only the rows for that arm's input metrics from the correct trial.
4. Optionally relativizing per-arm metric means w.r.t. the experiment's
   status quo arm (when ``relativize_inputs=True``).
5. Delegating to ``_compute_derived_values`` for subclass-specific computation.

Subclasses must override ``_compute_derived_values`` to define how the
collected per-arm data is transformed into derived metric data.
Subclasses that need cross-trial data lookup (e.g., when an arm's base
metric data lives in a different trial) should override
``_resolve_source_trial_indices``.

Concrete subclasses:

* ``ExpressionDerivedMetric`` (below) – computes values from a mathematical
  expression of other metrics (e.g. ``log(a) - log(b)``).

.. note:: **Transform compatibility.**
   Derived metrics are computed *before* any adapter transforms run.
   Transforms that modify metric values (e.g. ``Relativize``, ``Log``) will
   be applied to the already-computed derived value, **not** to its inputs
   individually.  This means a derived metric ``log(a) - log(b)`` followed
   by a ``Log`` transform would double-log the result.  Avoid using
   transforms that overlap with operations already baked into the derivation.
"""

from __future__ import annotations

from logging import Logger
from typing import Any, Callable, cast

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.exceptions.core import UserInputError
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from ax.utils.common.string_utils import sanitize_name, unsanitize_name
from pyre_extensions import none_throws
from sympy import lambdify, sympify
from sympy.core.expr import Expr
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol


logger: Logger = get_logger(__name__)


class DerivedMetric(Metric):
    """Base class for metrics that depend on other metrics.

    A ``DerivedMetric`` declares the names of metrics whose data must be
    available before this metric can be computed.  The experiment's two-phase
    fetch loop (see ``Experiment._lookup_or_fetch_trials_results``) separates
    derived metrics from base metrics and fetches base metrics first.

    Subclasses must override ``_compute_derived_values`` to define how the
    derived value is produced from collected per-arm metric values.

    Attributes:
        input_metric_names: Names of metrics that must be fetched first.
    """

    def __init__(
        self,
        name: str,
        input_metric_names: list[str],
        relativize_inputs: bool = False,
        as_percent: bool = True,
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if not input_metric_names:
            raise UserInputError(
                f"DerivedMetric '{name}' must declare at least one input "
                f"metric in input_metric_names."
            )
        super().__init__(
            name=name,
            lower_is_better=lower_is_better,
            properties=properties,
        )
        self._input_metric_names = input_metric_names
        self._relativize_inputs = relativize_inputs
        self._as_percent = as_percent

    @property
    def input_metric_names(self) -> list[str]:
        """Names of metrics that this metric depends on."""
        return self._input_metric_names

    @property
    def relativize_inputs(self) -> bool:
        """Whether to relativize input metric values w.r.t. status quo
        before passing to ``_compute_derived_values``."""
        return self._relativize_inputs

    @property
    def as_percent(self) -> bool:
        """Whether to express relativized values as percentage change.

        Only relevant when ``relativize_inputs=True``.  When ``True``,
        a 50% improvement is represented as ``50.0``; when ``False``, as
        ``0.5``.
        """
        return self._as_percent

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup_metric_values_for_arm(
        arm_df: pd.DataFrame,
        metric_name: str,
    ) -> pd.DataFrame:
        """Look up rows for *metric_name* by ``metric_name`` or
        ``metric_signature`` column."""
        return arm_df[
            (arm_df["metric_name"] == metric_name)
            | (arm_df["metric_signature"] == metric_name)
        ]

    @staticmethod
    def _extract_means(arm_df: pd.DataFrame) -> dict[str, float]:
        """Extract ``{metric_name: mean}`` from a per-arm input metric DF."""
        means: dict[str, float] = {}
        for _, row in arm_df.iterrows():
            means[row["metric_name"]] = float(row["mean"])
        return means

    # ------------------------------------------------------------------
    # Template method: fetch_trial_data
    # ------------------------------------------------------------------

    def _resolve_source_trial_indices(
        self,
        trial: BaseTrial,
    ) -> dict[str, tuple[int, str]] | None:
        """Resolve where each arm's input metric data lives.

        Returns ``None`` to indicate that all arms' data comes from the
        current trial (the common case).  Subclasses that need cross-trial
        data lookup should override this to return a mapping of::

            {arm_name: (source_trial_index, source_arm_name)}

        where ``source_trial_index`` is the trial containing the base
        metric data and ``source_arm_name`` is the arm name in that trial
        (which may differ from ``arm_name`` if the arm was renamed).
        """
        return None

    def _lookup_base_data(
        self,
        trial: BaseTrial,
        source_trial_indices: dict[str, tuple[int, str]] | None,
    ) -> MetricFetchResult:
        """Look up cached base metric data needed for this derived metric.

        When ``source_trial_indices`` is ``None``, looks up data from the
        current trial only.  When provided, looks up data from the union
        of the current trial and all source trials.
        """
        try:
            if source_trial_indices is None:
                cached_data = trial.lookup_data()
                # When relativize_inputs is enabled, the SQ arm's data
                # may live in a different trial.  Widen the lookup to
                # include experiment-wide data so _relativize_arm_data
                # can find SQ values.
                if self._relativize_inputs:
                    sq = trial.experiment.status_quo
                    sq_name = sq.name if sq is not None else None
                    sq_in_trial = any(a.name == sq_name for a in trial.arms)
                    if not sq_in_trial:
                        cached_data = trial.experiment.lookup_data()
            else:
                trial_indices = {trial.index} | {
                    idx for idx, _ in source_trial_indices.values()
                }
                cached_data = trial.experiment.lookup_data(
                    trial_indices=trial_indices,
                )
        except Exception as e:
            return Err(
                MetricFetchE(
                    message=(
                        f"Failed to look up cached data for "
                        f"DerivedMetric '{self.name}' "
                        f"in trial {trial.index}: {e}"
                    ),
                    exception=e,
                )
            )

        if cached_data.empty:
            return Err(
                MetricFetchE(
                    message=(
                        f"Cannot compute DerivedMetric '{self.name}': "
                        f"no cached data available for trial {trial.index}."
                    ),
                    exception=None,
                )
            )

        return Ok(value=cached_data)

    def _validate_input_metrics_present(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Return input metric names not found in the cached DataFrame.

        Checks both ``metric_name`` and ``metric_signature`` columns.
        """
        available: set[str] = set(df["metric_name"].unique())
        if "metric_signature" in df.columns:
            available |= set(df["metric_signature"].unique())
        return [m for m in self._input_metric_names if m not in available]

    def _collect_arm_data(
        self,
        trial: BaseTrial,
        df: pd.DataFrame,
        source_trial_indices: dict[str, tuple[int, str]] | None,
    ) -> dict[str, pd.DataFrame] | MetricFetchE:
        """Collect input metric data for every arm in the trial.

        Returns ``{arm_name: DataFrame}`` on success, where each DataFrame
        contains only the rows for that arm's input metrics (filtered by
        ``metric_name`` or ``metric_signature``).  Returns a
        ``MetricFetchE`` on failure (missing metrics for any arm).

        When ``source_trial_indices`` is provided, each arm's data is
        looked up from the specified source trial and arm name.  Otherwise,
        data is looked up from the current trial using each arm's own name.
        """
        arm_data: dict[str, pd.DataFrame] = {}

        for arm in trial.arms:
            if source_trial_indices is not None and arm.name in source_trial_indices:
                src_trial_idx, src_arm_name = source_trial_indices[arm.name]
                arm_df = df[
                    (df["trial_index"] == src_trial_idx)
                    & (df["arm_name"] == src_arm_name)
                ]
            else:
                arm_df = df[
                    (df["trial_index"] == trial.index) & (df["arm_name"] == arm.name)
                ]

            # Filter to only input metric rows and validate presence.
            per_metric: list[pd.DataFrame] = []
            missing: list[str] = []
            for metric_name in self._input_metric_names:
                rows = self._lookup_metric_values_for_arm(arm_df, metric_name)
                if rows.empty:
                    missing.append(metric_name)
                else:
                    per_metric.append(rows)

            if missing:
                return MetricFetchE(
                    message=(
                        f"Cannot compute DerivedMetric '{self.name}' for arm "
                        f"'{arm.name}' in trial {trial.index}: "
                        f"missing input metrics: {missing}."
                    ),
                    exception=None,
                )
            arm_data[arm.name] = pd.concat(per_metric, ignore_index=True)

        return arm_data

    def _relativize_arm_data(
        self,
        trial: BaseTrial,
        arm_data: dict[str, pd.DataFrame],
        cached_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame] | MetricFetchE:
        """Optionally relativize per-arm metric values w.r.t. status quo.

        Each arm is relativized independently against the status quo data
        from that arm's own source trial.  This handles non-stationarity:
        the same SQ arm may have different metric values across trials, and
        each arm should be compared to *its* trial's SQ baseline.

        When the SQ data is not found in the arm's source trial (e.g., SQ
        was observed in a separate baseline trial), falls back to the
        latest trial that has SQ data.

        Delegates to ``Data.relativize``, which uses the delta method to
        properly transform both means and SEMs.

        When ``relativize_inputs`` is ``False``, returns ``arm_data``
        unchanged.  When ``True``, the status quo arm is included with
        zero-valued inputs so the expression can be evaluated on it
        (e.g., ``exp(0)=1``).
        """
        if not self._relativize_inputs:
            return arm_data

        status_quo = trial.experiment.status_quo
        if status_quo is None:
            return MetricFetchE(
                message=(
                    f"Cannot relativize inputs for DerivedMetric "
                    f"'{self.name}': experiment has no status quo arm."
                ),
                exception=None,
            )

        sq_name = status_quo.name

        # Relativize each arm independently against the SQ from the same
        # source trial.  This correctly handles cross-trial lookup where
        # different arms come from different trials with potentially
        # different SQ metric values (non-stationarity).
        relativized: dict[str, pd.DataFrame] = {}
        for arm_name, arm_df in arm_data.items():
            # SQ relativized against itself is trivially zero for all inputs.
            # Include it so _compute_derived_values can evaluate the expression
            # on zeros (e.g., exp(0)=1, a+b=0).
            if arm_name == sq_name:
                sq_rel_rows: list[dict[str, Any]] = []
                status_quo_trial_index = int(arm_df["trial_index"].iloc[0])
                for metric_name in self._input_metric_names:
                    sq_rel_rows.append(
                        {
                            "trial_index": status_quo_trial_index,
                            "arm_name": sq_name,
                            "metric_name": metric_name,
                            "metric_signature": metric_name,
                            "mean": 0.0,
                            "sem": 0.0,
                        }
                    )
                relativized[sq_name] = pd.DataFrame(sq_rel_rows)
                continue

            # Determine this arm's source trial_index from its data.
            arm_trial_idx = int(arm_df["trial_index"].iloc[0])

            # Find SQ data from the same source trial.  If SQ data is not
            # in that trial (e.g., the SQ was observed in a separate
            # baseline trial), fall back to the latest trial that has it.
            sq_rows = cached_df[
                (cached_df["arm_name"] == sq_name)
                & (cached_df["trial_index"] == arm_trial_idx)
            ]
            if sq_rows.empty:
                # Fallback: find SQ data from any trial.
                all_sq_rows = cached_df[cached_df["arm_name"] == sq_name]
                if all_sq_rows.empty:
                    return MetricFetchE(
                        message=(
                            f"Cannot relativize inputs for DerivedMetric "
                            f"'{self.name}' in trial {trial.index}: status "
                            f"quo arm '{sq_name}' not found in cached data."
                        ),
                        exception=None,
                    )
                # Use the latest trial's SQ data.
                latest_sq_idx = int(all_sq_rows["trial_index"].max())
                sq_rows = all_sq_rows[all_sq_rows["trial_index"] == latest_sq_idx]
            # Filter SQ rows to input metrics only.
            sq_input_rows: list[pd.DataFrame] = []
            for metric_name in self._input_metric_names:
                rows = self._lookup_metric_values_for_arm(sq_rows, metric_name)
                if not rows.empty:
                    sq_input_rows.append(rows)
            if not sq_input_rows:
                return MetricFetchE(
                    message=(
                        f"Cannot relativize inputs for DerivedMetric "
                        f"'{self.name}' in trial {trial.index}: no input "
                        f"metric data found for status quo arm '{sq_name}' "
                        f"in source trial {arm_trial_idx}."
                    ),
                    exception=None,
                )

            # Build a mini-DataFrame with this arm + its SQ, then
            # relativize.  Both must share the same trial_index so
            # Data.relativize groups them together.  The SQ rows may
            # come from a different trial (fallback case), so normalize
            # their trial_index to match the arm's.
            sq_df = pd.concat(sq_input_rows, ignore_index=True)
            if int(sq_df["trial_index"].iloc[0]) != arm_trial_idx:
                sq_df = sq_df.copy()
                sq_df["trial_index"] = arm_trial_idx
            combined = pd.concat([arm_df, sq_df], ignore_index=True)

            try:
                rel_data = Data(df=combined).relativize(
                    status_quo_name=sq_name,
                    as_percent=self._as_percent,
                    include_sq=False,
                    control_as_constant=True,
                )
            except Exception as e:
                return MetricFetchE(
                    message=(
                        f"Error relativizing inputs for DerivedMetric "
                        f"'{self.name}' in trial {trial.index}: {e}"
                    ),
                    exception=e,
                )

            rel_rows = rel_data.df[rel_data.df["arm_name"] == arm_name]
            if not rel_rows.empty:
                relativized[arm_name] = rel_rows.reset_index(drop=True)

        return relativized

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        """Template method: look up data, collect values, delegate to subclass.

        Subclasses should not override this method.  Instead, override
        ``_compute_derived_values`` (and optionally
        ``_resolve_source_trial_indices``).
        """
        # Step 1: Resolve source trial indices (subclass hook).
        source_trial_indices = self._resolve_source_trial_indices(trial)

        # Step 2: Look up cached base metric data.
        lookup_result = self._lookup_base_data(trial, source_trial_indices)
        if isinstance(lookup_result, Err):
            return lookup_result
        cached_data: Data = none_throws(lookup_result.ok)

        df = cached_data.df

        # Step 3: Validate that input metrics exist in the cached data.
        missing_global = self._validate_input_metrics_present(df)
        if missing_global:
            return Err(
                MetricFetchE(
                    message=(
                        f"Cannot compute DerivedMetric '{self.name}' for "
                        f"trial {trial.index}: input metrics not found in "
                        f"cached data: {missing_global}."
                    ),
                    exception=None,
                )
            )

        # Step 4: Collect per-arm input metric data.
        arm_data_result = self._collect_arm_data(
            trial,
            df,
            source_trial_indices,
        )
        if isinstance(arm_data_result, MetricFetchE):
            return Err(arm_data_result)

        # Step 5: Optionally relativize arm data w.r.t. status quo.
        arm_data_result = self._relativize_arm_data(trial, arm_data_result, df)
        if isinstance(arm_data_result, MetricFetchE):
            return Err(arm_data_result)

        # After relativization, arm_data may be empty (e.g., a trial with
        # no arms).  Return empty data, not an error.
        if not arm_data_result:
            return Ok(value=Data())

        # Step 6: Delegate to subclass for transformation.
        return self._compute_derived_values(
            trial=trial,
            arm_data=arm_data_result,
        )

    def _compute_derived_values(
        self,
        trial: BaseTrial,
        arm_data: dict[str, pd.DataFrame],
    ) -> MetricFetchResult:
        """Subclass hook: compute derived metric values from collected data.

        Args:
            trial: The trial being fetched.
            arm_data: ``{arm_name: DataFrame}`` for all arms in the trial.
                Each DataFrame contains only the input metric rows for
                that arm (columns include ``metric_name``, ``mean``,
                ``sem``, etc.).  All input metrics are guaranteed present.
                Use ``_lookup_metric_values_for_arm`` to extract rows for
                a specific metric.

        Returns:
            ``Ok(Data)`` with derived metric rows, or ``Err`` on failure.
        """
        raise NotImplementedError(
            f"DerivedMetric subclass {type(self).__name__} must implement "
            f"_compute_derived_values."
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @property
    def summary_dict(self) -> dict[str, Any]:
        """Fields of this metric's configuration that will appear
        in the ``Summary`` analysis table.
        """
        return {
            **super().summary_dict,
            "input_metric_names": self._input_metric_names,
            "relativize_inputs": self._relativize_inputs,
            "as_percent": self._as_percent,
        }


class ExpressionDerivedMetric(DerivedMetric):
    """A metric computed from a mathematical expression of other metrics.

    The expression is parsed using sympy (consistent with other expression
    parsing in Ax) and compiled via ``lambdify`` for fast numeric evaluation.
    It may reference:

    * Input metric names as variables
    * Mathematical operators: ``+``, ``-``, ``*``, ``/``, ``**``
    * Any function available in Python's ``math`` module (e.g. ``log``,
      ``exp``, ``sqrt``, ``abs``, ``sin``, ``cos``, ``asin``, ``pow``, etc.)
    * Numeric constants

    Attributes:
        expression_str: The mathematical expression string.

    Example::

        >>> log_ratio = ExpressionDerivedMetric(
        ...     name="log_ratio",
        ...     input_metric_names=["metric_a", "metric_b"],
        ...     expression_str="log(metric_a) - log(metric_b)",
        ... )
    """

    def __init__(
        self,
        name: str,
        input_metric_names: list[str],
        expression_str: str,
        relativize_inputs: bool = False,
        as_percent: bool = True,
        lower_is_better: bool | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_metric_names=input_metric_names,
            relativize_inputs=relativize_inputs,
            as_percent=as_percent,
            lower_is_better=lower_is_better,
            properties=properties,
        )
        self._expression_str = expression_str

        # Parse & validate once; cache the compiled evaluator for reuse.
        # sanitize_name handles metric names with dots, slashes, etc.
        # (consistent with DerivedParameter's expression parsing).
        try:
            self._sympy_expr: Expr = sympify(  # pyre-ignore[8]
                sanitize_name(self._expression_str),
            )
        except Exception as e:
            raise UserInputError(
                f"Invalid expression in ExpressionDerivedMetric "
                f"'{self.name}': {self._expression_str}. Error: {e}"
            ) from e
        self._validate_expression()
        # _sympy_symbols are the sanitized names used by sympy/lambdify.
        # _symbols are the original (unsanitized) metric names used for
        # looking up values at evaluation time.
        # Cast free_symbols to set[Symbol] since Pyre stubs use Basic
        # pyre-fixme[16]: Pyre cannot infer that free_symbols contains Symbol
        free_syms = cast(set[Symbol], self._sympy_expr.free_symbols)
        sympy_symbols: list[str] = sorted(s.name for s in free_syms)
        self._symbols: list[str] = [unsanitize_name(s) for s in sympy_symbols]
        self._evaluator: Callable[..., float] = lambdify(
            sympy_symbols, self._sympy_expr, modules="math"
        )

    @property
    def expression_str(self) -> str:
        """The expression string defining the derivation."""
        return self._expression_str

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_expression(self) -> None:
        """Validate that the parsed expression is a numeric expression with
        only declared input metrics as free symbols."""
        if isinstance(self._sympy_expr, Relational):
            raise UserInputError(
                "Comparison operators are not allowed in "
                "ExpressionDerivedMetric expressions. "
                "Use outcome constraints for comparisons."
            )

        # Reject undeclared variable names.
        # Cast free_symbols to set[Symbol] since Pyre stubs use Basic
        # pyre-fixme[16]: Pyre cannot infer that free_symbols contains Symbol
        free_syms = cast(set[Symbol], self._sympy_expr.free_symbols)
        referenced_names = {unsanitize_name(s.name) for s in free_syms}
        input_metric_set = set(self._input_metric_names)

        unknown_names = referenced_names - input_metric_set
        if unknown_names:
            raise UserInputError(
                f"Expression for ExpressionDerivedMetric '{self.name}' references "
                f"unknown names: {unknown_names}. Allowed metric names: "
                f"{input_metric_set}."
            )

        unused_inputs = input_metric_set - referenced_names
        if unused_inputs:
            logger.warning(
                f"ExpressionDerivedMetric '{self.name}' declares input metrics "
                f"that are not used in the expression: {unused_inputs}."
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_expression(self, metric_values: dict[str, float]) -> float:
        """Evaluate the expression with the given metric values."""
        args = [metric_values[s] for s in self._symbols]
        return float(self._evaluator(*args))

    # ------------------------------------------------------------------
    # Core computation (subclass hook)
    # ------------------------------------------------------------------

    def _compute_derived_values(
        self,
        trial: BaseTrial,
        arm_data: dict[str, pd.DataFrame],
    ) -> MetricFetchResult:
        """Evaluate the expression for each arm using pre-collected data.

        When ``relativize_inputs`` is ``True``, the base class has already
        relativized the ``mean`` values.  The status quo arm is included
        with zero-valued inputs.
        """
        result_rows: list[dict[str, Any]] = []

        for arm_name, arm_df in arm_data.items():
            try:
                metric_values = self._extract_means(arm_df)
                derived_value = self._evaluate_expression(metric_values)
            except Exception as e:
                return Err(
                    MetricFetchE(
                        message=(
                            f"Error evaluating ExpressionDerivedMetric "
                            f"'{self.name}' for arm '{arm_name}' "
                            f"in trial {trial.index}: {e}"
                        ),
                        exception=e,
                    )
                )

            result_rows.append(
                {
                    "trial_index": trial.index,
                    "arm_name": arm_name,
                    "metric_name": self.name,
                    "metric_signature": self.signature,
                    "mean": derived_value,
                    "sem": float("nan"),
                }
            )

        return Ok(value=Data(df=pd.DataFrame(result_rows)))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"name='{self.name}'",
            f"expression='{self._expression_str}'",
        ]
        if self._relativize_inputs:
            parts.append("relativize_inputs=True")
        return f"ExpressionDerivedMetric({', '.join(parts)})"

    @property
    def summary_dict(self) -> dict[str, Any]:
        """Fields of this metric's configuration that will appear
        in the ``Summary`` analysis table.
        """
        return {
            **super().summary_dict,
            "expression_str": self._expression_str,
        }
