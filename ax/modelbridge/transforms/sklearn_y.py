from __future__ import annotations

from collections import defaultdict
from logging import Logger

from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import get_data, match_ci_width_truncated
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast_list
from pyre_extensions import assert_is_instance

from scipy import stats

from sklearn.base import (
    _fit_context,
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


logger: Logger = get_logger(__name__)


class LogWarpingTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {"offset": [float], "copy": ["boolean"]}

    def __init__(self, *, offset=1.5, copy=True):
        if offset <= 1:
            raise ValueError("offset must be greater than 1")
        self.offset = offset
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "labels_min_"):
            del self.labels_min_
            del self.labels_max_

    def fit(self, X, y=None, sample_weight=None):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        X = self._check_input(X, in_fit=True, check_shape=False)
        self.labels_min_ = np.nanmin(X, axis=0, keepdims=True)
        self.labels_max_ = np.nanmax(X, axis=0, keepdims=True)

        return self

    def transform(self, X, copy=None):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        X[:, :] = (self.labels_max_ - X) / (self.labels_max_ - self.labels_min_)
        finite_mask = np.isfinite(X)
        positive_mask = X >= 0
        X[:, :] = np.where(
            finite_mask & positive_mask,
            0.5 - (np.log1p(X * (self.offset - 1)) / np.log(self.offset)),
            X,
        )
        # This isn't in the vizier implementation but allows us to handle the
        # transformation of values of X that are larger than the maximum value
        # of X seen in the fitting data.
        X[:, :] = np.where(
            finite_mask & ~positive_mask,
            0.5 - X * (self.offset - 1) / np.log(self.offset),
            X,
        )

        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        above_max_mask = X > 0.5
        X[:, :] = np.where(
            ~above_max_mask,
            np.expm1(np.log(self.offset) * (0.5 - X)) / (self.offset - 1),
            X,
        )
        # This isn't in the vizier implementation but allows us to handle the
        # inverse transformation of values of X that are larger than the maximum
        # value of X seen in the fitting data.
        X[:, :] = np.where(
            above_max_mask,
            (0.5 - X) * np.log(self.offset) / (self.offset - 1),
            X,
        )

        X[:, :] = self.labels_max_ - X * (self.labels_max_ - self.labels_min_)
        return X

    def _check_input(self, X, in_fit, check_shape=False):
        X = self._validate_data(
            X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

        if check_shape and not X.shape[1] == self.labels_min_.shape[1]:
            n = self.labels_min_.shape[1]
            m = X.shape[1]
            raise ValueError(
                "Input data has a different number of features "
                f"than fitting data. Should have {n}, data has {m}"
            )

        return X

    def _more_tags(self):
        return {"allow_nan": True}


class InfeasibleTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {"offset": [float], "copy": ["boolean"]}

    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "warped_bad_value_"):
            del self.warped_bad_value_
            del self.shift_

    def fit(self, X, y=None, sample_weight=None):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y, sample_weight)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None, sample_weight=None):
        X = self._check_input(X, in_fit=True, check_shape=False)

        if np.isnan(X).all(axis=0).any():
            raise ValueError(
                "Cannot fit InfeasibleTransformer on all-NaN feature columns."
            )

        labels_range = np.nanmax(X, axis=0, keepdims=True) - np.nanmin(
            X, axis=0, keepdims=True
        )
        warped_bad_value = np.nanmin(X, axis=0, keepdims=True) - (
            0.5 * labels_range + 1
        )
        num_feasible = X.shape[0] - np.isnan(X).sum(axis=0)

        # Estimate the relative frequency of feasible points
        p_feasible = (0.5 + num_feasible) / (1 + X.shape[0])

        self.warped_bad_value_ = warped_bad_value
        self.shift_ = -np.nanmean(X, axis=0) * p_feasible - warped_bad_value * (
            1 - p_feasible
        )

        return self

    def transform(self, X, copy=None):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        X[:, :] = np.where(
            np.isnan(X),
            self.warped_bad_value_,
            X + self.shift_,
        )

        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        X[:, :] = X - self.shift_
        return X

    def _check_input(self, X, in_fit, check_shape=False):
        X = self._validate_data(
            X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

        if check_shape and not X.shape[1] == self.warped_bad_value_.shape[1]:
            n = self.warped_bad_value_.shape[1]
            m = X.shape[1]
            raise ValueError(
                "Input data has a different number of features "
                f"than fitting data. Should have {n}, data has {m}"
            )

        return X

    def _more_tags(self):
        return {"allow_nan": True}


class HalfRankTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {"copy": ["boolean"]}

    def __init__(self, *, copy=True):
        self.copy = copy

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        self._fit(X, y=y, force_transform=False)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        return self._fit(X, y, force_transform=True)

    def _get_std_above_median(self, unique_labels, median):
        good_half = unique_labels[unique_labels >= median]
        std = np.sqrt(((good_half - median) ** 2).mean())

        if std == 0:
            std = np.sqrt(((unique_labels - median) ** 2).mean())

        if np.isnan(std):
            std = np.abs(unique_labels - median).mean()

        return std

    def _fit(self, X, y=None, force_transform=False):
        X = self._check_input(X, in_fit=True)

        if not self.copy and not force_transform:  # if call from fit()
            X = X.copy()  # force copy so that fit does not change X inplace

        self.original_label_median_ = np.empty(X.shape[1])
        self.warped_labels_ = {}
        self.unique_labels_ = {}

        for i, col in enumerate(X.T):
            median = np.nanmedian(col)

            # Get finite values and their ranks for each batch
            is_finite_mask = np.isfinite(col)
            unique_labels, unique_indices = np.unique(
                col[is_finite_mask], return_index=True
            )

            # Calculate rank quantiles
            ranks = stats.rankdata(col, method="dense", nan_policy="omit")
            dedup_median_index = np.searchsorted(unique_labels, median)
            denominator = 2 * dedup_median_index + (
                unique_labels[dedup_median_index] == median
            )
            rank_quantile = (ranks - 0.5) / denominator

            above_median_std = self._get_std_above_median(unique_labels, median)

            # Apply transformation
            rank_ppf = stats.norm.ppf(rank_quantile) * above_median_std * np.sqrt(2.0)
            X[:, i] = np.where(
                col < median,
                rank_ppf + median,
                col,
            )

            # save intermediate values for untransform
            self.original_label_median_[i] = median
            self.unique_labels_[i] = unique_labels
            self.warped_labels_[i] = X[is_finite_mask, i][unique_indices]

        return X

    @staticmethod
    def _extrapolate(col, unique_labels, warped_labels):
        return warped_labels[0] - (
            (unique_labels[0] - col) / (unique_labels[-1] - unique_labels[0])
        ) * (warped_labels[-1] - warped_labels[0])

    @staticmethod
    def _extrapolate_inverse(col, unique_labels, warped_labels):
        return unique_labels[0] + (
            (col - warped_labels[0]) / (warped_labels[-1] - warped_labels[0])
        ) * (unique_labels[-1] - unique_labels[0])

    @staticmethod
    def _expand_values_and_mask(below_median_indices, col, mask, values):
        full_mask = np.full_like(col, False, dtype=bool)
        extrapolate_indices = below_median_indices[mask]
        full_mask[extrapolate_indices] = True

        full_values = np.zeros_like(col)
        full_values[full_mask] = values

        return full_mask, full_values

    def transform(self, X):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        for i, col in enumerate(X.T):
            median = self.original_label_median_[i]
            warped_labels: np.ndarray = self.warped_labels_[i]
            unique_labels: np.ndarray = self.unique_labels_[i]

            # Process values below median
            below_median = col < median
            if below_median.any():
                below_median_indices = np.where(below_median)[0]

                # 1) if the value is below the original minimum, we need to
                # extrapolate outside the range
                extrapolate_mask = col[below_median] < unique_labels.min()
                extrapolated_values = self._extrapolate(
                    col[below_median][extrapolate_mask], unique_labels, warped_labels
                )
                assert (extrapolated_values < warped_labels.min()).all()

                full_extrapolate_mask, full_extrapolate_values = (
                    self._expand_values_and_mask(
                        below_median_indices, col, extrapolate_mask, extrapolated_values
                    )
                )

                X[:, i] = np.where(
                    full_extrapolate_mask,
                    full_extrapolate_values,
                    col,
                )

                # 2) otherwise, find nearest original values and try to perform lookup
                original_idx = np.searchsorted(unique_labels, col[below_median])

                # Create indices for neighboring values
                left_idx = np.clip(original_idx - 1, a_min=0, a_max=None)
                right_idx = np.clip(
                    original_idx + 1, a_min=None, a_max=len(unique_labels)
                )

                # Gather neighboring values
                candidates = np.stack(
                    [
                        unique_labels[left_idx],
                        unique_labels[original_idx],
                        unique_labels[right_idx],
                    ],
                    axis=-1,
                )

                # Find nearest original values and perform lookup
                best_idx = np.argmin(
                    np.abs(candidates - col[below_median][:, None]), axis=-1
                )
                lookup_mask = (
                    np.isclose(
                        candidates[np.arange(len(best_idx)), best_idx],
                        col[below_median],
                    )
                    & ~extrapolate_mask
                )

                full_lookup_mask, full_lookup_values = self._expand_values_and_mask(
                    below_median_indices,
                    col,
                    lookup_mask,
                    warped_labels[original_idx[lookup_mask]],
                )

                X[:, i] = np.where(
                    full_lookup_mask,
                    full_lookup_values,
                    col,
                )

                # 3) otherwise linearly interpolate between the nearest original values
                interpolate_mask = ~(extrapolate_mask | lookup_mask)
                interpolate_labels = col[below_median][interpolate_mask]
                interpolate_idx = original_idx[interpolate_mask]

                lower_idx = interpolate_idx - 1
                upper_idx = interpolate_idx

                original_gap = unique_labels[upper_idx] - unique_labels[lower_idx]
                warped_gap = warped_labels[upper_idx] - warped_labels[lower_idx]

                interpolated_values = np.where(
                    original_gap > 0,
                    warped_labels[lower_idx]
                    + (interpolate_labels - unique_labels[lower_idx])
                    / original_gap
                    * warped_gap,
                    warped_labels[lower_idx],
                )

                full_interpolated_mask, full_interpolated_values = (
                    self._expand_values_and_mask(
                        below_median_indices, col, interpolate_mask, interpolated_values
                    )
                )

                X[:, i] = np.where(
                    full_interpolated_mask,
                    full_interpolated_values,
                    col,
                )

        return X

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = self._check_input(X, in_fit=False, check_shape=True)

        for i, col in enumerate(X.T):
            median = self.original_label_median_[i]
            warped_labels: np.ndarray = self.warped_labels_[i]
            unique_labels: np.ndarray = self.unique_labels_[i]

            # Process values below median
            below_median = col < median
            if below_median.any():
                below_median_indices = np.where(below_median)[0]

                # 1) if the value is below the original minimum, we need to
                # extrapolate outside the range
                extrapolate_mask = col[below_median] < warped_labels.min()
                extrapolated_values = self._extrapolate_inverse(
                    col[below_median][extrapolate_mask], unique_labels, warped_labels
                )
                assert (extrapolated_values < unique_labels.min()).all()

                full_extrapolate_mask, full_extrapolate_values = (
                    self._expand_values_and_mask(
                        below_median_indices, col, extrapolate_mask, extrapolated_values
                    )
                )

                X[:, i] = np.where(full_extrapolate_mask, full_extrapolate_values, col)

                # 2) otherwise, find nearest original values and try to perform lookup
                warped_idx = np.searchsorted(warped_labels, col[below_median])

                # Create indices for neighboring values
                left_idx = np.clip(warped_idx - 1, a_min=0, a_max=None)
                right_idx = np.clip(
                    warped_idx + 1, a_min=None, a_max=len(warped_labels)
                )

                # Gather neighboring values
                candidates = np.stack(
                    [
                        warped_labels[left_idx],
                        warped_labels[warped_idx],
                        warped_labels[right_idx],
                    ],
                    axis=-1,
                )

                # Find nearest original values and perform lookup
                best_idx = np.argmin(
                    np.abs(candidates - col[below_median][:, None]), axis=-1
                )

                lookup_mask = (
                    np.isclose(
                        candidates[np.arange(len(best_idx)), best_idx],
                        col[below_median],
                    )
                    & ~extrapolate_mask
                )

                full_lookup_mask, full_lookup_values = self._expand_values_and_mask(
                    below_median_indices,
                    col,
                    lookup_mask,
                    unique_labels[warped_idx[lookup_mask]],
                )

                X[:, i] = np.where(
                    full_lookup_mask,
                    full_lookup_values,
                    col,
                )

                # 3) otherwise linearly interpolate between the nearest original values
                interpolate_mask = ~(extrapolate_mask | lookup_mask)
                interpolate_labels = col[below_median][interpolate_mask]
                interpolate_idx = warped_idx[interpolate_mask]
                lower_idx = interpolate_idx - 1
                upper_idx = interpolate_idx

                original_gap = unique_labels[upper_idx] - unique_labels[lower_idx]
                warped_gap = warped_labels[upper_idx] - warped_labels[lower_idx]

                interpolated_values = np.where(
                    warped_gap > 0,
                    unique_labels[lower_idx]
                    + (interpolate_labels - warped_labels[lower_idx])
                    / warped_gap
                    * original_gap,
                    unique_labels[lower_idx],
                )

                full_interpolated_mask, full_interpolated_values = (
                    self._expand_values_and_mask(
                        below_median_indices, col, interpolate_mask, interpolated_values
                    )
                )

                X[:, i] = np.where(
                    full_interpolated_mask,
                    full_interpolated_values,
                    col,
                )

        return X

    def _check_input(self, X, in_fit, check_shape=False):
        X = self._validate_data(
            X,
            ensure_2d=True,
            dtype=FLOAT_DTYPES,
            force_writeable=True,
            copy=self.copy,
            force_all_finite="allow-nan",
            reset=in_fit,
        )

        if check_shape and not X.shape[1] == self.original_label_median_.shape[0]:
            n = self.original_label_median_.shape[0]
            m = X.shape[1]
            raise ValueError(
                f"Input data has a different number of features than fitting data. "
                f"Should have {n}, data has {m}."
            )

        return X

    def _more_tags(self):
        return {"allow_nan": True}


class SklearnTransform(Transform):
    """A transform that wraps a sklearn transformer."""

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the ``SklearnTransform`` transform.

        Args:
            search_space: The search space of the experiment. Unused.
            observations: A list of observations from the experiment.
            modelbridge: The `ModelBridge` within which the transform is used. Unused.
            config: A dictionary of options to control the behavior of the transform.
                Can contain the following keys:
                - "metrics": A list of metric names to apply the transform to. If
                    omitted, all metrics found in `observations` are transformed.
                - "transformer": A callable for the sklearn transformer to use.
                - "transformer_kwargs": A dictionary of keyword arguments to pass to
                    the sklearn transformer.
                - "clip_mean": A boolean indicating whether to clip the mean of the
                    transformed data to the bounds.
                - "match_ci_width": A boolean indicating whether to match the width of
                    the confidence interval of the transformed data to the width of the
                    confidence interval of the original data.
        """
        if observations is None or len(observations) == 0:
            raise DataRequiredError("SklearnTransform requires observations.")
        # pyre-fixme[9]: Can't annotate config["metrics"] properly.
        metric_names: list[str] | None = config.get("metrics", None) if config else None
        self.clip_mean: bool = (
            assert_is_instance(config.get("clip_mean", True), bool) if config else True
        )
        self.match_ci_width: bool = (
            assert_is_instance(config.get("match_ci_width", False), bool)
            if config
            else False
        )
        observation_data = [obs.data for obs in observations]
        Ys = get_data(observation_data=observation_data, metric_names=metric_names)
        self.metric_names: list[str] = list(Ys.keys())

        # pyre-fixme[4]: Attribute must be annotated.
        self.transforms = self._compute_sklearn_transforms(
            Ys=Ys,
            transformer=config["transformer"],
            transformer_kwargs=config.get("transformer_kwargs", {}),
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.inv_bounds = self._compute_inverse_bounds(self.transforms)

    @staticmethod
    def _compute_inverse_bounds(
        transforms: dict[str, TransformerMixin], **kwargs
    ) -> dict[str, tuple[float, float]]:
        inv_bounds = defaultdict()
        for k, _t in transforms.items():
            inv_bounds[k] = tuple(checked_cast_list(float, [-np.inf, np.inf]))
        return inv_bounds

    @staticmethod
    def _compute_sklearn_transforms(
        Ys: dict[str, list[float]],
        transformer: Callable[[dict[str, Any]], TransformerMixin],
        transformer_kwargs: dict[str, Any],
    ) -> dict[str, TransformerMixin]:
        """Compute sklearn transforms."""
        transforms = {}
        for k, ys in Ys.items():
            y = np.array(ys)[:, None]  # Need to unsqueeze the last dimension
            t = transformer(**transformer_kwargs).fit(y)
            transforms[k] = t
        return transforms

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_names:
                    transform = self.transforms[m].transform
                    if self.match_ci_width:
                        obsd.means[i], obsd.covariance[i, i] = match_ci_width_truncated(
                            mean=obsd.means[i],
                            variance=obsd.covariance[i, i],
                            transform=lambda y: transform(np.array(y, ndmin=2)),
                            lower_bound=-np.inf,
                            upper_bound=np.inf,
                        )
                    else:
                        # TODO: for sklearn transformers that would have known
                        # variance/covariance transforms we should work out an
                        # interface to use them.
                        obsd.means[i] = transform(np.array(obsd.means[i], ndmin=2))
        return observation_data

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_names:
                    transform = self.transforms[m].inverse_transform
                    if self.match_ci_width:
                        lower_bound, upper_bound = self.inv_bounds[m]
                        if not self.clip_mean and (
                            obsd.means[i] < lower_bound or obsd.means[i] > upper_bound
                        ):
                            raise ValueError(
                                "Can't untransform mean outside the bounds "
                                "without clipping"
                            )
                        obsd.means[i], obsd.covariance[i, i] = match_ci_width_truncated(
                            mean=obsd.means[i],
                            variance=obsd.covariance[i, i],
                            transform=lambda y: transform(np.array(y, ndmin=2)),
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            clip_mean=self.clip_mean,
                        )
                    else:
                        # TODO: for sklearn transformers that would have known
                        # variance/covariance inverse transforms we should work
                        # out an interface to use them.
                        obsd.means[i] = transform(np.array(obsd.means[i], ndmin=2))
        return observation_data

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                c_metric_names = [metric.name for metric in c.metrics]
                intersection = set(c_metric_names) & set(self.metric_names)
                if intersection:
                    raise NotImplementedError(
                        f"{self.__class__.__name__} cannot be used for metric(s) "
                        f"{intersection} that are part of a "
                        "ScalarizedOutcomeConstraint."
                    )
            elif c.metric.name in self.metric_names:
                if c.relative:
                    raise ValueError(
                        f"{self.__class__.__name__} cannot be applied to metric "
                        f"{c.metric.name} since it is subject to a relative constraint."
                    )
                else:
                    transform = self.transforms[c.metric.name].transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        for c in outcome_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                raise ValueError("ScalarizedOutcomeConstraint not supported here")
            elif c.metric.name in self.metric_names:
                if c.relative:
                    raise ValueError("Relative constraints not supported here.")
                else:
                    transform = self.transforms[c.metric.name].inverse_transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return outcome_constraints


class PowerTransformY(SklearnTransform):
    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        config["transformer"] = PowerTransformer
        config["transformer_kwargs"] = {"method": "yeo-johnson"}
        config["match_ci_width"] = True
        config["clip_mean"] = True
        super().__init__(search_space, observations, modelbridge, config)

    @property
    def power_transforms(self) -> dict[str, TransformerMixin]:
        """Getter for power_transforms that returns transforms."""
        return self.transforms

    @power_transforms.setter
    def power_transforms(self, value: dict[str, TransformerMixin]) -> None:
        """Setter for power_transforms that sets transforms."""
        self.transforms = value

    @staticmethod
    def _compute_inverse_bounds(
        transforms: dict[str, PowerTransformer],
        tol=1e-10,
        **kwargs,
    ) -> dict[str, tuple[float, float]]:
        """Computes the image of the transform so we can clip when we untransform.

        The inverse of the Yeo-Johnson transform is given by:
            if X >= 0 and lambda == 0:
                X = exp(X_trans) - 1
            elif X >= 0 and lambda != 0:
                X = (X_trans * lambda + 1) ** (1 / lambda) - 1
            elif X < 0 and lambda != 2:
                X = 1 - (-(2 - lambda) * X_trans + 1) ** (1 / (2 - lambda))
            elif X < 0 and lambda == 2:
                X = 1 - exp(-X_trans)

        We can break this down into three cases:
            lambda < 0:        X < -1 / lambda
            0 <= lambda <= 2:  X is unbounded
            lambda > 2:        X > 1 / (2 - lambda)

        Sklearn standardizes the transformed values to have mean zero and standard
        deviation 1, so we also need to account for this when we compute the bounds.
        """
        inv_bounds = defaultdict()
        for k, pt in transforms.items():
            if not isinstance(pt, PowerTransformer):
                raise ValueError(f"Unexpected transformer type: {type(pt)}")
            bounds = [-np.inf, np.inf]
            mu, sigma = pt._scaler.mean_.item(), pt._scaler.scale_.item()  # pyre-ignore
            lambda_ = pt.lambdas_.item()  # pyre-ignore
            if lambda_ < -1 * tol:
                bounds[1] = (-1.0 / lambda_ - mu) / sigma
            elif lambda_ > 2.0 + tol:
                bounds[0] = (1.0 / (2.0 - lambda_) - mu) / sigma
            inv_bounds[k] = tuple(checked_cast_list(float, bounds))
        return inv_bounds


class LogWarpingY(SklearnTransform):
    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        config["transformer"] = LogWarpingTransformer
        config["match_ci_width"] = True
        super().__init__(search_space, observations, modelbridge, config)


class InfeasibleY(SklearnTransform):
    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        config["transformer"] = InfeasibleTransformer
        config["match_ci_width"] = True
        super().__init__(search_space, observations, modelbridge, config)

    @staticmethod
    def _compute_inverse_bounds(
        transforms: dict[str, InfeasibleTransformer], **kwargs
    ) -> dict[str, tuple[float, float]]:
        inv_bounds = defaultdict()
        for k, it in transforms.items():
            if not isinstance(it, InfeasibleTransformer):
                raise ValueError(f"Unexpected transformer type: {type(it)}")
            # If we encounter a value that is lower than the warped bad value
            # that we assign to infeasible values then it makes sense to clip
            # to the warped bad value to avoid giving the idea that a value is
            # worse than being infeasible.
            inv_bounds[k] = tuple(
                checked_cast_list(float, [it.warped_bad_value_[0, 0], np.inf])
            )
        return inv_bounds


class HalfRankY(SklearnTransform):
    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        config["transformer"] = HalfRankTransformer
        super().__init__(search_space, observations, modelbridge, config)
