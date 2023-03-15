#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.cross_validation import (
    compute_diagnostics,
    cross_validate,
    CVDiagnostics,
    CVResult,
)
from ax.modelbridge.registry import ModelRegistryBase
from ax.utils.common.base import Base
from ax.utils.common.kwargs import (
    consolidate_kwargs,
    filter_kwargs,
    get_function_argument_names,
)
from ax.utils.common.typeutils import not_none


TModelFactory = Callable[..., ModelBridge]


class ModelSpecJSONEncoder(json.JSONEncoder):
    """Generic encoder to avoid JSON errors in ModelSpec.__repr__"""

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def default(self, o: Any) -> str:
        return repr(o)  # pragma: no cover


@dataclass
class ModelSpec(Base):
    model_enum: ModelRegistryBase
    # Kwargs to pass into the `Model` + `ModelBridge` constructors in
    # `ModelRegistryBase.__call__`.
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass to `ModelBridge.gen`.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass to `cross_validate`.
    model_cv_kwargs: Optional[Dict[str, Any]] = None

    # Fitted model, constructed using specified `model_kwargs` and `Data`
    # on `ModelSpec.fit`
    _fitted_model: Optional[ModelBridge] = None

    # stored cross validation results set in cross validate
    _cv_results: Optional[List[CVResult]] = None

    # stored cross validation diagnostics set in cross validate
    _diagnostics: Optional[CVDiagnostics] = None

    def __post_init__(self) -> None:
        self.model_kwargs = self.model_kwargs or {}
        self.model_gen_kwargs = self.model_gen_kwargs or {}
        self.model_cv_kwargs = self.model_cv_kwargs or {}

    @property
    def fitted_model(self) -> ModelBridge:
        """Returns the fitted Ax model, asserting fit() was called"""
        self._assert_fitted()
        return not_none(self._fitted_model)

    @property
    def fixed_features(self) -> Optional[ObservationFeatures]:
        """
        Fixed generation features to pass into the Model's `.gen` function.
        """
        return (
            self.model_gen_kwargs.get("fixed_features")
            if self.model_gen_kwargs is not None
            else None
        )

    @fixed_features.setter
    def fixed_features(self, value: Optional[ObservationFeatures]) -> None:
        """
        Fixed generation features to pass into the Model's `.gen` function.
        """
        if self.model_gen_kwargs is None:
            self.model_gen_kwargs = {}  # pragma: no cover
        self.model_gen_kwargs["fixed_features"] = value

    @property
    def model_key(self) -> str:
        """Key string to identify the model used by this ``ModelSpec``."""
        # NOTE: In the future, might need to add more to model key to make
        # model specs with the same model (but different kwargs) easier to
        # distinguish from their key. Could also add separate property, just
        # `key` (for `ModelSpec.key`, which will be unique even between model
        # specs with same model type).
        return self.model_enum.value

    def fit(
        self,
        experiment: Experiment,
        data: Data,
        **model_kwargs: Any,
    ) -> None:
        """Fits the specified model on the given experiment + data using the
        model kwargs set on the model spec, alongside any passed down as
        kwargs to this function (local kwargs take precedent)
        """
        # unset any cross validation cache
        self._cv_results, self._diagnostics = None, None
        # NOTE: It's important to copy `self.model_kwargs` here to avoid actually
        # adding contents of `model_kwargs` passed to this method, to
        # `self.model_kwargs`.
        combined_model_kwargs = {**(self.model_kwargs or {}), **model_kwargs}
        self._fitted_model = self.model_enum(
            experiment=experiment,
            data=data,
            **combined_model_kwargs,
        )

    def cross_validate(
        self,
    ) -> Tuple[Optional[List[CVResult]], Optional[CVDiagnostics]]:
        """
        Call cross_validate, compute_diagnostics and cache the results
        If the model cannot be cross validated, warn and return None
        """
        if self._cv_results is not None and self._diagnostics is not None:
            return self._cv_results, self._diagnostics

        self._assert_fitted()
        try:
            self._cv_results = cross_validate(
                model=self.fitted_model,
                **(self.model_cv_kwargs or {}),
            )
        except NotImplementedError:
            warnings.warn(f"{self.model_enum.value} cannot be cross validated")
            return None, None

        self._diagnostics = compute_diagnostics(self._cv_results)
        return self._cv_results, self._diagnostics

    @property
    def cv_results(self) -> Optional[List[CVResult]]:
        """
        Cached CV results from `self.cross_validate()`
        if it has been successfully called
        """
        return self._cv_results

    @property
    def diagnostics(self) -> Optional[CVDiagnostics]:
        """
        Cached CV diagnostics from `self.cross_validate()`
        if it has been successfully called
        """
        return self._diagnostics

    def update(self, experiment: Experiment, new_data: Data) -> None:
        """Updates the current fitted model on the given experiment + new data

        Model must have been fit prior to calling update()
        """
        raise NotImplementedError("update() is not supported yet")

    def gen(self, **model_gen_kwargs: Any) -> GeneratorRun:
        """Generates candidates from the fitted model, using the model gen
        kwargs set on the model spec, alongside any passed as kwargs
        to this function (local kwargs take precedent)

        NOTE: Model must have been fit prior to calling gen()

        Args:
            n: Integer representing how many arms should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
        """
        fitted_model = self.fitted_model
        model_gen_kwargs = consolidate_kwargs(
            kwargs_iterable=[
                self.model_gen_kwargs,
                model_gen_kwargs,
            ],
            keywords=get_function_argument_names(fitted_model.gen),
        )
        return fitted_model.gen(**model_gen_kwargs)

    def copy(self) -> ModelSpec:
        """`ModelSpec` is both a spec and an object that performs actions.
        Copying is useful to avoid changes to a singleton model spec.
        """
        return self.__class__(  # pragma: no cover
            model_enum=self.model_enum,
            model_kwargs=deepcopy(self.model_kwargs),
            model_gen_kwargs=deepcopy(self.model_gen_kwargs),
            model_cv_kwargs=deepcopy(self.model_cv_kwargs),
        )

    def _assert_fitted(self) -> None:
        """Helper that verifies a model was fitted, raising an error if not"""
        if self._fitted_model is None:
            raise UserInputError("No fitted model found. Call fit() to generate one")

    def __repr__(self) -> str:
        model_kwargs = json.dumps(
            self.model_kwargs, sort_keys=True, cls=ModelSpecJSONEncoder
        )
        model_gen_kwargs = json.dumps(
            self.model_gen_kwargs, sort_keys=True, cls=ModelSpecJSONEncoder
        )
        model_cv_kwargs = json.dumps(
            self.model_cv_kwargs, sort_keys=True, cls=ModelSpecJSONEncoder
        )
        return (
            "ModelSpec("
            f"\tmodel_enum={self.model_enum.value},\n"
            f"\tmodel_kwargs={model_kwargs},\n"
            f"\tmodel_gen_kwargs={model_gen_kwargs},\n"
            f"\tmodel_cv_kwargs={model_cv_kwargs},\n"
            ")"
        )

    def __hash__(self) -> int:
        return hash(repr(self))  # pragma: no cover

    def __eq__(self, other: ModelSpec) -> bool:
        return repr(self) == repr(other)


@dataclass
class FactoryFunctionModelSpec(ModelSpec):
    factory_function: Optional[TModelFactory] = None
    # pyre-ignore[15]: `ModelSpec` has this as non-optional
    model_enum: Optional[ModelRegistryBase] = None

    def __post_init__(self) -> None:
        if self.model_enum is not None:
            raise UserInputError(  # pragma: no cover
                "Use regular `ModelSpec` when it's possible to describe the "
                "model as `ModelRegistryBase` subclass enum member."
            )
        if self.factory_function is None:
            raise UserInputError(  # pragma: no cover
                "Please specify a valid function returning a `ModelBridge` instance "
                "as the required `factory_function` argument to "
                "`FactoryFunctionModelSpec`."
            )
        warnings.warn(
            "Using a factory function to describe the model, so optimization state "
            "cannot be stored and optimization is not resumable if interrupted."
        )

    @property
    def model_key(self) -> str:
        """Key string to identify the model used by this ``ModelSpec``."""
        try:
            # `model` is defined via a factory function.
            return not_none(self.factory_function).__name__  # pyre-ignore[16]
        except Exception:  # pragma: no cover
            raise TypeError(  # pragma: no cover
                f"{self.factory_function} is not a valid function, cannot extract name."
            )

    def fit(
        self,
        experiment: Experiment,
        data: Data,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        **model_kwargs: Any,
    ) -> None:
        """Fits the specified model on the given experiment + data using the
        model kwargs set on the model spec, alongside any passed down as
        kwargs to this function (local kwargs take precedent)
        """
        factory_function = not_none(self.factory_function)
        all_kwargs = deepcopy((self.model_kwargs or {}))
        all_kwargs.update(model_kwargs)
        self._fitted_model = factory_function(
            # Factory functions do not have a unified signature; e.g. some factory
            # functions (like `get_sobol`) require search space instead of experiment.
            # Therefore, we filter kwargs to remove unnecessary ones and add additional
            # arguments like `search_space` and `optimization_config`.
            **filter_kwargs(
                factory_function,
                experiment=experiment,
                data=data,
                search_space=search_space or experiment.search_space,
                optimization_config=optimization_config
                or experiment.optimization_config,
                **all_kwargs,
            )
        )
