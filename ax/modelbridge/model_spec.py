#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import ModelRegistryBase
from ax.utils.common.base import Base
from ax.utils.common.kwargs import (
    consolidate_kwargs,
    get_function_argument_names,
    filter_kwargs,
)
from ax.utils.common.typeutils import not_none


TModelFactory = Callable[..., ModelBridge]


@dataclass
class ModelSpec(Base):
    model_enum: ModelRegistryBase
    # Kwargs to pass into the `Model` + `ModelBridge` constructors in
    # `ModelRegistryBase.__call__`.
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass to `ModelBridge.gen`.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    # Fixed generation features to pass into the Model's `.gen` function.
    fixed_features: Optional[ObservationFeatures] = None

    # Fitted model, constructed using specified `model_kwargs` and `Data`
    # on `ModelSpec.fit`
    _fitted_model: Optional[ModelBridge] = None

    @property
    def fitted_model(self) -> ModelBridge:
        """Returns the fitted Ax model, asserting fit() was called"""
        self._assert_fitted()
        return not_none(self._fitted_model)

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
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        **model_kwargs: Any,
    ) -> None:
        """Fits the specified model on the given experiment + data using the
        model kwargs set on the model spec, alongside any passed down as
        kwargs to this function (local kwargs take precedent)
        """
        self._fitted_model = self.model_enum(
            experiment=experiment,
            data=data,
            **self._prepare_model_fit_kwargs(
                experiment=experiment,
                search_space=search_space,
                optimization_config=optimization_config,
                **model_kwargs,
            ),
        )

    def update(self, experiment: Experiment, new_data: Data) -> None:
        """Updates the current fitted model on the given experiment + new data

        Model must have been fit prior to calling update()
        """
        raise NotImplementedError("update() is not supported yet")

    def gen(self, **model_gen_kwargs: Any) -> GeneratorRun:
        """Generates candidates from the fitted model, using the model gen
        kwargs set on the model spec, alongside any passed as kwargs
        to this function (local kwargs take precedent)


        Model must have been fit prior to calling gen()
        """
        fitted_model = self.fitted_model
        model_gen_kwargs = consolidate_kwargs(
            kwargs_iterable=[self.model_gen_kwargs, model_gen_kwargs],
            keywords=get_function_argument_names(fitted_model.gen),
        )
        return fitted_model.gen(
            **model_gen_kwargs,
            fixed_features=self.fixed_features,
        )

    def _assert_fitted(self) -> None:
        """Helper that verifies a model was fitted, raising an error if not"""
        if self._fitted_model is None:
            raise UserInputError("No fitted model found. Call fit() to generate one")

    def _prepare_model_fit_kwargs(
        self,
        experiment: Experiment,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Consolidate keyword arguments to ``Model`` and ``ModelBridge``
        constructors, plugging in search space and optimization config
        from the ``Experiment`` object if those are not explicitly specified.
        """
        return {
            "search_space": search_space or experiment.search_space,
            "optimization_config": optimization_config
            or experiment.optimization_config,
            **(self.model_kwargs or {}),
            **model_kwargs,
        }


@dataclass
class FactoryFunctionModelSpec(ModelSpec):
    factory_function: Optional[TModelFactory] = None
    # pyre-ignore[15]: `ModelSpec` has this as non-optional
    model_enum: Optional[ModelRegistryBase] = None

    def __post_init__(self) -> None:
        if self.model_enum is not None:
            raise UserInputError(
                "Use regular `ModelSpec` when it's possible to describe the "
                "model as `ModelRegistryBase` subclass enum member."
            )
        if self.factory_function is None:
            raise UserInputError(
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
        except Exception:
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
        self._fitted_model = factory_function(
            **filter_kwargs(
                factory_function,
                experiment=experiment,
                data=data,
                # Some factory functions (like `get_sobol`) require search space
                # instead of experiment.
                **self._prepare_model_fit_kwargs(
                    experiment=experiment,
                    search_space=search_space,
                    optimization_config=optimization_config,
                    **model_kwargs,
                ),
            )
        )
