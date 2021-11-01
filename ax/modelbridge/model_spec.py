#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
from ax.utils.common.kwargs import consolidate_kwargs, get_function_argument_names
from ax.utils.common.typeutils import not_none


@dataclass
class ModelSpec(Base):
    model_enum: ModelRegistryBase
    # Kwargs to pass into the `Model` + `ModelBridge` constructor,
    # (`ModelRegistryBase.__call__`).
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass into the `ModelBridge.gen`.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    # Fixed generation features to pass into the Model's `.gen` function.
    fixed_features: Optional[ObservationFeatures] = None

    # Fitted model, constructed using specified model_kwargs, Data on fit()
    _fitted_model: Optional[ModelBridge] = None

    @property
    def fitted_model(self) -> ModelBridge:
        """Returns the fitted Ax model, asserting fit() was called"""
        self._assert_fitted()
        return not_none(self._fitted_model)

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
        search_space = search_space or experiment.search_space
        optimization_config = optimization_config or experiment.optimization_config
        model_kwargs = {
            **(self.model_kwargs or {}),
            **model_kwargs,
        }
        self._fitted_model = self.model_enum(
            experiment=experiment,
            data=data,
            search_space=search_space,
            optimization_config=optimization_config,
            **model_kwargs,
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
