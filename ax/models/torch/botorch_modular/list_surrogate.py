#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.types import TCandidateMetadata
from ax.models.torch.botorch_modular.surrogate import NOT_YET_FIT_MSG, Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none
from botorch.models.model import Model, TrainingData
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import Kernel
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


class ListSurrogate(Surrogate):
    """Special type of `Surrogate` that wraps a set of submodels into
    `ModelListGP` under the hood for multi-outcome or multi-task
    models.

    Args:
        botorch_model_class_per_outcome: Mapping from metric name to
            BoTorch model class that should be used as surrogate model for
            that metric.
        submodel_options_per_outcome: Optional mapping from metric name to
            dictionary of kwargs for the submodel for that outcome.
        mll_class: `MarginalLogLikelihood` class to use for model-fitting.
    """

    botorch_model_class_per_outcome: Dict[str, Type[Model]]
    mll_class: Type[MarginalLogLikelihood]
    kernel_class: Optional[Type[Kernel]] = None
    _training_data_per_outcome: Optional[Dict[str, TrainingData]] = None
    _model: Optional[Model] = None
    # Special setting for surrogates instantiated via `Surrogate.from_BoTorch`,
    # to avoid re-constructing the underlying BoTorch model on `Surrogate.fit`
    # when set to `False`.
    _should_reconstruct: bool = True

    def __init__(
        self,
        botorch_model_class_per_outcome: Dict[str, Type[Model]],
        submodel_options_per_outcome: Optional[Dict[str, Dict[str, Any]]] = None,
        mll_class: Type[MarginalLogLikelihood] = SumMarginalLogLikelihood,
    ) -> None:
        self.botorch_model_class_per_outcome = botorch_model_class_per_outcome
        self.submodel_options_per_outcome = submodel_options_per_outcome
        super().__init__(botorch_model_class=ModelListGP, mll_class=mll_class)

    @property
    def training_data(self) -> TrainingData:
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement `training_data`, "
            "use `training_data_per_outcome`."
        )

    @property
    def training_data_per_outcome(self) -> Dict[str, TrainingData]:
        if self._training_data_per_outcome is None:
            raise ValueError(NOT_YET_FIT_MSG)
        return not_none(self._training_data_per_outcome)

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.training_data_per_outcome.values())).X.dtype

    @property
    def device(self) -> torch.device:
        return next(iter(self.training_data_per_outcome.values())).X.device

    # pyre-ignore[14]: `construct` takes in list of training data in list surrogate,
    # whereas it takes just a single training data in base surrogate.
    def construct(self, training_data: List[TrainingData], **kwargs: Any) -> None:
        """Constructs the underlying BoTorch `Model` using the training data.

        Args:
            training_data: List of `TrainingData` for the submodels of `ModelListGP`.
                Each training data is for one outcome, and the order of outcomes
                should match the order of metrics in `metric_names` argument.
            **kwargs: Keyword arguments, accepts:
                - `metric_names` (required): Names of metrics, in the same order
                as training data (so if training data is `[tr_A, tr_B]`, the metrics
                would be `["A" and "B"]`). These are used to match training data
                with correct submodels of `ModelListGP`,
                - `fidelity_features`: Indices of columns in X that represent
                fidelity,
                - `task_features`: Indices of columns in X that represent tasks.
        """
        metric_names = kwargs.get(Keys.METRIC_NAMES)
        fidelity_features = kwargs.get(Keys.FIDELITY_FEATURES, [])
        task_features = kwargs.get(Keys.TASK_FEATURES, [])
        if metric_names is None:
            raise ValueError("Metric names are required.")

        self._training_data_per_outcome = {
            metric_name: tr for metric_name, tr in zip(metric_names, training_data)
        }
        submodel_options = self.submodel_options_per_outcome or {}
        submodels = []

        for metric_name, model_cls in self.botorch_model_class_per_outcome.items():
            if metric_name not in self.training_data_per_outcome:
                continue  # pragma: no cover
            tr = self.training_data_per_outcome[metric_name]
            formatted_model_inputs = model_cls.construct_inputs(
                training_data=tr,
                fidelity_features=fidelity_features,
                task_features=task_features,
            )
            kwargs = submodel_options.get(metric_name, {})
            # pyre-ignore[45]: Py raises informative msg if `model_cls` abstract.
            submodels.append(model_cls(**formatted_model_inputs, **kwargs))
        self._model = ModelListGP(*submodels)

    # pyre-ignore[14]: `fit` takes in list of training data in list surrogate,
    # whereas it takes just a single training data in base surrogate.
    def fit(
        self,
        training_data: List[TrainingData],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        target_fidelities: Optional[Dict[int, float]] = None,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        refit: bool = True,
    ) -> None:
        super().fit(
            # pyre-ignore[6]: `Surrogate.fit` expects single training data
            # and in `ListSurrogate` we use a list of training data.
            training_data=training_data,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            metric_names=metric_names,
            fidelity_features=fidelity_features,
            target_fidelities=target_fidelities,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
        )
