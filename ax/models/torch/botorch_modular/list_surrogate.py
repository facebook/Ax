#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import torch
from ax.models.torch.botorch_modular.surrogate import NOT_YET_FIT_MSG, Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from botorch.models.model import Model, TrainingData
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import Kernel
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


logger = get_logger(__name__)


class ListSurrogate(Surrogate):
    """Special type of ``Surrogate`` that wraps a set of submodels into
    ``ModelListGP`` under the hood for multi-outcome or multi-task
    models.

    Args:
        botorch_submodel_class_per_outcome: Mapping from metric name to
            BoTorch model class that should be used as surrogate model for
            that metric. Use instead of ``botorch_submodel_class``.
        botorch_submodel_class: BoTorch ``Model`` class, shortcut for when
            all submodels of this surrogate's underlying ``ModelListGP`` are
            of the same type.
            Use instead of ``botorch_submodel_class_per_outcome``.
        submodel_options_per_outcome: Optional mapping from metric name to
            dictionary of kwargs for the submodel for that outcome.
        submodel_options: Optional dictionary of kwargs, shared between all
            submodels.
            NOTE: kwargs for submodel are ``submodel_options`` (shared) +
            ``submodel_outions_per_outcome[submodel_outcome]`` (individual).
        mll_class: ``MarginalLogLikelihood`` class to use for model-fitting.
    """

    botorch_submodel_class_per_outcome: Dict[str, Type[Model]]
    botorch_submodel_class: Optional[Type[Model]]
    submodel_options_per_outcome: Dict[str, Dict[str, Any]]
    submodel_options: Dict[str, Any]
    mll_class: Type[MarginalLogLikelihood]
    kernel_class: Optional[Type[Kernel]] = None

    _training_data_per_outcome: Optional[Dict[str, TrainingData]] = None
    _model: Optional[Model] = None
    # Special setting for surrogates instantiated via `Surrogate.from_botorch`,
    # to avoid re-constructing the underlying BoTorch model on `Surrogate.fit`
    # when set to `False`.
    _should_reconstruct: bool = True

    def __init__(
        self,
        botorch_submodel_class_per_outcome: Optional[Dict[str, Type[Model]]] = None,
        botorch_submodel_class: Optional[Type[Model]] = None,
        submodel_options_per_outcome: Optional[Dict[str, Dict[str, Any]]] = None,
        submodel_options: Optional[Dict[str, Any]] = None,
        mll_class: Type[MarginalLogLikelihood] = SumMarginalLogLikelihood,
    ) -> None:
        if not bool(botorch_submodel_class_per_outcome) ^ bool(botorch_submodel_class):
            raise ValueError(  # pragma: no cover
                "Please specify either `botorch_submodel_class_per_outcome` or "
                "`botorch_model_class`. In the latter case, the same submodel "
                "class will be used for all outcomes."
            )
        self.botorch_submodel_class_per_outcome = (
            botorch_submodel_class_per_outcome or {}
        )
        self.botorch_submodel_class = botorch_submodel_class
        self.submodel_options_per_outcome = submodel_options_per_outcome or {}
        self.submodel_options = submodel_options or {}
        super().__init__(botorch_model_class=ModelListGP, mll_class=mll_class)

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

    def construct(self, training_data: TrainingData, **kwargs: Any) -> None:
        """Constructs the underlying BoTorch ``Model`` using the training data.

        Args:
            training_data: List of ``TrainingData`` for the submodels of
                ``ModelListGP``. Each training data is for one outcome,
                and the order of outcomes should match the order of metrics
                in ``metric_names`` argument.
            **kwargs: Keyword arguments, accepts:
                - ``metric_names`` (required): Names of metrics, in the same order
                as training data (so if training data is ``[tr_A, tr_B]``, the
                metrics are ``["A" and "B"]``). These are used to match training data
                with correct submodels of ``ModelListGP``,
                - ``fidelity_features``: Indices of columns in X that represent
                fidelity,
                - ``task_features``: Indices of columns in X that represent tasks.
        """
        metric_names = kwargs.get(Keys.METRIC_NAMES)
        fidelity_features = kwargs.get(Keys.FIDELITY_FEATURES, [])
        task_features = kwargs.get(Keys.TASK_FEATURES, [])
        if metric_names is None:
            raise ValueError("Metric names are required.")

        self._training_data = training_data
        self._training_data_per_outcome = {
            metric_name: TrainingData.from_block_design(X=X, Y=Y, Yvar=Yvar)
            for metric_name, X, Y, Yvar in zip(
                metric_names,
                training_data.Xs,
                training_data.Ys,
                # `TrainingData.Yvars` can be none, in which case each per-outcome
                # training data should have null `Yvar`.
                training_data.Yvars or [None] * len(metric_names),
            )
        }

        submodels = []
        for m in metric_names:
            model_cls = self.botorch_submodel_class_per_outcome.get(
                m, self.botorch_submodel_class
            )
            if not model_cls:
                raise ValueError(f"No model class specified for outcome {m}.")

            if m not in self.training_data_per_outcome:  # pragma: no cover
                logger.info(f"Metric {m} not in training data.")
                continue

            # NOTE: here we do a shallow copy of `self.submodel_options`, to
            # protect from accidental modification of shared options. As it is
            # a shallow copy, it does not protect the objects in the dictionary,
            # just the dictionary itself.
            submodel_options = {
                **self.submodel_options,
                **self.submodel_options_per_outcome.get(m, {}),
            }

            formatted_model_inputs = model_cls.construct_inputs(
                training_data=self.training_data_per_outcome[m],
                fidelity_features=fidelity_features,
                task_features=task_features,
                **submodel_options,
            )
            # pyre-ignore[45]: Py raises informative error if model is abstract.
            submodels.append(model_cls(**formatted_model_inputs))
        self._model = ModelListGP(*submodels)

    def _serialize_attributes_as_kwargs(self) -> Dict[str, Any]:
        """Serialize attributes of this surrogate, to be passed back to it
        as kwargs on reinstantiation.
        """
        submodel_classes = self.botorch_submodel_class_per_outcome
        return {
            "botorch_submodel_class_per_outcome": submodel_classes,
            "botorch_submodel_class": self.botorch_submodel_class,
            "submodel_options_per_outcome": self.submodel_options_per_outcome,
            "submodel_options": self.submodel_options,
            "mll_class": self.mll_class,
        }
