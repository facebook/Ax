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
    # Special setting for surrogates instantiated via `Surrogate.from_BoTorch`,
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

        self._training_data_per_outcome = {
            metric_name: tr for metric_name, tr in zip(metric_names, training_data)
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
        """Fits the underlying BoTorch ``Model`` to ``m`` outcomes.

        NOTE: ``state_dict`` and ``refit`` keyword arguments control how the
        underlying BoTorch ``Model`` will be fit: whether its parameters will
        be reoptimized and whether it will be warm-started from a given state.

        There are three possibilities:

        * ``fit(state_dict=None)``: fit model from stratch (optimize model
          parameters and set its training data used for inference),
        * ``fit(state_dict=some_state_dict, refit=True)``: warm-start refit
          with a state dict of parameters (still re-optimize model parameters
          and set the training data),
        * ``fit(state_dict=some_state_dict, refit=False)``: load model parameters
          without refitting, but set new training data (used in cross-validation,
          for example).

        Args:
            training data: List of BoTorch ``TrainingData`` container (with Xs,
                Ys, and possibly Yvars), one per outcome, in order corresponding
                to the order of outcome names in ``metric_names``. Each ``TrainingData``
                from this list will be passed to ``Model.construct_inputs`` method
                of the corresponding submodel in ``ModelListGP``.
            bounds: A list of d (lower, upper) tuples for each column of X.
            task_features: Columns of X that take integer values and should be
                treated as task parameters.
            feature_names: Names of each column of X.
            metric_names: Names of each outcome Y in Ys.
            fidelity_features: Columns of X that should be treated as fidelity
                parameters.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: Optional state dict to load.
            refit: Whether to re-optimize model parameters.
        """
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
