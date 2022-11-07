#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
from copy import deepcopy

from logging import Logger
from typing import Any, Dict, List, Optional, Type

import torch
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputPerturbation, InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood

logger: Logger = get_logger(__name__)


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
        mll_options: Dictionary of options / kwargs for the MLL.
        submodel_outcome_transforms: An outcome transform that will be used
            by all outcomes. Gets passed down to the BoTorch ``Model``s.
            To use multiple outcome transforms on a submodel, chain them
            together using ``ChainedOutcomeTransform``.
        submodel_input_transforms: An input transform that will be used
            by all outcomes. Gets passed down to the BoTorch ``Model``.
            If sharing a single ``InputTransform`` object across submodels is
            preferred, pass in a dictionary where each outcome key references the
            same ``InputTransform`` object. To use multiple input transfroms on
            a submodel, chain them together using ``ChainedInputTransform``.
        submodel_covar_module_class: A covar module that will be used by all outcomes.
        submodel_covar_module_options: Options for a BoTorch covar module or options
            that will be used by all outcomes.
        submodel_likelihood_class: A likelihood that will be used by all outcomes.
        submodel_likelihood_options: Options for a BoTorch likelihood or options that
            will be used by all outcomes.
    """

    botorch_submodel_class_per_outcome: Dict[str, Type[Model]]
    botorch_submodel_class: Optional[Type[Model]]
    submodel_options_per_outcome: Dict[str, Dict[str, Any]]
    submodel_options: Dict[str, Any]
    mll_class: Type[MarginalLogLikelihood]
    mll_options: Dict[str, Any]
    submodel_outcome_transforms: Optional[OutcomeTransform]
    submodel_input_transforms: Optional[InputTransform]
    submodel_covar_module_class: Optional[Type[Kernel]]
    submodel_covar_module_options: Dict[str, Any]
    submodel_likelihood_class: Optional[Type[Likelihood]]
    submodel_likelihood_options: Dict[str, Any]
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
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        mll_options: Optional[Dict[str, Any]] = None,
        submodel_outcome_transforms: Optional[OutcomeTransform] = None,
        submodel_input_transforms: Optional[InputTransform] = None,
        submodel_covar_module_class: Optional[Type[Kernel]] = None,
        submodel_covar_module_options: Optional[Dict[str, Any]] = None,
        submodel_likelihood_class: Optional[Type[Likelihood]] = None,
        submodel_likelihood_options: Optional[Dict[str, Any]] = None,
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
        self.submodel_outcome_transforms = submodel_outcome_transforms
        self.submodel_input_transforms = submodel_input_transforms
        self.submodel_covar_module_class = submodel_covar_module_class
        self.submodel_covar_module_options = submodel_covar_module_options or {}
        self.submodel_likelihood_class = submodel_likelihood_class
        self.submodel_likelihood_options = submodel_likelihood_options or {}
        super().__init__(
            botorch_model_class=ModelListGP,
            mll_class=mll_class,
            mll_options=mll_options,
        )

    def construct(
        self, datasets: List[SupervisedDataset], metric_names: List[str], **kwargs: Any
    ) -> None:
        """Constructs the underlying BoTorch ``Model`` using the training data.

        Args:
            datasets: List of ``SupervisedDataset`` for the submodels of
                ``ModelListGP``. Each training data is for one outcome, and the order
                of outcomes should match the order of metrics in ``metric_names``
                argument.
            metric_names: Names of metrics, in the same order as datasets (so if
                datasets is ``[ds_A, ds_B]``, the metrics are ``["A" and "B"]``).
                These are used to match training data with correct submodels of
                ``ModelListGP``.
            **kwargs: Keyword arguments, accepts:
                - ``fidelity_features``: Indices of columns in X that represent
                    fidelity
                - ``task_features``: Indices of columns in X that represent tasks
        """
        fidelity_features = kwargs.get(Keys.FIDELITY_FEATURES, [])
        task_features = kwargs.get(Keys.TASK_FEATURES, [])

        if len(fidelity_features) > 0 and len(task_features) > 0:
            raise NotImplementedError(
                "Multi-Fidelity GP models with task_features are "
                "currently not supported."
            )
        # TODO: Allow each metric having different task_features or fidelity_features
        # TODO: Need upstream change in the modelbrdige
        if len(task_features) > 1:
            raise NotImplementedError("This model only supports 1 task feature!")
        elif len(task_features) == 1:
            task_feature = task_features[0]
        else:
            task_feature = None

        self._training_data = datasets

        # Construct input perturbation if doing robust optimization.
        # NOTE: Doing this here rather than in `_set_formatted_inputs` to make sure
        # we use the same perturbations for each sub-model.
        robust_digest: Optional[Dict[str, Any]] = kwargs.get("robust_digest", None)
        if robust_digest is not None:
            if len(robust_digest["environmental_variables"]):
                # TODO[T131759269]: support env variables.
                raise NotImplementedError(
                    "Environmental variable support is not yet implemented."
                )
            samples = torch.as_tensor(
                robust_digest["sample_param_perturbations"](),
                dtype=self.dtype,
                device=self.device,
            )
            perturbation = InputPerturbation(
                perturbation_set=samples, multiplicative=robust_digest["multiplicative"]
            )

            if self.submodel_input_transforms is not None:
                # TODO: Support mixing with user supplied transforms.
                raise NotImplementedError(
                    "User supplied input transforms are not supported "
                    "in robust optimization."
                )
            submodel_input_transforms = perturbation
        else:
            submodel_input_transforms = self.submodel_input_transforms

        submodels = []
        for m, dataset in zip(metric_names, datasets):
            model_cls = self.botorch_submodel_class_per_outcome.get(
                m, self.botorch_submodel_class
            )
            if not model_cls:
                raise ValueError(f"No model class specified for outcome {m}.")

            if self._outcomes is not None and m not in self._outcomes:
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
                training_data=dataset,
                fidelity_features=fidelity_features,
                task_feature=task_feature,
                **submodel_options,
            )
            # Add input / outcome transforms.
            # TODO: The use of `inspect` here is not ideal. We should find a better
            # way to filter the arguments. See the comment in `Surrogate.construct`
            # regarding potential use of a `ModelFactory` in the future.
            model_cls_args = inspect.getfullargspec(model_cls).args
            self._set_formatted_inputs(
                formatted_model_inputs=formatted_model_inputs,
                inputs=[
                    [
                        "covar_module",
                        self.submodel_covar_module_class,
                        self.submodel_covar_module_options,
                        None,
                    ],
                    [
                        "likelihood",
                        self.submodel_likelihood_class,
                        self.submodel_likelihood_options,
                        None,
                    ],
                    [
                        "outcome_transform",
                        None,
                        None,
                        deepcopy(self.submodel_outcome_transforms),
                    ],
                    [
                        "input_transform",
                        None,
                        None,
                        deepcopy(submodel_input_transforms),
                    ],
                ],
                dataset=dataset,
                botorch_model_class_args=model_cls_args,
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
            "mll_options": self.mll_options,
            "submodel_outcome_transforms": self.submodel_outcome_transforms,
            "submodel_input_transforms": self.submodel_input_transforms,
            "submodel_covar_module_class": self.submodel_covar_module_class,
            "submodel_covar_module_options": self.submodel_covar_module_options,
            "submodel_likelihood_class": self.submodel_likelihood_class,
            "submodel_likelihood_options": self.submodel_likelihood_options,
        }
