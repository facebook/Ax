#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Mapping

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.exceptions.core import AxError, UnsupportedError, UserInputError
from ax.exceptions.model import ModelError
from ax.generators.model_utils import best_in_sample_point
from ax.generators.torch.botorch_modular.input_constructors.covar_modules import (
    covar_module_argparse,
)
from ax.generators.torch.botorch_modular.input_constructors.input_transforms import (
    input_transform_argparse,
)
from ax.generators.torch.botorch_modular.input_constructors.outcome_transform import (
    outcome_transform_argparse,
)
from ax.generators.torch.botorch_modular.utils import (
    convert_to_block_design,
    copy_model_config_with_default_values,
    fit_botorch_model,
    get_all_task_values_from_ssd,
    get_cv_fold,
    ModelConfig,
    subset_state_dict,
    use_model_list,
)
from ax.generators.torch.utils import (
    _to_inequality_constraints,
    pick_best_out_of_sample_point_acqf_class,
    predict_from_model,
)
from ax.generators.torch_base import TorchOptConfig
from ax.generators.types import TConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    _argparse_type_encoder,
    assert_is_instance_optional,
)
from ax.utils.stats.model_fit_stats import (
    DIAGNOSTIC_FN_DIRECTIONS,
    DIAGNOSTIC_FNS,
    ModelFitMetricDirection,
    RANK_CORRELATION,
)
from botorch.exceptions.errors import ModelFittingError
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputPerturbation,
    InputTransform,
    Normalize,
)
from botorch.models.transforms.outcome import ChainedOutcomeTransform, OutcomeTransform
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.settings import validate_input_scaling
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import MultiTaskDataset, RankingDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.evaluation import AIC, BIC, compute_in_sample_model_fit_metric, MLL
from botorch.utils.transforms import normalize_indices
from botorch.utils.types import _DefaultType
from gpytorch.models.exact_gp import ExactGP
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor
from torch.nn import Module

NOT_YET_FIT_MSG = (
    "Underlying BoTorch `Model` has not yet received its training_data. "
    "Please fit the model first."
)

logger: Logger = get_logger(__name__)
MODEL_SELECTION_METRIC_DIRECTIONS: dict[str, ModelFitMetricDirection] = {
    **DIAGNOSTIC_FN_DIRECTIONS,
    MLL: ModelFitMetricDirection.MAXIMIZE,
    AIC: ModelFitMetricDirection.MINIMIZE,
    BIC: ModelFitMetricDirection.MINIMIZE,
}


def _extract_model_kwargs(
    search_space_digest: SearchSpaceDigest, botorch_model_class: type[Model]
) -> dict[str, list[int] | dict[int, dict[int, list[int]]] | int]:
    """
    Extracts keyword arguments that are passed to the `construct_inputs`
    method of a BoTorch `Model` class.

    Args:
        search_space_digest: A `SearchSpaceDigest`.
        botorch_model_class: The BoTorch model class to extract kwargs for.

    Returns:
        A dict of fidelity features, categorical features, and, if present, task
        features.
    """
    fidelity_features = search_space_digest.fidelity_features
    task_features = search_space_digest.task_features
    if len(fidelity_features) > 0 and len(task_features) > 0:
        raise NotImplementedError(
            "Multi-Fidelity GP models with task_features are "
            "currently not supported."
        )
    if len(task_features) > 1:
        raise NotImplementedError("Multiple task features are not supported.")
    elif len(task_features) == 0 and issubclass(botorch_model_class, MultiTaskGP):
        # This is handled in Surrogate.model_selection and the MTGP will be
        # skipped if there is no task feature.
        raise ModelFittingError("Cannot fit MultiTaskGP without task feature.")

    kwargs: dict[str, list[int] | dict[int, dict[int, list[int]]] | int] = {}
    if len(search_space_digest.categorical_features) > 0:
        kwargs["categorical_features"] = search_space_digest.categorical_features
    if len(fidelity_features) > 0:
        kwargs["fidelity_features"] = fidelity_features
    if len(task_features) == 1:
        task_feature = task_features[0]
        if task_feature == len(search_space_digest.bounds) - 1:
            # to support heterogeneous search spaces
            task_feature = -1
        kwargs["task_feature"] = task_feature
    # Regular BoTorch models do not expect the argument `hierarchical_dependencies`.
    # For now, it is the user's responsibility to make sure a hierarchical model is used
    # when the HSS is not flattened.
    if search_space_digest.hierarchical_dependencies:
        kwargs["hierarchical_dependencies"] = (
            search_space_digest.hierarchical_dependencies
        )
    return kwargs


def _make_botorch_input_transform(
    input_transform_classes: list[type[InputTransform]] | _DefaultType,
    input_transform_options: dict[str, dict[str, Any]],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
) -> InputTransform | None:
    """
    Makes a BoTorch input transform from the provided input classes and options.
    """
    if isinstance(input_transform_classes, _DefaultType):
        transforms = _construct_default_input_transforms(
            search_space_digest=search_space_digest, dataset=dataset
        )
    else:
        transforms = _construct_specified_input_transforms(
            input_transform_classes=input_transform_classes,
            dataset=dataset,
            search_space_digest=search_space_digest,
            input_transform_options=input_transform_options,
        )
    if len(transforms) == 0:
        return None
    elif len(transforms) > 1:
        return ChainedInputTransform(
            **{f"tf{i}": t_i for i, t_i in enumerate(transforms)}
        )
    else:
        return transforms[0]


def _construct_specified_input_transforms(
    input_transform_classes: list[type[InputTransform]],
    input_transform_options: dict[str, dict[str, Any]],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
) -> list[InputTransform]:
    """Constructs a list of input transforms from input transform classes and
    options provided in ``ModelConfig``.
    """
    if not (
        isinstance(input_transform_classes, list)
        and all(issubclass(c, InputTransform) for c in input_transform_classes)
    ):
        raise UserInputError(
            "Expected a list of input transform classes. "
            f"Got {input_transform_classes=}."
        )
    if search_space_digest.robust_digest is not None:
        input_transform_classes = [InputPerturbation] + input_transform_classes

    input_transform_kwargs = [
        input_transform_argparse(
            transform_class,
            dataset=dataset,
            search_space_digest=search_space_digest,
            input_transform_options=deepcopy(  # In case of in-place modifications.
                input_transform_options.get(transform_class.__name__, {})
            ),
        )
        for transform_class in input_transform_classes
    ]

    return [
        # pyre-fixme[45]: Cannot instantiate abstract class `InputTransform`.
        transform_class(**single_input_transform_kwargs)
        for transform_class, single_input_transform_kwargs in zip(
            input_transform_classes, input_transform_kwargs
        )
    ]


def _construct_default_input_transforms(
    search_space_digest: SearchSpaceDigest,
    dataset: SupervisedDataset,
) -> list[InputTransform]:
    """Construct the default input transforms for the given search space digest.

    The default transforms are added in this order:
    - If the search space digest has a robust digest, an ``InputPerturbation`` transform
        is used.
    - If the bounds for the non-task features are not [0, 1], a ``Normalize`` transform
        is used. The transfrom only applies to the non-task features.
    """
    transforms = []
    # Add InputPerturbation if there is a robust digest.
    if search_space_digest.robust_digest is not None:
        transforms.append(
            InputPerturbation(
                **input_transform_argparse(
                    InputPerturbation,
                    dataset=dataset,
                    search_space_digest=search_space_digest,
                )
            )
        )
    # Processing for Normalize.
    input_transform_options = input_transform_argparse(
        Normalize,
        dataset=dataset,
        search_space_digest=search_space_digest,
    )
    bounds = input_transform_options.get("bounds")
    indices = input_transform_options.get("indices")
    # Skip the Normalize transform if the bounds are [0, 1].
    if bounds is not None:
        if indices is not None:
            bounds = bounds[:, indices]
        lower_bounds, upper_bounds = bounds
        if torch.allclose(
            lower_bounds, torch.zeros_like(lower_bounds)
        ) and torch.allclose(upper_bounds, torch.ones_like(upper_bounds)):
            return transforms
    transforms.append(Normalize(**input_transform_options))
    return transforms


def _make_botorch_outcome_transform(
    outcome_transform_classes: list[type[OutcomeTransform]],
    outcome_transform_options: dict[str, dict[str, Any]],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
) -> OutcomeTransform | None:
    """
    Makes a BoTorch outcome transform from the provided classes and options.
    """
    if not (
        isinstance(outcome_transform_classes, list)
        and all(issubclass(c, OutcomeTransform) for c in outcome_transform_classes)
    ):
        raise UserInputError("Expected a list of outcome transforms.")
    if len(outcome_transform_classes) == 0:
        return None

    outcome_transform_kwargs = [
        outcome_transform_argparse(
            transform_class,
            outcome_transform_options=deepcopy(  # In case of in-place modifications.
                outcome_transform_options.get(transform_class.__name__, {})
            ),
            dataset=dataset,
            search_space_digest=search_space_digest,
        )
        for transform_class in outcome_transform_classes
    ]

    outcome_transforms = [
        # pyre-fixme[45]: Cannot instantiate abstract class `OutcomeTransform`.
        transform_class(**single_outcome_transform_kwargs)
        for transform_class, single_outcome_transform_kwargs in zip(
            outcome_transform_classes, outcome_transform_kwargs
        )
    ]

    outcome_transform_instance = (
        ChainedOutcomeTransform(
            **{f"otf{i}": otf for i, otf in enumerate(outcome_transforms)}
        )
        if len(outcome_transforms) > 1
        else outcome_transforms[0]
    )
    return outcome_transform_instance


def _construct_submodules(
    model_config: ModelConfig,
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    botorch_model_class: type[Model],
) -> dict[str, Module | None]:
    """Constructs the submodules for the BoTorch model from the inputs
    extracted from the ``ModelConfig``. If the corresponding inputs are
    specified, the `covar_module`, `likelihood`, `input_transform`, and
    `outcome_transform` submodules are constructed.
    """
    botorch_model_class_args: list[str] = inspect.getfullargspec(
        botorch_model_class
    ).args

    def _error_if_arg_not_supported(arg_name: str) -> None:
        if arg_name not in botorch_model_class_args:
            raise UserInputError(
                f"The BoTorch model class {botorch_model_class.__name__} does not "
                f"support the input {arg_name}."
            )

    submodules: dict[str, Module | None] = {}
    # NOTE: Using the walrus operator here and below helps pyre.
    if (covar_class := model_config.covar_module_class) is not None:
        _error_if_arg_not_supported("covar_module")
        covar_module_kwargs = covar_module_argparse(
            covar_class,
            dataset=dataset,
            botorch_model_class=botorch_model_class,
            **deepcopy(model_config.covar_module_options),
        )
        # pyre-ignore [45]: Cannot instantiate abstract class `Kernel`.
        submodules["covar_module"] = covar_class(**covar_module_kwargs)

    if (likelihood_class := model_config.likelihood_class) is not None:
        _error_if_arg_not_supported("likelihood")
        # pyre-ignore [45]: Cannot instantiate abstract class `Likelihood`.
        submodules["likelihood"] = likelihood_class(
            **deepcopy(model_config.likelihood_options)
        )

    if (
        input_transform_classes := model_config.input_transform_classes
    ) is not None or search_space_digest.robust_digest is not None:
        _error_if_arg_not_supported("input_transform")
        submodules["input_transform"] = _make_botorch_input_transform(
            input_transform_classes=input_transform_classes or [],
            input_transform_options=model_config.input_transform_options or {},
            dataset=dataset,
            search_space_digest=search_space_digest,
        )

    if (
        outcome_transform_classes := model_config.outcome_transform_classes
    ) is not None:
        _error_if_arg_not_supported("outcome_transform")
        submodules["outcome_transform"] = _make_botorch_outcome_transform(
            outcome_transform_classes=outcome_transform_classes,
            outcome_transform_options=model_config.outcome_transform_options or {},
            dataset=dataset,
            search_space_digest=search_space_digest,
        )
    elif "outcome_transform" in botorch_model_class_args:
        # This is a temporary solution until all BoTorch models use
        # `Standardize` by default, see TODO [T197435440].
        # After this, we should update `Surrogate` to use `DEFAULT`
        # (https://fburl.com/code/22f4397e) for both of these args. This will
        # allow users to explicitly disable the default transforms by passing
        # in `None`.
        submodules["outcome_transform"] = None

    return submodules


@dataclass(frozen=True)
class SurrogateSpec:
    """
    Fields in the SurrogateSpec dataclass correspond to arguments in
    ``Surrogate.__init__``, except for ``outcomes`` which is used to specify which
    outcomes the Surrogate is responsible for modeling.
    When ``BoTorchGenerator.fit`` is called, these fields will be used to construct the
    requisite Surrogate objects.
    If ``outcomes`` is left empty then no outcomes will be fit to the Surrogate.

    Args:
        model_configs: List of model configs. Each model config is a specification of
            a surrogate model. Defaults to a single ``ModelConfig`` with all defaults.
        metric_to_model_configs: Dictionary mapping metric signatures to a list of model
            configs for that metric.
        eval_criterion: The name of the evaluation criteria to use. These are defined in
            ``ax.utils.stats.model_fit_stats``. Defaults to rank correlation.
        outcomes: List of outcomes names.
        use_posterior_predictive: Whether to use posterior predictive in
            cross-validation.
        num_folds: The number of folds to use in cross-validation. If None, then
            leave-one-out.
    """

    model_configs: list[ModelConfig] = field(default_factory=lambda: [ModelConfig()])
    metric_to_model_configs: dict[str, list[ModelConfig]] = field(default_factory=dict)
    eval_criterion: str = RANK_CORRELATION
    outcomes: list[str] = field(default_factory=list)
    allow_batched_models: bool = True
    use_posterior_predictive: bool = False
    num_folds: int | None = 10


class Surrogate(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch ``Model``, subcomponent of ``BoTorchGenerator``
    and is not meant to be used outside of it.

    Args:
        surrogate_spec: A ``SurrogateSpec`` that specifies the option to use when
            constructing the surrogate models. See the docstring of ``SurrogateSpec``
            for supported options and ``ModelConfig`` for additional details.
        allow_batched_models: Set to true to fit the models in a batch if supported.
            Set to false to fit individual models to each metric in a loop.
        refit_on_cv: Whether to refit the model on the cross-validation folds.
        warm_start_refit: Whether to warm-start refitting from the current state_dict
            during cross-validation. If refit_on_cv is True, generally one
            would set this to be False, so that no information is leaked between or
            across folds.
        metric_to_best_model_config: Dictionary mapping a metric signature to the best
            model config. This is only used by `BoTorchGenerator.cross_validate` and
            for logging what model was used.

    """

    def __init__(
        self,
        surrogate_spec: SurrogateSpec | None = None,
        allow_batched_models: bool = True,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
        metric_to_best_model_config: dict[str, ModelConfig] | None = None,
    ) -> None:
        if surrogate_spec is None:
            surrogate_spec = SurrogateSpec(allow_batched_models=allow_batched_models)

        self.surrogate_spec: SurrogateSpec = surrogate_spec
        # Store the last dataset used to fit the model for a given metric(s).
        # If the new dataset is identical, we will skip model fitting for that metric.
        # The keys are `tuple(dataset.outcome_names)`.
        self._last_datasets: dict[tuple[str], SupervisedDataset] = {}
        # Store a reference from a tuple of metric signatures to the BoTorch Model
        # corresponding to those metrics. In most cases this will be a one-tuple,
        # though we need n-tuples for LCE-M models. This will be used to skip model
        # construction & fitting if the datasets are identical.
        self._submodels: dict[tuple[str], Model] = {}
        self.metric_to_best_model_config: dict[str, ModelConfig] = (
            metric_to_best_model_config or {}
        )
        # Store a reference to search space digest used while fitting the cached models.
        # We will re-fit the models if the search space digest changes.
        self._last_search_space_digest: SearchSpaceDigest | None = None

        # These are later updated during model fitting.
        self._training_data: list[SupervisedDataset] | None = None
        self._outcomes: list[str] | None = None
        self._model: Model | None = None
        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit
        # Updated during model selection
        self._model_name_to_eval: dict[str, dict[str, float]] = {}
        self._model_name_to_model: dict[str, dict[str, Model]] = {}

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}" f" surrogate_spec={self.surrogate_spec}>"

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ModelError(
                "BoTorch `Model` has not yet been constructed, please fit the "
                "surrogate first (done via `BoTorchGenerator.fit`)."
            )
        return self._model

    @property
    def training_data(self) -> list[SupervisedDataset]:
        if self._training_data is None:
            raise ModelError(NOT_YET_FIT_MSG)
        return self._training_data

    @property
    def Xs(self) -> list[Tensor]:
        # Handles multi-output models. TODO: Improve this!
        training_data = self.training_data
        Xs = []
        for dataset in training_data:
            if isinstance(dataset, RankingDataset):
                # directly accessing the d-dim X tensor values
                # instead of the augmented 2*d-dim dataset.X from RankingDataset
                Xi = assert_is_instance(
                    dataset._X,
                    SliceContainer,
                ).values
            else:
                Xi = dataset.X
            for _ in range(dataset.Y.shape[-1]):
                Xs.append(Xi)
        return Xs

    @property
    def dtype(self) -> torch.dtype:
        return self.training_data[0].X.dtype

    @property
    def device(self) -> torch.device:
        return self.training_data[0].X.device

    def clone_reset(self) -> Surrogate:
        return self.__class__(**self._serialize_attributes_as_kwargs())

    def _construct_model(
        self,
        dataset: SupervisedDataset,
        search_space_digest: SearchSpaceDigest,
        model_config: ModelConfig,
        state_dict: Mapping[str, Tensor] | None,
        refit: bool,
    ) -> Model:
        """Constructs the underlying BoTorch ``Model`` using the training data.

        If the dataset and model class are identical to those used while training
        the cached sub-model, we skip model fitting and return the cached model.

        Args:
            dataset: Training data for the model (for one outcome for
                the default `Surrogate`, with the exception of batched
                multi-output case, where training data is formatted with just
                one X and concatenated Ys).
            search_space_digest: Search space digest used to set up model arguments.
            model_config: The model_config.
            default_botorch_model_class: The default ``Model`` class to be used as the
                underlying BoTorch model, if the model_config does not specify one.
            state_dict: Optional state dict to load. This should be subsetted for
                the current submodel being constructed.
            refit: Whether to re-optimize model parameters.
        """
        outcome_names = tuple(dataset.outcome_names)
        # Fill in default values for model_configs given dataset
        model_config = copy_model_config_with_default_values(
            model_config=model_config,
            dataset=dataset,
            search_space_digest=search_space_digest,
        )
        botorch_model_class = none_throws(model_config.botorch_model_class)
        if self._dataset_matches_cache(dataset=dataset):
            return self._submodels[outcome_names]
        formatted_model_inputs = submodel_input_constructor(
            botorch_model_class,  # Do not pass as kwarg since this is used to dispatch.
            model_config=model_config,
            dataset=dataset,
            search_space_digest=search_space_digest,
            surrogate=self,
        )
        # pyre-ignore [45]
        model = botorch_model_class(**formatted_model_inputs)
        if state_dict is not None and (not refit or self.warm_start_refit):
            model.load_state_dict(state_dict)
        if state_dict is None or refit:
            fit_botorch_model(
                model,  # Intentionally not using named args for dispatcher.
                mll_class=model_config.mll_class,
                mll_options=model_config.mll_options,
            )

        return model

    def _dataset_matches_cache(
        self,
        dataset: SupervisedDataset,
    ) -> bool:
        """Returns `True` if the given dataset matches the last dataset used to fit
        the model for the corresponding outcomes.
        """
        outcome_names = tuple(dataset.outcome_names)
        return (
            outcome_names in self._submodels
            and dataset == self._last_datasets[outcome_names]
        )

    def fit(
        self,
        datasets: Sequence[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
        state_dict: Mapping[str, Tensor] | None = None,
        refit: bool = True,
        repeat_model_selection_if_dataset_changed: bool = True,
    ) -> None:
        """Fits the underlying BoTorch ``Model`` to ``m`` outcomes.

        NOTE: ``state_dict`` and ``refit`` keyword arguments control how the
        underlying BoTorch ``Model`` will be fit: whether its parameters will
        be reoptimized and whether it will be warm-started from a given state.

        There are three possibilities:

        * ``fit(state_dict=None)``: fit model from scratch (optimize model
          parameters and set its training data used for inference),
        * ``fit(state_dict=some_state_dict, refit=True)``: warm-start refit
          with a state dict of parameters (still re-optimize model parameters
          and set the training data),
        * ``fit(state_dict=some_state_dict, refit=False)``: load model parameters
          without refitting, but set new training data (used in cross-validation,
          for example).

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome), to be passed
                to ``Model.construct_inputs`` in BoTorch.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: Optional state dict to load.
            refit: Whether to re-optimize model parameters.
            repeat_model_selection_if_dataset_changed: Whether to repeat model
                selection, ignoring previously found best config, if the dataset
                for the corresponding outcomes has changed. This is typically
                set to `True` when called from ``BoTorchGenerator.fit`` but set
                to `False` when called from ``BoTorchGenerator.cross_validate``.
                During cross_validation, we want to evaluate the quality of the
                previously selected best model, rather than repeating model selection
                for each fold.
        """
        self._discard_cached_model_and_data_if_search_space_digest_changed(
            search_space_digest=search_space_digest
        )
        # Deepcopy so that we are not making in-place changes when fill default values
        metric_to_model_configs = deepcopy(self.surrogate_spec.metric_to_model_configs)

        # To determine whether to use ModelList under the hood, we need to check for
        # the batched multi-output case, so we first see which model would be chosen
        # given the Yvars and the properties of data.
        should_use_model_list = use_model_list(
            datasets=datasets,
            search_space_digest=search_space_digest,
            model_configs=self.surrogate_spec.model_configs,
            allow_batched_models=self.surrogate_spec.allow_batched_models,
            metric_to_model_configs=metric_to_model_configs,
        )

        if not should_use_model_list and len(datasets) > 1:
            try:
                datasets = convert_to_block_design(datasets=datasets, force=False)
            except UnsupportedError as e:
                # If the block design conversion fails, use model-list.
                logger.warning(
                    "Conversion to block design failed. Using model-list instead. "
                    f"Original error: {e}"
                )
                should_use_model_list = True
        self._training_data = list(datasets)  # So that it can be modified if needed.

        feature_names_set = set(search_space_digest.feature_names)
        models = []
        outcome_names = []
        for i, dataset in enumerate(datasets):
            submodel_state_dict = None
            if state_dict is not None:
                if should_use_model_list:
                    submodel_state_dict = subset_state_dict(
                        state_dict=state_dict, submodel_index=i
                    )
                else:
                    submodel_state_dict = state_dict
            outcome_name_tuple = tuple(dataset.outcome_names)
            first_outcome_name = outcome_name_tuple[0]

            # If no model config is specified for the (likely aux) preference dataset,
            # fallback to the default model config to avoid model fitting failure
            if (
                isinstance(dataset, RankingDataset)
                and first_outcome_name not in metric_to_model_configs
            ):
                metric_to_model_configs[first_outcome_name] = [ModelConfig()]

            model_configs = (
                metric_to_model_configs[first_outcome_name]
                if first_outcome_name in metric_to_model_configs
                else self.surrogate_spec.model_configs
            )
            # Case 1: There is either 1 model config, or we don't want to re-do
            # model selection and we know what the previous best model was.
            if (
                not repeat_model_selection_if_dataset_changed
                or self._dataset_matches_cache(dataset=dataset)
            ):
                # Re-use the best model config, if the dataset hasn't changed or
                # `repeat_model_selection_if_dataset_changed` is set to `False`.
                model_config = self.metric_to_best_model_config.get(first_outcome_name)
            else:
                model_config = None
            # Model selection is not performed if the best `ModelConfig` has already
            # been identified (as specified in `metric_to_best_model_config`).
            # The reason for doing this is to support the following flow:
            # - Fit model to data and perform model selection, refitting on each fold
            #   if `refit_on_cv=True`. This will set the best ModelConfig in
            #   metric_to_best_model_config.
            # - Evaluate the choice of model/visualize its performance via
            #   `Adapter.cross_validate``. This also will refit on each fold if
            #   `refit_on_cv=True`, but we wouldn't want to perform model selection
            #   on each fold, but rather show the performance of the selecting
            #   `ModelConfig`` since that is what will be used.
            if len(model_configs) == 1 or (model_config is not None):
                best_model_config = model_config or model_configs[0]
                model = self._construct_model(
                    dataset=dataset,
                    search_space_digest=search_space_digest,
                    model_config=best_model_config,
                    state_dict=submodel_state_dict,
                    refit=refit,
                )
            # Case 2: There is more than 1 model config and we want to refit
            # or don't know what the previous best model was
            else:
                if len(dataset.outcome_names) > 1:
                    raise UnsupportedError(
                        "Multiple model configs are not supported with datasets that"
                        " contain multiple outcomes. Each dataset must contain only "
                        "one outcome."
                    )
                model, best_model_config = self.model_selection(
                    dataset=dataset,
                    model_configs=model_configs,
                    search_space_digest=search_space_digest,
                    candidate_metadata=candidate_metadata,
                )

            # Only update the outcome names and models if the dataset input matches
            # the feature names from the search space digest. Otherwise we only
            # keep the model within self._submodels as it may be models fitted on
            # auxiliary data such as the preference model for BOPE
            if set(dataset.feature_names) == feature_names_set:
                models.append(model)
                outcome_names.extend(dataset.outcome_names)

            # store best model config, model, and dataset
            for metric_signature in dataset.outcome_names:
                self.metric_to_best_model_config[metric_signature] = none_throws(
                    best_model_config
                )
            self._submodels[outcome_name_tuple] = model
            self._last_datasets[outcome_name_tuple] = dataset

        if should_use_model_list:
            if all(isinstance(model, GPyTorchModel) for model in models):
                self._model = ModelListGP(*models)
            else:
                self._model = ModelList(*models)
        else:
            self._model = models[0]
        self._outcomes = outcome_names  # In the order of input datasets

    def model_selection(
        self,
        dataset: SupervisedDataset,
        model_configs: list[ModelConfig],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
    ) -> tuple[Model, ModelConfig]:
        """Perform model selection over a list of model configs.

        This selects the best botorch Model across the provided model configs
        based on the SurrogateSpec's eval_criteria. The eval_criteria is
        computed using LOOCV on the provided dataset. The best model config is saved
        in self.metric_to_best_model_config for future use (e.g. for using cross-
        validation at the Adapter level).

        Args:
            dataset: Training data for the model
            model_configs: The model_configs.
            default_botorch_model_class: The default ``Model`` class to be used as
                the default, if no botorch_model_class is specified in the
                    model_config.
            search_space_digest: Search space digest.
            candidate_metadata: Model-produced metadata for candidates.

        Returns:
            A two element tuple containing:
                - The best model according to the eval_criterion.
                - The ModelConfig for the best model.

        """
        if (
            isinstance(dataset, MultiTaskDataset)
            and assert_is_instance(dataset, MultiTaskDataset).has_heterogeneous_features
        ):
            raise UnsupportedError(
                "Model selection is not supported for datasets with heterogeneous "
                "features."
            )
        # loop over model configs, fit model for each config, perform LOOCV, select
        # best model according to specified criterion
        maximize = (
            MODEL_SELECTION_METRIC_DIRECTIONS[self.surrogate_spec.eval_criterion]
            == ModelFitMetricDirection.MAXIMIZE
        )
        prefix = "-" if maximize else ""
        best_eval_metric = float(f"{prefix}inf")
        best_model = None
        best_model_config = None
        outcome_name = dataset.outcome_names[0]
        self._model_name_to_eval[outcome_name] = {}
        self._model_name_to_model[outcome_name] = {}
        for model_config in model_configs:
            # fit model to all data
            try:
                model = self._construct_model(
                    dataset=dataset,
                    search_space_digest=search_space_digest,
                    model_config=model_config,
                    state_dict=None,
                    refit=True,
                )
                state_dict = model.state_dict()
                # perform LOOCV
                if self.surrogate_spec.eval_criterion in (AIC, BIC, MLL):
                    eval_metric = compute_in_sample_model_fit_metric(
                        model=assert_is_instance(model, ExactGP),
                        criterion=self.surrogate_spec.eval_criterion,
                    )
                else:
                    eval_metric = self.cross_validate(
                        dataset=dataset,
                        search_space_digest=search_space_digest,
                        model_config=model_config,
                        # pyre-fixme [6]: In call `Surrogate.cross_validate`, for
                        # argument  `state_dict`, expected `Optional[OrderedDict[str,
                        # Tensor]]` but got `Dict[str, typing.Any]`.
                        state_dict=state_dict,
                    )
            except ModelFittingError as e:
                logger.warning(
                    f"Model {model_config} failed to fit with error {e}. Skipping."
                )
                continue
            self._model_name_to_eval[outcome_name][model_config.identifier] = (
                eval_metric
            )
            self._model_name_to_model[outcome_name][model_config.identifier] = model
            if maximize ^ (eval_metric < best_eval_metric):
                best_eval_metric = eval_metric
                best_model = model
                best_model_config = model_config
        if best_model is None:
            raise AxError(
                "No model configs were able to fit the data. Please check your "
                "model configs and/or data."
            )
        return none_throws(best_model), none_throws(best_model_config)

    def cross_validate(
        self,
        dataset: SupervisedDataset,
        model_config: ModelConfig,
        search_space_digest: SearchSpaceDigest,
        state_dict: OrderedDict[str, Tensor] | None = None,
    ) -> float:
        """Cross-validation for a single outcome.

        Args:
            dataset: Training data for the model (for one outcome for
                the default `Surrogate`, with the exception of batched
                multi-output case, where training data is formatted with just
                one X and concatenated Ys).
            model_config: The model_config.
            search_space_digest: Search space digest used to set up model arguments.
            state_dict: Optional state dict to load.

        Returns:
            The eval criterion value for the given model config.
        """
        if isinstance(dataset, MultiTaskDataset):
            # only evaluate model on target task
            target_dataset = dataset.datasets[dataset.target_outcome_name]
        else:
            target_dataset = dataset
        X, Y = target_dataset.X, target_dataset.Y
        num_folds = self.surrogate_spec.num_folds
        if num_folds is None or num_folds > X.shape[0]:
            num_folds = X.shape[0]
        test_folds = np.array_split(
            ary=torch.arange(X.shape[0], device=X.device),
            indices_or_sections=num_folds,
        )
        cv_folds = (
            get_cv_fold(dataset=dataset, X=X, Y=Y, idcs=idcs)  # pyre-ignore[6]
            for idcs in test_folds
        )

        pred_Y = []
        pred_Yvar = []
        obs_Y = []
        for fold in cv_folds:
            # Since each CV fold removes points from the training data, the
            # remaining observations will not pass the input scaling checks.
            # To avoid confusing users with warnings, we disable these checks.
            with validate_input_scaling(False):
                loo_model = self._construct_model(
                    dataset=fold.train_dataset,
                    search_space_digest=search_space_digest,
                    model_config=model_config,
                    state_dict=state_dict,
                    refit=self.refit_on_cv,
                )
            # evaluate model
            with torch.no_grad():
                posterior = loo_model.posterior(
                    fold.test_X,
                    observation_noise=self.surrogate_spec.use_posterior_predictive,
                )
                # TODO: support non-GPyTorch posteriors
                posterior = assert_is_instance(posterior, GPyTorchPosterior)
                if isinstance(posterior, GaussianMixturePosterior):
                    pred_mean = posterior.mixture_mean
                    pred_var = posterior.mixture_variance
                else:
                    pred_mean = posterior.mean
                    pred_var = posterior.variance
            pred_Y.append(pred_mean)
            pred_Yvar.append(pred_var)
            obs_Y.append(fold.test_Y)
        # Stack results
        pred_Y = torch.cat(pred_Y)
        pred_Yvar = torch.cat(pred_Yvar)
        obs_Y = torch.cat(obs_Y)
        # evaluate model fit metric
        diag_fn = DIAGNOSTIC_FNS[none_throws(self.surrogate_spec.eval_criterion)]
        return diag_fn(
            y_obs=obs_Y.view(-1).cpu().numpy(),
            y_pred=pred_Y.view(-1).cpu().numpy(),
            se_pred=np.sqrt(pred_Yvar.view(-1).cpu().numpy()),
        )

    def _discard_cached_model_and_data_if_search_space_digest_changed(
        self, search_space_digest: SearchSpaceDigest
    ) -> None:
        """Checks whether the search space digest has changed since the last call
        to `fit`. If it has, discards cached model and datasets. Also updates
        `self._last_search_space_digest` for future checks.
        """
        if (
            self._last_search_space_digest is not None
            and search_space_digest != self._last_search_space_digest
        ):
            logger.debug(
                "Discarding all previously trained models due to a change "
                "in the search space digest."
            )
            self._submodels = {}
            self._last_datasets = {}
            self.metric_to_best_model_config = {}
        self._last_search_space_digest = search_space_digest

    def predict(
        self, X: Tensor, use_posterior_predictive: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Predicts outcomes given an input tensor.

        Args:
            X: A ``n x d`` tensor of input parameters.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).

        Returns:
            Tensor: The predicted posterior mean as an ``n x o``-dim tensor.
            Tensor: The predicted posterior covariance as a ``n x o x o``-dim tensor.
        """
        return predict_from_model(
            model=self.model, X=X, use_posterior_predictive=use_posterior_predictive
        )

    def best_in_sample_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        options: TConfig | None = None,
    ) -> tuple[Tensor, float]:
        """Finds the best observed point and the corresponding observed outcome
        values.
        """
        if torch_opt_config.is_moo:
            raise NotImplementedError(
                "Best observed point is incompatible with MOO problems."
            )
        best_point_and_observed_value = best_in_sample_point(
            Xs=self.Xs,
            model=self,
            bounds=search_space_digest.bounds,
            objective_weights=torch_opt_config.objective_weights,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            risk_measure=torch_opt_config.risk_measure,
            options=options,
        )
        if best_point_and_observed_value is None:
            raise ValueError("Could not obtain best in-sample point.")
        best_point, observed_value = best_point_and_observed_value
        return (
            best_point.to(dtype=self.dtype, device=torch.device("cpu")),
            observed_value,
        )

    def best_out_of_sample_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        options: TConfig | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Finds the best predicted point and the corresponding value of the
        appropriate best point acquisition function.

        Args:
            search_space_digest: A `SearchSpaceDigest`.
            torch_opt_config: A `TorchOptConfig`; none-None `fixed_features` is
                not supported.
            options: Optional. If present, `seed_inner` (default None) and `qmc`
                (default True) will be parsed from `options`; any other keys
                will be ignored.

        Returns:
            A two-tuple (`candidate`, `acqf_value`), where `candidate` is a 1d
            Tensor of the best predicted point and `acqf_value` is a scalar (0d)
            Tensor of the acquisition function value at the best point.
        """
        if torch_opt_config.fixed_features:
            # When have fixed features, need `FixedFeatureAcquisitionFunction`
            # which has peculiar instantiation (wraps another acquisition fn.),
            # so need to figure out how to handle.
            # TODO (ref: https://fburl.com/diff/uneqb3n9)
            raise NotImplementedError("Fixed features not yet supported.")

        options = options or {}
        botorch_acqf_class, botorch_acqf_options = (
            pick_best_out_of_sample_point_acqf_class(
                outcome_constraints=torch_opt_config.outcome_constraints,
                seed_inner=assert_is_instance_optional(
                    options.get(Keys.SEED_INNER, None), int
                ),
                qmc=assert_is_instance(
                    options.get(Keys.QMC, True),
                    bool,
                ),
                risk_measure=torch_opt_config.risk_measure,
            )
        )

        # Avoiding circular import between `Surrogate` and `Acquisition`.
        from ax.generators.torch.botorch_modular.acquisition import Acquisition

        acqf = Acquisition(
            surrogate=self,
            botorch_acqf_class=botorch_acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_options=botorch_acqf_options,
        )
        candidates, acqf_value, _ = acqf.optimize(
            n=1,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
        )
        return candidates[0], acqf_value

    def pareto_frontier(self) -> tuple[Tensor, Tensor]:
        """For multi-objective optimization, retrieve Pareto frontier instead
        of best point.

        Returns: A two-tuple of:
            - tensor of points in the feature space,
            - tensor of corresponding (multiple) outcomes.
        """
        raise NotImplementedError("Pareto frontier not yet implemented.")

    def compute_diagnostics(self) -> dict[str, Any]:
        """Computes model diagnostics like cross-validation measure of fit, etc."""
        return {}

    def _serialize_attributes_as_kwargs(self) -> dict[str, Any]:
        """Serialize attributes of this surrogate, to be passed back to it
        as kwargs on reinstantiation.
        """
        return {
            "surrogate_spec": self.surrogate_spec,
            "refit_on_cv": self.refit_on_cv,
            "warm_start_refit": self.warm_start_refit,
            "metric_to_best_model_config": self.metric_to_best_model_config,
        }

    @property
    def outcomes(self) -> list[str]:
        if self._outcomes is None:
            raise ModelError("`outcomes` was not initialized. Please call `fit` first.")
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: list[str]) -> None:
        raise RuntimeError("Setting outcomes manually is disallowed.")

    @property
    def model_name_by_metric(self) -> dict[str, str]:
        """Returns a dictionary mapping metric signatures to model names."""
        return {
            metric_signature: model_config.identifier
            for metric_signature, model_config in (
                self.metric_to_best_model_config.items()
            )
        }

    def models_for_gen(self, n: int) -> tuple[list[dict[str, str]], list[Model]]:
        """For each metric, draw `n` samples of models to use for generating `n`
        candidates.

        This method samples which models to use proportional to their model
        selection eval criterion value. Sampling is made scale-free with
        respect to the eval criterion by scaling with respect to the minimum
        value across models. This means the model with worst eval criterion
        will never be sampled, and other models will be sampled proportional
        to the amount of the gap to best value they are able to capture. This
        also means that sampling requires at least 3 models, and if fitting
        here used less than 3, the default self.model will be returned. In that
        case, the method will return a list with just the 1 model and not n.

        The output length will thus be either 1 or n, depending on if the method
        was successful in sampling n models or not.

        Args:
            n: The number of models to return.

        Returns:
            model_names: A list of the model_name_by_metric dict used for each
                candidate.
            models: A list of the models to be used for each candidate.
        """
        if len(self._model_name_to_eval[self.outcomes[0]]) < 3:
            logger.debug("Less than 3 models, no sampling to do for batch generation.")
            return [self.model_name_by_metric], [self.model]

        models = []
        model_names = []
        maximize = (
            MODEL_SELECTION_METRIC_DIRECTIONS[self.surrogate_spec.eval_criterion]
            == ModelFitMetricDirection.MAXIMIZE
        )
        for _ in range(n):
            models_i = []
            model_names_i = {}
            for outcome in self.outcomes:
                w = np.array(list(self._model_name_to_eval[outcome].values()))
                if not maximize:
                    w = -w
                w = w - w.min()
                if w.max() < 1e-10:
                    # All eval values are identical within tolerance.
                    # Uniform sampling.
                    w = np.ones_like(w)
                w = w / w.sum()
                model_name = str(
                    np.random.choice(
                        list(self._model_name_to_eval[outcome].keys()), p=w, size=1
                    )[0]
                )
                models_i.append(self._model_name_to_model[outcome][model_name])
                model_names_i[outcome] = model_name
            if isinstance(self._model, ModelListGP):
                models.append(ModelListGP(*models_i))
            elif isinstance(self._model, ModelList):
                models.append(ModelList(*models_i))
            elif len(models_i) > 1:
                # If MBM supports ModelList in the future, this will need to be
                # updated here.
                raise ValueError(
                    "Got multiple models but not a ModelListGP or ModelList."
                )  # pragma: no cover
            else:
                models.append(models_i[0])
            model_names.append(model_names_i)
        return model_names, models


submodel_input_constructor = Dispatcher(
    name="submodel_input_constructor", encoder=_argparse_type_encoder
)


@submodel_input_constructor.register(Model)
def _submodel_input_constructor_base(
    botorch_model_class: type[Model],
    model_config: ModelConfig,
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    surrogate: Surrogate,
) -> dict[str, Any]:
    """Construct the inputs required to initialize a BoTorch model.

    Args:
        botorch_model_class: The BoTorch model class to instantiate.
        model_config: The model config.
        dataset: The training data for the model.
        search_space_digest: Search space digest used to set up model arguments.
        surrogate: A reference to the surrogate that created the model.
            This can be used by the constructor to obtain any additional
            arguments that are not readily available.

    Returns:
        A dictionary of inputs for constructing the model.
    """
    model_kwargs_from_ss = _extract_model_kwargs(
        search_space_digest=search_space_digest, botorch_model_class=botorch_model_class
    )
    # pop task feature if this is not a multi-task model
    task_feature = model_kwargs_from_ss.get("task_feature", None)
    if not issubclass(botorch_model_class, MultiTaskGP):
        if task_feature is not None:
            model_kwargs_from_ss.pop("task_feature")
            logger.debug(
                f"Removing task feature for {botorch_model_class.__name__}",
                stacklevel=5,
            )
    formatted_model_inputs: dict[str, Any] = botorch_model_class.construct_inputs(
        training_data=dataset,
        **model_config.model_options,
        **model_kwargs_from_ss,
    )
    submodules = _construct_submodules(
        model_config=model_config,
        dataset=dataset,
        # This is used when constructing the input transforms.
        search_space_digest=search_space_digest,
        # Used to check for supported arguments and in covar module input constructors.
        botorch_model_class=botorch_model_class,
    )
    formatted_model_inputs.update(submodules)
    return formatted_model_inputs


@submodel_input_constructor.register(MultiTaskGP)
def _submodel_input_constructor_mtgp(
    botorch_model_class: type[Model],
    model_config: ModelConfig,
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    surrogate: Surrogate,
) -> dict[str, Any]:
    if len(dataset.outcome_names) > 1:
        raise NotImplementedError("Multi-output Multi-task GPs are not yet supported.")
    formatted_model_inputs = _submodel_input_constructor_base(
        botorch_model_class=botorch_model_class,
        model_config=model_config,
        dataset=dataset,
        search_space_digest=search_space_digest,
        surrogate=surrogate,
    )
    task_feature = formatted_model_inputs.get("task_feature")
    if task_feature is None:
        return formatted_model_inputs
    # specify output tasks so that model.num_outputs = 1
    # since the model only models a single outcome
    if formatted_model_inputs.get("output_tasks") is None:
        # SSD doesn't use -1, so we need to normalize here
        task_feature = none_throws(
            normalize_indices(indices=[task_feature], d=len(dataset.feature_names))
        )[0]
        if (search_space_digest.target_values is not None) and (
            target_value := search_space_digest.target_values.get(task_feature)
        ) is not None:
            formatted_model_inputs["output_tasks"] = [int(target_value)]
            # This enables making predictions for inputs at unobserved task values,
            # by making predictions for the target task.
            # This is important for MTGP models that are used in ModelListGPs where
            # some metrics have only been observed for some tasks and not others.
            formatted_model_inputs["validate_task_values"] = False
            formatted_model_inputs["all_tasks"] = get_all_task_values_from_ssd(
                search_space_digest=search_space_digest
            )
        else:
            raise UserInputError(
                "output_tasks or target task value must be provided for MultiTaskGP."
            )
    return formatted_model_inputs
