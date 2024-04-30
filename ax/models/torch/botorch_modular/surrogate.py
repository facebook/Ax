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
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.exceptions.core import UserInputError
from ax.models.model_utils import best_in_sample_point
from ax.models.torch.botorch_modular.input_constructors.covar_modules import (
    covar_module_argparse,
)
from ax.models.torch.botorch_modular.input_constructors.input_transforms import (
    input_transform_argparse,
)
from ax.models.torch.botorch_modular.input_constructors.outcome_transform import (
    outcome_transform_argparse,
)
from ax.models.torch.botorch_modular.utils import (
    choose_model_class,
    convert_to_block_design,
    fit_botorch_model,
    subset_state_dict,
    use_model_list,
)
from ax.models.torch.utils import (
    _to_inequality_constraints,
    pick_best_out_of_sample_point_acqf_class,
    predict_from_model,
)
from ax.models.torch_base import TorchOptConfig
from ax.models.types import TConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    _argparse_type_encoder,
    checked_cast,
    checked_cast_optional,
)
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputPerturbation,
    InputTransform,
)
from botorch.models.transforms.outcome import ChainedOutcomeTransform, OutcomeTransform
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

NOT_YET_FIT_MSG = (
    "Underlying BoTorch `Model` has not yet received its training_data. "
    "Please fit the model first."
)

logger: Logger = get_logger(__name__)


def _extract_model_kwargs(
    search_space_digest: SearchSpaceDigest,
) -> Dict[str, Union[List[int], int]]:
    """
    Extracts keyword arguments that are passed to the `construct_inputs`
    method of a BoTorch `Model` class.

    Args:
        search_space_digest: A `SearchSpaceDigest`.

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

    # TODO: Allow each metric having different task_features or fidelity_features
    # TODO: Need upstream change in the modelbrdige
    if len(task_features) > 1:
        raise NotImplementedError("Multiple task features are not supported.")

    kwargs: Dict[str, Union[List[int], int]] = {}
    if len(search_space_digest.categorical_features) > 0:
        kwargs["categorical_features"] = search_space_digest.categorical_features
    if len(fidelity_features) > 0:
        kwargs["fidelity_features"] = fidelity_features
    if len(task_features) == 1:
        kwargs["task_feature"] = task_features[0]
    return kwargs


class Surrogate(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch ``Model``, subcomponent of ``BoTorchModel``
    and is not meant to be used outside of it.

    Args:
        botorch_model_class: ``Model`` class to be used as the underlying
            BoTorch model. If None is provided a model class will be selected (either
            one for all outcomes or a ModelList with separate models for each outcome)
            will be selected automatically based off the datasets at `construct` time.
        model_options: Dictionary of options / kwargs for the BoTorch
            ``Model`` constructed during ``Surrogate.fit``.
            Note that the corresponding attribute will later be updated to include any
            additional kwargs passed into ``BoTorchModel.fit``.
        mll_class: ``MarginalLogLikelihood`` class to use for model-fitting.
        mll_options: Dictionary of options / kwargs for the MLL.
        outcome_transform_classes: List of BoTorch outcome transforms classes. Passed
            down to the BoTorch ``Model``. Multiple outcome transforms can be chained
            together using ``ChainedOutcomeTransform``.
        outcome_transform_options: Outcome transform classes kwargs. The keys are
            class string names and the values are dictionaries of outcome transform
            kwargs. For example,
            `
            outcome_transform_classes = [Standardize]
            outcome_transform_options = {
                "Standardize": {"m": 1},
            `
            For more options see `botorch/models/transforms/outcome.py`.
        input_transform_classes: List of BoTorch input transforms classes.
            Passed down to the BoTorch ``Model``. Multiple input transforms
            will be chained together using ``ChainedInputTransform``.
        input_transform_options: Input transform classes kwargs. The keys are
            class string names and the values are dictionaries of input transform
            kwargs. For example,
            `
            input_transform_classes = [Normalize, Round]
            input_transform_options = {
                "Normalize": {"d": 3},
                "Round": {"integer_indices": [0], "categorical_features": {1: 2}},
            }
            `
            For more input options see `botorch/models/transforms/input.py`.
        covar_module_class: Covariance module class. This gets initialized after
            parsing the ``covar_module_options`` in ``covar_module_argparse``,
            and gets passed to the model constructor as ``covar_module``.
        covar_module_options: Covariance module kwargs.
        likelihood: ``Likelihood`` class. This gets initialized with
            ``likelihood_options`` and gets passed to the model constructor.
        likelihood_options: Likelihood options.
        allow_batched_models: Set to true to fit the models in a batch if supported.
            Set to false to fit individual models to each metric in a loop.
    """

    def __init__(
        self,
        botorch_model_class: Optional[Type[Model]] = None,
        model_options: Optional[Dict[str, Any]] = None,
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        mll_options: Optional[Dict[str, Any]] = None,
        outcome_transform_classes: Optional[List[Type[OutcomeTransform]]] = None,
        outcome_transform_options: Optional[Dict[str, Dict[str, Any]]] = None,
        input_transform_classes: Optional[List[Type[InputTransform]]] = None,
        input_transform_options: Optional[Dict[str, Dict[str, Any]]] = None,
        covar_module_class: Optional[Type[Kernel]] = None,
        covar_module_options: Optional[Dict[str, Any]] = None,
        likelihood_class: Optional[Type[Likelihood]] = None,
        likelihood_options: Optional[Dict[str, Any]] = None,
        allow_batched_models: bool = True,
    ) -> None:
        self.botorch_model_class = botorch_model_class
        # Copying model options to avoid mutating the original dict.
        # We later update it with any additional kwargs passed into `BoTorchModel.fit`.
        self.model_options: Dict[str, Any] = (model_options or {}).copy()
        self.mll_class = mll_class
        self.mll_options: Dict[str, Any] = mll_options or {}
        self.outcome_transform_classes = outcome_transform_classes
        self.outcome_transform_options: Dict[str, Any] = outcome_transform_options or {}
        self.input_transform_classes = input_transform_classes
        self.input_transform_options: Dict[str, Any] = input_transform_options or {}
        self.covar_module_class = covar_module_class
        self.covar_module_options: Dict[str, Any] = covar_module_options or {}
        self.likelihood_class = likelihood_class
        self.likelihood_options: Dict[str, Any] = likelihood_options or {}
        self.allow_batched_models = allow_batched_models
        # Store the last dataset used to fit the model for a given metric(s).
        # If the new dataset is identical, we will skip model fitting for that metric.
        # The keys are `tuple(dataset.outcome_names)`.
        self._last_datasets: Dict[Tuple[str], SupervisedDataset] = {}
        # Store a reference from a tuple of metric names to the BoTorch Model
        # corresponding to those metrics. In most cases this will be a one-tuple,
        # though we need n-tuples for LCE-M models. This will be used to skip model
        # construction & fitting if the datasets are identical.
        self._submodels: Dict[Tuple[str], Model] = {}
        # Store a reference to search space digest used while fitting the cached models.
        # We will re-fit the models if the search space digest changes.
        self._last_search_space_digest: Optional[SearchSpaceDigest] = None

        # These are later updated during model fitting.
        self._training_data: Optional[List[SupervisedDataset]] = None
        self._outcomes: Optional[List[str]] = None
        self._model: Optional[Model] = None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f" botorch_model_class={self.botorch_model_class} "
            f"mll_class={self.mll_class} "
            f"outcome_transform_classes={self.outcome_transform_classes} "
            f"input_transform_classes={self.input_transform_classes} "
        )

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError(
                "BoTorch `Model` has not yet been constructed, please fit the "
                "surrogate first (done via `BoTorchModel.fit`)."
            )
        return self._model

    @property
    def training_data(self) -> List[SupervisedDataset]:
        if self._training_data is None:
            raise ValueError(NOT_YET_FIT_MSG)
        return self._training_data

    @property
    def Xs(self) -> List[Tensor]:
        # Handles multi-output models. TODO: Improve this!
        training_data = self.training_data
        Xs = []
        for dataset in training_data:
            if self.botorch_model_class == PairwiseGP and isinstance(
                dataset, RankingDataset
            ):
                # directly accessing the d-dim X tensor values
                # instead of the augmented 2*d-dim dataset.X from RankingDataset
                Xi = checked_cast(SliceContainer, dataset._X).values
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
        botorch_model_class: Type[Model],
        state_dict: Optional[OrderedDict[str, Tensor]],
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
            botorch_model_class: ``Model`` class to be used as the underlying
                BoTorch model.
            state_dict: Optional state dict to load. This should be subsetted for
                the current submodel being constructed.
            refit: Whether to re-optimize model parameters.
        """
        outcome_names = tuple(dataset.outcome_names)
        if self._should_reuse_last_model(
            dataset=dataset, botorch_model_class=botorch_model_class
        ):
            return self._submodels[outcome_names]
        formatted_model_inputs = submodel_input_constructor(
            botorch_model_class,  # Do not pass as kwarg since this is used to dispatch.
            dataset=dataset,
            search_space_digest=search_space_digest,
            surrogate=self,
        )
        # pyre-ignore [45]
        model = botorch_model_class(**formatted_model_inputs)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        if state_dict is None or refit:
            fit_botorch_model(
                model=model, mll_class=self.mll_class, mll_options=self.mll_options
            )
        self._submodels[outcome_names] = model
        self._last_datasets[outcome_names] = dataset
        return model

    def _should_reuse_last_model(
        self, dataset: SupervisedDataset, botorch_model_class: Type[Model]
    ) -> bool:
        """Checks whether the given dataset and model class match the last
        dataset and model class used to train the cached sub-model.
        """
        outcome_names = tuple(dataset.outcome_names)
        if (
            outcome_names in self._submodels
            and dataset == self._last_datasets[outcome_names]
        ):
            last_model = self._submodels[outcome_names]
            if type(last_model) is not botorch_model_class:
                logger.info(
                    f"The model class for outcome(s) {dataset.outcome_names} "
                    f"changed from {type(last_model)} to {botorch_model_class}. "
                    "Will refit the model."
                )
            else:
                return True
        return False

    def _set_formatted_inputs(
        self,
        formatted_model_inputs: Dict[str, Any],
        # pyre-ignore [2] The proper hint for the second arg is Union[None,
        # Type[Kernel], Type[Likelihood], List[Type[OutcomeTransform]],
        # List[Type[InputTransform]]]. Keeping it as Any saves us from a
        # bunch of checked_cast calls within the for loop.
        inputs: List[Tuple[str, Any, Dict[str, Any]]],
        dataset: SupervisedDataset,
        botorch_model_class_args: List[str],
        search_space_digest: SearchSpaceDigest,
    ) -> None:
        for input_name, input_class, input_options in inputs:
            if input_class is None:
                continue
            if input_name not in botorch_model_class_args:
                # TODO: We currently only pass in `covar_module` and `likelihood`
                # if they are inputs to the BoTorch model. This interface will need
                # to be expanded to a ModelFactory, see D22457664, to accommodate
                # different models in the future.
                raise UserInputError(
                    f"The BoTorch model class {self.botorch_model_class} does not "
                    f"support the input {input_name}."
                )
            input_options = deepcopy(input_options) or {}

            if input_name == "covar_module":
                covar_module_with_defaults = covar_module_argparse(
                    input_class,
                    dataset=dataset,
                    botorch_model_class=self.botorch_model_class,
                    **input_options,
                )

                formatted_model_inputs[input_name] = input_class(
                    **covar_module_with_defaults
                )

            elif input_name == "input_transform":
                formatted_model_inputs[input_name] = self._make_botorch_input_transform(
                    input_classes=input_class,
                    input_options=input_options,
                    dataset=dataset,
                    search_space_digest=search_space_digest,
                )

            elif input_name == "outcome_transform":
                formatted_model_inputs[input_name] = (
                    self._make_botorch_outcome_transform(
                        input_classes=input_class,
                        input_options=input_options,
                        dataset=dataset,
                    )
                )
            else:
                formatted_model_inputs[input_name] = input_class(**input_options)

    def _make_botorch_input_transform(
        self,
        input_classes: List[Type[InputTransform]],
        dataset: SupervisedDataset,
        search_space_digest: SearchSpaceDigest,
        input_options: Dict[str, Dict[str, Any]],
    ) -> Optional[InputTransform]:
        """
        Makes a BoTorch input transform from the provided input classes and options.
        """
        if not (
            isinstance(input_classes, list)
            and all(issubclass(c, InputTransform) for c in input_classes)
        ):
            raise UserInputError("Expected a list of input transforms.")
        if len(input_classes) == 0:
            return None

        input_transform_kwargs = [
            input_transform_argparse(
                single_input_class,
                dataset=dataset,
                search_space_digest=search_space_digest,
                input_transform_options=input_options.get(
                    single_input_class.__name__, {}
                ),
            )
            for single_input_class in input_classes
        ]

        input_transforms = [
            # pyre-fixme[45]: Cannot instantiate abstract class `InputTransform`.
            single_input_class(**single_input_transform_kwargs)
            for single_input_class, single_input_transform_kwargs in zip(
                input_classes, input_transform_kwargs
            )
        ]

        input_instance = (
            ChainedInputTransform(
                **{f"tf{i}": input_transforms[i] for i in range(len(input_transforms))}
            )
            if len(input_transforms) > 1
            else input_transforms[0]
        )

        return input_instance

    def _make_botorch_outcome_transform(
        self,
        input_classes: List[Type[OutcomeTransform]],
        input_options: Dict[str, Dict[str, Any]],
        dataset: SupervisedDataset,
    ) -> Optional[OutcomeTransform]:
        """
        Makes a BoTorch outcome transform from the provided classes and options.
        """
        if not (
            isinstance(input_classes, list)
            and all(issubclass(c, OutcomeTransform) for c in input_classes)
        ):
            raise UserInputError("Expected a list of outcome transforms.")
        if len(input_classes) == 0:
            return None

        outcome_transform_kwargs = [
            outcome_transform_argparse(
                input_class,
                outcome_transform_options=input_options.get(input_class.__name__, {}),
                dataset=dataset,
            )
            for input_class in input_classes
        ]

        outcome_transforms = [
            # pyre-fixme[45]: Cannot instantiate abstract class `OutcomeTransform`.
            input_class(**single_outcome_transform_kwargs)
            for input_class, single_outcome_transform_kwargs in zip(
                input_classes, outcome_transform_kwargs
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

    def fit(
        self,
        datasets: Sequence[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[OrderedDict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        """Fits the underlying BoTorch ``Model`` to ``m`` outcomes.

        NOTE: ``state_dict`` and ``refit`` keyword arguments control how the
        undelying BoTorch ``Model`` will be fit: whether its parameters will
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
        """
        self._discard_cached_model_and_data_if_search_space_digest_changed(
            search_space_digest=search_space_digest
        )
        # To determine whether to use ModelList under the hood, we need to check for
        # the batched multi-output case, so we first see which model would be chosen
        # given the Yvars and the properties of data.
        botorch_model_class = self.botorch_model_class or choose_model_class(
            datasets=datasets, search_space_digest=search_space_digest
        )

        should_use_model_list = use_model_list(
            datasets=datasets,
            botorch_model_class=botorch_model_class,
            allow_batched_models=self.allow_batched_models,
        )

        if not should_use_model_list and len(datasets) > 1:
            datasets = convert_to_block_design(datasets=datasets, force=True)
        self._training_data = list(datasets)  # So that it can be modified if needed.

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
            model = self._construct_model(
                dataset=dataset,
                search_space_digest=search_space_digest,
                botorch_model_class=botorch_model_class,
                state_dict=submodel_state_dict,
                refit=refit,
            )
            models.append(model)
            outcome_names.extend(dataset.outcome_names)

        if should_use_model_list:
            self._model = ModelListGP(*models)
        else:
            self._model = models[0]
        self._outcomes = outcome_names  # In the order of input datasets

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
            logger.info(
                "Discarding all previously trained models due to a change "
                "in the search space digest."
            )
            self._submodels = {}
            self._last_datasets = {}
        self._last_search_space_digest = search_space_digest

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicts outcomes given an input tensor.

        Args:
            X: A ``n x d`` tensor of input parameters.

        Returns:
            Tensor: The predicted posterior mean as an ``n x o``-dim tensor.
            Tensor: The predicted posterior covariance as a ``n x o x o``-dim tensor.
        """
        return predict_from_model(model=self.model, X=X)

    def best_in_sample_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, float]:
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
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Finds the best predicted point and the corresponding value of the
        appropriate best point acquisition function.
        """
        if torch_opt_config.fixed_features:
            # When have fixed features, need `FixedFeatureAcquisitionFunction`
            # which has peculiar instantiation (wraps another acquisition fn.),
            # so need to figure out how to handle.
            # TODO (ref: https://fburl.com/diff/uneqb3n9)
            raise NotImplementedError("Fixed features not yet supported.")

        options = options or {}
        acqf_class, acqf_options = pick_best_out_of_sample_point_acqf_class(
            outcome_constraints=torch_opt_config.outcome_constraints,
            seed_inner=checked_cast_optional(int, options.get(Keys.SEED_INNER, None)),
            qmc=checked_cast(bool, options.get(Keys.QMC, True)),
            risk_measure=torch_opt_config.risk_measure,
        )

        # Avoiding circular import between `Surrogate` and `Acquisition`.
        from ax.models.torch.botorch_modular.acquisition import Acquisition

        acqf = Acquisition(  # TODO: For multi-fidelity, might need diff. class.
            surrogates={"self": self},
            botorch_acqf_class=acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acqf_options,
        )
        candidates, acqf_values, _ = acqf.optimize(
            n=1,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
        )
        return candidates[0], acqf_values[0]

    def pareto_frontier(self) -> Tuple[Tensor, Tensor]:
        """For multi-objective optimization, retrieve Pareto frontier instead
        of best point.

        Returns: A two-tuple of:
            - tensor of points in the feature space,
            - tensor of corresponding (multiple) outcomes.
        """
        raise NotImplementedError("Pareto frontier not yet implemented.")

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Computes model diagnostics like cross-validation measure of fit, etc."""
        return {}

    def _serialize_attributes_as_kwargs(self) -> Dict[str, Any]:
        """Serialize attributes of this surrogate, to be passed back to it
        as kwargs on reinstantiation.
        """
        return {
            "botorch_model_class": self.botorch_model_class,
            "model_options": self.model_options,
            "mll_class": self.mll_class,
            "mll_options": self.mll_options,
            "outcome_transform_classes": self.outcome_transform_classes,
            "outcome_transform_options": self.outcome_transform_options,
            "input_transform_classes": self.input_transform_classes,
            "input_transform_options": self.input_transform_options,
            "covar_module_class": self.covar_module_class,
            "covar_module_options": self.covar_module_options,
            "likelihood_class": self.likelihood_class,
            "likelihood_options": self.likelihood_options,
            "allow_batched_models": self.allow_batched_models,
        }

    def _extract_construct_input_transform_args(
        self, search_space_digest: SearchSpaceDigest
    ) -> Tuple[Optional[List[Type[InputTransform]]], Dict[str, Dict[str, Any]]]:
        """
        Extracts input transform classes and input transform options that will
        be used in `self._set_formatted_inputs` and ultimately passed to
        BoTorch.

        Args:
            search_space_digest: A `SearchSpaceDigest`.

        Returns:
            A tuple containing
                - Either `None` or a list of input transform classes,
                - A dictionary of input transform options.
        """

        # Construct input perturbation if doing robust optimization.
        # NOTE: Doing this here rather than in `_set_formatted_inputs` to make sure
        # we use the same perturbations for each sub-model.
        if (robust_digest := search_space_digest.robust_digest) is not None:
            submodel_input_transform_options = {
                "InputPerturbation": input_transform_argparse(
                    InputTransform,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=[], bounds=[], robust_digest=robust_digest
                    ),
                )
            }

            submodel_input_transform_classes: List[Type[InputTransform]] = [
                InputPerturbation
            ]

            if self.input_transform_classes is not None:
                # TODO: Support mixing with user supplied transforms.
                raise NotImplementedError(
                    "User supplied input transforms are not supported "
                    "in robust optimization."
                )
        else:
            submodel_input_transform_classes = self.input_transform_classes
            submodel_input_transform_options = self.input_transform_options

        return (
            submodel_input_transform_classes,
            submodel_input_transform_options,
        )

    @property
    def outcomes(self) -> List[str]:
        if self._outcomes is None:
            raise RuntimeError("outcomes not initialized. Please call `fit` first.")
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: List[str]) -> None:
        raise RuntimeError("Setting outcomes manually is disallowed.")


submodel_input_constructor = Dispatcher(
    name="submodel_input_constructor", encoder=_argparse_type_encoder
)


@submodel_input_constructor.register(Model)
def _submodel_input_constructor_base(
    botorch_model_class: Type[Model],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    surrogate: Surrogate,
) -> Dict[str, Any]:
    """Construct the inputs required to initialize a BoTorch model.

    Args:
        botorch_model_class: The BoTorch model class to instantiate.
        dataset: The training data for the model.
        search_space_digest: Search space digest used to set up model arguments.
        surrogate: A reference to the surrogate that created the model.
            This can be used by the constructor to obtain any additional
            arguments that are not readily available.

    Returns:
        A dictionary of inputs for constructing the model.
    """
    model_kwargs_from_ss = _extract_model_kwargs(
        search_space_digest=search_space_digest
    )

    (
        input_transform_classes,
        input_transform_options,
    ) = surrogate._extract_construct_input_transform_args(
        search_space_digest=search_space_digest
    )

    formatted_model_inputs = botorch_model_class.construct_inputs(
        training_data=dataset,
        **surrogate.model_options,
        **model_kwargs_from_ss,
    )

    botorch_model_class_args = inspect.getfullargspec(botorch_model_class).args
    surrogate._set_formatted_inputs(
        formatted_model_inputs=formatted_model_inputs,
        inputs=[
            (
                "covar_module",
                surrogate.covar_module_class,
                surrogate.covar_module_options,
            ),
            ("likelihood", surrogate.likelihood_class, surrogate.likelihood_options),
            (
                "outcome_transform",
                surrogate.outcome_transform_classes,
                surrogate.outcome_transform_options,
            ),
            (
                "input_transform",
                input_transform_classes,
                deepcopy(input_transform_options),
            ),
        ],
        dataset=dataset,
        # This is used when constructing the input transforms.
        search_space_digest=search_space_digest,
        # This is used to check if the arguments are supported.
        botorch_model_class_args=botorch_model_class_args,
    )
    return formatted_model_inputs


@submodel_input_constructor.register(MultiTaskGP)
def _submodel_input_constructor_mtgp(
    botorch_model_class: Type[Model],
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
    surrogate: Surrogate,
) -> Dict[str, Any]:
    if len(dataset.outcome_names) > 1:
        raise NotImplementedError("Multi-output Multi-task GPs are not yet supported.")
    formatted_model_inputs = _submodel_input_constructor_base(
        botorch_model_class=botorch_model_class,
        dataset=dataset,
        search_space_digest=search_space_digest,
        surrogate=surrogate,
    )
    task_feature = formatted_model_inputs.get("task_feature")
    if task_feature is None:
        return formatted_model_inputs
    # specify output tasks so that model.num_outputs = 1
    # since the model only models a single outcome
    formatted_model_inputs["output_tasks"] = dataset.X[:1, task_feature].tolist()
    return formatted_model_inputs
