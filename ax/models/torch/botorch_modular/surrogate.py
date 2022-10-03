#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import inspect
import warnings
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.exceptions.core import AxWarning, UnsupportedError, UserInputError
from ax.models.model_utils import best_in_sample_point
from ax.models.torch.botorch_modular.utils import (
    disable_one_to_many_transforms,
    fit_botorch_model,
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
from ax.utils.common.typeutils import checked_cast, checked_cast_optional, not_none
from botorch.models.model import Model
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import InputPerturbation, InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.datasets import FixedNoiseDataset, RankingDataset, SupervisedDataset
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


class Surrogate(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch ``Model``, subcomponent of ``BoTorchModel``
    and is not meant to be used outside of it.

    Args:
        botorch_model_class: ``Model`` class to be used as the underlying
            BoTorch model.
        model_options: Dictionary of options / kwargs for the BoTorch
            ``Model`` constructed during ``Surrogate.fit``.
        mll_class: ``MarginalLogLikelihood`` class to use for model-fitting.
        mll_options: Dictionary of options / kwargs for the MLL.
        outcome_transform: BoTorch outcome transforms. Passed down to the
            BoTorch ``Model``. Multiple outcome transforms can be chained
            together using ``ChainedOutcomeTransform``.
        input_transform: BoTorch input transforms. Passed down to the
            BoTorch ``Model``. Multiple input transforms can be chained
            together using ``ChainedInputTransform``.
        covar_module_class: Covariance module class, not yet used. Will be
            used to construct custom BoTorch ``Model`` in the future.
        covar_module_options: Covariance module kwargs, not yet used. Will be
            used to construct custom BoTorch ``Model`` in the future.
        likelihood: ``Likelihood`` class, not yet used. Will be used to
            construct custom BoTorch ``Model`` in the future.
        likelihood_options: Likelihood options, not yet used. Will be used to
            construct custom BoTorch ``Model`` in the future.
    """

    botorch_model_class: Type[Model]
    model_options: Dict[str, Any]
    mll_class: Type[MarginalLogLikelihood]
    mll_options: Dict[str, Any]
    outcome_transform: Optional[OutcomeTransform] = None
    input_transform: Optional[InputTransform] = None
    covar_module_class: Optional[Type[Kernel]] = None
    covar_module_options: Dict[str, Any]
    likelihood_class: Optional[Type[Likelihood]] = None
    likelihood_options: Dict[str, Any]
    _training_data: Optional[List[SupervisedDataset]] = None
    _outcomes: Optional[List[str]] = None
    _model: Optional[Model] = None
    # Special setting for surrogates instantiated via `Surrogate.from_botorch`,
    # to avoid re-constructing the underlying BoTorch model on `Surrogate.fit`
    # when set to `False`.
    _constructed_manually: bool = False

    def __init__(
        self,
        # TODO: make optional when BoTorch model factory is checked in.
        # Construction will then be possible from likelihood, kernel, etc.
        botorch_model_class: Type[Model],
        model_options: Optional[Dict[str, Any]] = None,
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        mll_options: Optional[Dict[str, Any]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        covar_module_class: Optional[Type[Kernel]] = None,
        covar_module_options: Optional[Dict[str, Any]] = None,
        likelihood_class: Optional[Type[Likelihood]] = None,
        likelihood_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.botorch_model_class = botorch_model_class
        self.model_options = model_options or {}
        self.mll_class = mll_class
        self.mll_options = mll_options or {}
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        self.covar_module_class = covar_module_class
        self.covar_module_options = covar_module_options or {}
        self.likelihood_class = likelihood_class
        self.likelihood_options = likelihood_options or {}

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError(
                "BoTorch `Model` has not yet been constructed, please fit the "
                "surrogate first (done via `BoTorchModel.fit`)."
            )
        return not_none(self._model)

    @property
    def training_data(self) -> List[SupervisedDataset]:
        if self._training_data is None:
            raise ValueError(NOT_YET_FIT_MSG)
        return not_none(self._training_data)

    @property
    def Xs(self) -> List[Tensor]:
        # Handles multi-output models. TODO: Improve this!
        training_data = self.training_data
        Xs = []
        for dataset in training_data:
            if self.botorch_model_class == PairwiseGP and isinstance(
                dataset, RankingDataset
            ):
                Xi = dataset.X.values
            else:
                Xi = dataset.X()
            for _ in range(dataset.Y.shape[-1]):
                Xs.append(Xi)
        return Xs

    @property
    def dtype(self) -> torch.dtype:
        return self.training_data[0].X.dtype

    @property
    def device(self) -> torch.device:
        return self.training_data[0].X.device

    @classmethod
    def from_botorch(
        cls,
        model: Model,
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
    ) -> Surrogate:
        """Instantiate a `Surrogate` from a pre-instantiated Botorch `Model`."""
        surrogate = cls(botorch_model_class=model.__class__, mll_class=mll_class)
        surrogate._model = model
        # Temporarily disallowing `update` for surrogates instantiated from
        # pre-made BoTorch `Model` instances to avoid reconstructing models
        # that were likely pre-constructed for a reason (e.g. if this setup
        # doesn't fully allow to constuct them).
        surrogate._constructed_manually = True
        return surrogate

    def clone_reset(self) -> Surrogate:
        return self.__class__(**self._serialize_attributes_as_kwargs())

    def construct(self, datasets: List[SupervisedDataset], **kwargs: Any) -> None:
        """Constructs the underlying BoTorch ``Model`` using the training data.

        Args:
            training_data: Training data for the model (for one outcome for
                the default `Surrogate`, with the exception of batched
                multi-output case, where training data is formatted with just
                one X and concatenated Ys).
            **kwargs: Optional keyword arguments, expects any of:
                - "fidelity_features": Indices of columns in X that represent
                fidelity.
        """
        if self._constructed_manually:
            logger.warning("Reconstructing a manually constructed `Model`.")
        if not len(datasets) == 1:
            raise ValueError(  # pragma: no cover
                "Base `Surrogate` expects training data for single outcome."
            )
        input_constructor_kwargs = {**self.model_options, **(kwargs or {})}
        dataset = datasets[0]
        botorch_model_class_args = inspect.getfullargspec(self.botorch_model_class).args

        # Temporary workaround to allow models to consume data from
        # `FixedNoiseDataset`s even if they don't accept variance observations
        if "train_Yvar" not in botorch_model_class_args and isinstance(
            dataset, FixedNoiseDataset
        ):
            warnings.warn(
                f"Provided model class {self.botorch_model_class} does not accept "
                "`train_Yvar` argument, but received `FixedNoiseDataset`. Ignoring "
                "variance observations and converting to `SupervisedDataset`.",
                AxWarning,
            )
            dataset = SupervisedDataset(X=dataset.X(), Y=dataset.Y())

        self._training_data = [dataset]

        formatted_model_inputs = self.botorch_model_class.construct_inputs(
            training_data=dataset, **input_constructor_kwargs
        )
        self._set_formatted_inputs(
            formatted_model_inputs=formatted_model_inputs,
            inputs=[
                [
                    "covar_module",
                    self.covar_module_class,
                    self.covar_module_options,
                    None,
                ],
                ["likelihood", self.likelihood_class, self.likelihood_options, None],
                ["outcome_transform", None, None, self.outcome_transform],
                ["input_transform", None, None, self.input_transform],
            ],
            dataset=dataset,
            botorch_model_class_args=botorch_model_class_args,
            robust_digest=kwargs.get("robust_digest", None),
        )
        # pyre-ignore [45]
        self._model = self.botorch_model_class(**formatted_model_inputs)

    def _set_formatted_inputs(
        self,
        formatted_model_inputs: Dict[str, Any],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        inputs: List[List[Any]],
        dataset: SupervisedDataset,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        botorch_model_class_args: Any,
        robust_digest: Optional[Dict[str, Any]] = None,
    ) -> None:
        for input_name, input_class, input_options, input_object in inputs:
            if input_class is None and input_object is None:
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
            if input_class is not None and input_object is not None:  # pragma: no cover
                raise RuntimeError(f"Got both a class and an object for {input_name}.")
            if input_class is not None:
                input_options = input_options or {}
                formatted_model_inputs[input_name] = input_class(**input_options)
            else:
                formatted_model_inputs[input_name] = input_object

        # Construct input perturbation if doing robust optimization.
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
            if formatted_model_inputs.get("input_transform") is not None:
                # TODO: Support mixing with user supplied transforms.
                raise NotImplementedError(
                    "User supplied input transforms are not supported "
                    "in robust optimization."
                )
            else:
                formatted_model_inputs["input_transform"] = perturbation

    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
        original_metric_names: Optional[List[str]] = None,
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
            metric_names: A list of metric names, with the i-th metric
                corresponding to the i-th dataset.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: Optional state dict to load.
            refit: Whether to re-optimize model parameters.
            # TODO: we should refactor the fit() API to get rid of the metric_names
            # and the concatenation hack that comes with it in BoTorchModel.fit()
            # by attaching the individual metric_name to each dataset directly.
            original_metric_names: sometimes the original list of metric_names
                got tranformed into a different format before being passed down
                into fit(). This arg preserves the original metric_names before
                the transformation.
        """
        if self._constructed_manually:
            logger.debug(
                "For manually constructed surrogates (via `Surrogate.from_botorch`), "
                "`fit` skips setting the training data on model and only reoptimizes "
                "its parameters if `refit=True`."
            )
        else:
            self.construct(
                datasets=datasets,
                metric_names=metric_names,
                **dataclasses.asdict(search_space_digest),
            )
            self._outcomes = (
                original_metric_names
                if original_metric_names is not None
                else metric_names
            )
        if state_dict:
            self.model.load_state_dict(not_none(state_dict))

        if state_dict is None or refit:
            fit_botorch_model(
                model=self.model, mll_class=self.mll_class, mll_options=self.mll_options
            )

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicts outcomes given a model and input tensor.

        Args:
            model: A botorch Model.
            X: A ``n x d`` tensor of input parameters.

        Returns:
            Tensor: The predicted posterior mean as an ``n x o``-dim tensor.
            Tensor: The predicted posterior covariance as a ``n x o x o``-dim tensor.
        """
        # This temporarily disables the one-to-many transforms to avoid perturbing
        # the user supplied parameterization.
        with disable_one_to_many_transforms(self.model):
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
            # pyre-ignore[6]: `best_in_sample_point` currently expects a `TorchModel`
            # as `model` kwarg, but only uses them for `predict` function, the
            # signature for which is the same on this `Surrogate`.
            # TODO: When we move `botorch_modular` directory to OSS, we will extend
            # the annotation for `model` kwarg to accept `Surrogate` too.
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
            surrogate=self,
            botorch_acqf_class=acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acqf_options,
        )
        candidates, acqf_values = acqf.optimize(
            n=1,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
        )
        return candidates[0], acqf_values[0]

    def update(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        """Updates the surrogate model with new data. In the base ``Surrogate``,
        just calls ``fit`` after checking that this surrogate was not created
        via ``Surrogate.from_botorch`` (in which case the ``Model`` comes premade,
        constructed manually and then supplied to ``Surrogate``).

        NOTE: Expects `training_data` to be all available data,
        not just the new data since the last time the model was updated.

        Args:
            training_data: Surrogate training_data containing all the data the model
                should use for inference.
            search_space_digest: A SearchSpaceDigest object containing
                metadata on the features in the training data.
            metric_names: Names of each outcome Y in Ys.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: Optional state dict to load.
            refit: Whether to re-optimize model parameters or just set the training
                data used for interence to new training data.
        """
        # NOTE: In the future, could have `incremental` kwarg, in which case
        # `training_data` could contain just the new data.
        if self._constructed_manually:
            raise NotImplementedError(
                "`update` not yet implemented for models that are "
                "constructed manually, but it is possible to create a new "
                "surrogate in the same way as the current manually constructed one, "
                "via `Surrogate.from_botorch`."
            )
        self.fit(
            datasets=datasets,
            metric_names=metric_names,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
        )

    def pareto_frontier(self) -> Tuple[Tensor, Tensor]:
        """For multi-objective optimization, retrieve Pareto frontier instead
        of best point.

        Returns: A two-tuple of:
            - tensor of points in the feature space,
            - tensor of corresponding (multiple) outcomes.
        """
        raise NotImplementedError(
            "Pareto frontier not yet implemented."
        )  # pragma: no cover

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Computes model diagnostics like cross-validation measure of fit, etc."""
        return {}  # pragma: no cover

    def _serialize_attributes_as_kwargs(self) -> Dict[str, Any]:
        """Serialize attributes of this surrogate, to be passed back to it
        as kwargs on reinstantiation.
        """
        if self._constructed_manually:
            raise UnsupportedError(
                "Surrogates constructed manually (ie Surrogate.from_botorch) may not "
                "be serialized. If serialization is necessary please initialize from "
                "the constructor."
            )

        return {
            "botorch_model_class": self.botorch_model_class,
            "model_options": self.model_options,
            "mll_class": self.mll_class,
            "mll_options": self.mll_options,
            "outcome_transform": self.outcome_transform,
            "input_transform": self.input_transform,
            "covar_module_class": self.covar_module_class,
            "covar_module_options": self.covar_module_options,
            "likelihood_class": self.likelihood_class,
            "likelihood_options": self.likelihood_options,
        }
