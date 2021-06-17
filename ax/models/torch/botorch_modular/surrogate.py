#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TConfig
from ax.models.model_utils import best_in_sample_point
from ax.models.torch.utils import (
    _to_inequality_constraints,
    pick_best_out_of_sample_point_acqf_class,
    predict_from_model,
)
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, checked_cast_optional, not_none
from botorch.fit import fit_gpytorch_model
from botorch.models.model import Model
from botorch.utils.containers import TrainingData
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
        mll_class: ``MarginalLogLikelihood`` class to use for model-fitting.
        model_options: Dictionary of options / kwargs for the BoTorch
            ``Model`` constructed during ``Surrogate.fit``.
        kernel_class: ``Kernel`` class, not yet used. Will be used to
            construct custom BoTorch ``Model`` in the future.
        kernel_options: Kernel kwargs, not yet used. Will be used to
            construct custom BoTorch ``Model`` in the future.
        likelihood: ``Likelihood`` class, not yet used. Will be used to
            construct custom BoTorch ``Model`` in the future.
    """

    botorch_model_class: Type[Model]
    mll_class: Type[MarginalLogLikelihood]
    model_options: Dict[str, Any]
    kernel_class: Optional[Type[Kernel]] = None
    _training_data: Optional[TrainingData] = None
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
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        model_options: Optional[Dict[str, Any]] = None,
        kernel_class: Optional[Type[Kernel]] = None,  # TODO: use.
        kernel_options: Optional[Dict[str, Any]] = None,  # TODO: use.
        likelihood: Optional[Type[Likelihood]] = None,  # TODO: use.
    ) -> None:
        self.botorch_model_class = botorch_model_class
        self.mll_class = mll_class
        self.model_options = model_options or {}

        # Temporary validation while we develop these customizations.
        if likelihood is not None:
            raise NotImplementedError("Customizing likelihood not yet implemented.")
        if kernel_class is not None or kernel_options:
            raise NotImplementedError("Customizing kernel not yet implemented.")

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError(
                "BoTorch `Model` has not yet been constructed, please fit the "
                "surrogate first (done via `BoTorchModel.fit`)."
            )
        return not_none(self._model)

    @property
    def training_data(self) -> TrainingData:
        if self._training_data is None:
            raise ValueError(NOT_YET_FIT_MSG)
        return not_none(self._training_data)

    @property
    def training_data_per_outcome(self) -> Dict[str, TrainingData]:
        raise NotImplementedError(  # pragma: no cover
            "`training_data_per_outcome` is only used in `ListSurrogate`."
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.training_data.X.dtype

    @property
    def device(self) -> torch.device:
        return self.training_data.X.device

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

    def construct(self, training_data: TrainingData, **kwargs: Any) -> None:
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
        if not isinstance(training_data, TrainingData):
            raise ValueError(  # pragma: no cover
                "Base `Surrogate` expects training data for single outcome."
            )

        input_constructor_kwargs = {**self.model_options, **(kwargs or {})}
        self._training_data = training_data

        formatted_model_inputs = self.botorch_model_class.construct_inputs(
            training_data=self.training_data, **input_constructor_kwargs
        )
        # pyre-ignore[45]: Py raises informative msg if `model_cls` abstract.
        self._model = self.botorch_model_class(**formatted_model_inputs)

    def fit(
        self,
        training_data: TrainingData,
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        """Fits the underlying BoTorch ``Model`` to ``m`` outcomes.

        NOTE: ``state_dict`` and ``refit`` keyword arguments control how the
        undelying BoTorch ``Model`` will be fit: whether its parameters will
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
            training data: BoTorch ``TrainingData`` container with Xs, Ys, and
                possibly Yvars, to be passed to ``Model.construct_inputs`` in
                BoTorch.
            search_space_digest: A SearchSpaceDigest object containing
                metadata on the features in the trainig data.
            metric_names: Names of each outcome Y in Ys.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: Optional state dict to load.
            refit: Whether to re-optimize model parameters.
        """
        if self._constructed_manually:
            logger.debug(
                "For manually constructed surrogates (via `Surrogate.from_botorch`), "
                "`fit` skips setting the training data on model and only reoptimizes "
                "its parameters if `refit=True`."
            )
        else:
            self.construct(
                training_data=training_data,
                metric_names=metric_names,
                **dataclasses.asdict(search_space_digest)
            )
        if state_dict:
            # pyre-fixme[6]: Expected `OrderedDict[typing.Any, typing.Any]` for 1st
            #  param but got `Dict[str, Tensor]`.
            self.model.load_state_dict(not_none(state_dict))

        if state_dict is None or refit:
            mll = self.mll_class(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicts outcomes given a model and input tensor.

        Args:
            model: A botorch Model.
            X: A ``n x d`` tensor of input parameters.

        Returns:
            Tensor: The predicted posterior mean as an ``n x o``-dim tensor.
            Tensor: The predicted posterior covariance as a ``n x o x o``-dim tensor.
        """
        return predict_from_model(model=self.model, X=X)

    def best_in_sample_point(
        self,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Optional[Tensor],
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, float]:
        """Finds the best observed point and the corresponding observed outcome
        values.
        """
        best_point_and_observed_value = best_in_sample_point(
            Xs=[self.training_data.X],
            # pyre-ignore[6]: `best_in_sample_point` currently expects a `TorchModel`
            # or a `NumpyModel` as `model` kwarg, but only uses them for `predict`
            # function, the signature for which is the same on this `Surrogate`.
            # TODO: When we move `botorch_modular` directory to OSS, we will extend
            # the annotation for `model` kwarg to accept `Surrogate` too.
            model=self,
            bounds=search_space_digest.bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=options,
        )
        if best_point_and_observed_value is None:
            raise ValueError("Could not obtain best in-sample point.")
        best_point, observed_value = best_point_and_observed_value
        return checked_cast(Tensor, best_point), observed_value

    def best_out_of_sample_point(
        self,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[TConfig] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Finds the best predicted point and the corresponding value of the
        appropriate best point acquisition function.
        """
        if fixed_features:
            # When have fixed features, need `FixedFeatureAcquisitionFunction`
            # which has peculiar instantiation (wraps another acquisition fn.),
            # so need to figure out how to handle.
            # TODO (ref: https://fburl.com/diff/uneqb3n9)
            raise NotImplementedError("Fixed features not yet supported.")

        options = options or {}
        acqf_class, acqf_options = pick_best_out_of_sample_point_acqf_class(
            outcome_constraints=outcome_constraints,
            seed_inner=checked_cast_optional(int, options.get(Keys.SEED_INNER, None)),
            qmc=checked_cast(bool, options.get(Keys.QMC, True)),
        )

        # Avoiding circular import between `Surrogate` and `Acquisition`.
        from ax.models.torch.botorch_modular.acquisition import Acquisition

        acqf = Acquisition(  # TODO: For multi-fidelity, might need diff. class.
            surrogate=self,
            botorch_acqf_class=acqf_class,
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=acqf_options,
        )
        candidates, acqf_values = acqf.optimize(
            n=1,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=linear_constraints
            ),
            fixed_features=fixed_features,
        )
        return candidates[0], acqf_values[0]

    def update(
        self,
        training_data: TrainingData,
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
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
            training_data=training_data,
            search_space_digest=search_space_digest,
            metric_names=metric_names,
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
        return {
            "botorch_model_class": self.botorch_model_class,
            "mll_class": self.mll_class,
            "model_options": self.model_options,
        }
