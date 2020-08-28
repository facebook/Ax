#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from inspect import isabstract
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.types import TCandidateMetadata, TConfig
from ax.models.model_utils import best_in_sample_point
from ax.models.torch.utils import (
    _to_inequality_constraints,
    pick_best_out_of_sample_point_acqf_class,
    predict_from_model,
)
from ax.utils.common.constants import Keys
from ax.utils.common.equality import Base
from ax.utils.common.typeutils import checked_cast, checked_cast_optional, not_none
from botorch.fit import fit_gpytorch_model
from botorch.models.model import Model, TrainingData
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor


class Surrogate(Base):
    """
    **All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Ax wrapper for BoTorch `Model`, subcomponent of `BoTorchModel`
    and is not meant to be used outside of it.

    Args:
        botorch_model_class: `Model` class to be used as the underlying
            BoTorch model.
        mll_class: `MarginalLogLikelihood` class to use for model-fitting.
        kernel_class: `Kernel` class, not yet used. Will be used to
            construct custom BoTorch `Model` in the future.
        kernel_options: Kernel kwargs, not yet used. Will be used to
            construct custom BoTorch `Model` in the future.
        likelihood: `Likelihood` class, not yet used. Will be used to
            construct custom BoTorch `Model` in the future.
    """

    botorch_model_class: Type[Model]
    mll_class: Type[MarginalLogLikelihood]
    kernel_class: Optional[Type[Kernel]] = None
    _training_data: Optional[TrainingData] = None
    _model: Optional[Model] = None
    # Special setting for surrogates instantiated via `Surrogate.from_BoTorch`,
    # to avoid re-constructing the underlying BoTorch model on `Surrogate.fit`
    # when set to `False`.
    _should_reconstruct: bool = True

    def __init__(
        self,
        # TODO: make optional when BoTorch model factory is checked in.
        # Construction will then be possible from likelihood, kernel, etc.
        botorch_model_class: Type[Model],
        mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood,
        kernel_class: Optional[Type[Kernel]] = None,  # TODO: use.
        kernel_options: Optional[Dict[str, Any]] = None,  # TODO: use.
        likelihood: Optional[Type[Likelihood]] = None,  # TODO: use.
    ) -> None:
        self.botorch_model_class = botorch_model_class
        self.mll_class = mll_class
        # NOTE: assuming here that plugging in kernels will be made easier on the
        # BoTorch side (for v0 can just always raise `NotImplementedError` if
        # `kernel` kwarg is not None).
        if kernel_class:
            self.kernel_class = kernel_class
            # NOTE: `validate_kernel_class` to be implemented on BoTorch `Model`.
            # self.botorch_model_class.validate_kernel_class(kernel_class)

        # Temporary validation while we develop these customizations.
        if likelihood is not None:
            raise NotImplementedError("Customizing likelihood not yet implemented.")
        if kernel_class is not None or kernel_options:
            raise NotImplementedError("Customizing kernel not yet implemented.")

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError("BoTorch `Model` has not yet been constructed.")
        return not_none(self._model)

    @property
    def training_data(self) -> TrainingData:
        if self._training_data is None:
            raise ValueError(
                "Underlying BoTorch `Model` has not yet received its training_data."
            )
        return not_none(self._training_data)

    @property
    def dtype(self) -> torch.dtype:
        return self.training_data.Xs[0].dtype

    @property
    def device(self) -> torch.device:
        return self.training_data.Xs[0].device

    @classmethod
    def from_BoTorch(
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
        surrogate._should_reconstruct = False
        return surrogate

    def construct(
        self, training_data: TrainingData, fidelity_features: List[int]
    ) -> None:
        # NOTE: `validate_training_data` to be implemented on BoTorch `Model`.
        # self.botorch_model_class.validate_training_data(training_data)
        self._training_data = training_data

        if isabstract(self.botorch_model_class):
            raise TypeError("Cannot construct an abstract model.")

        formatted_model_inputs = self.botorch_model_class.construct_inputs(
            training_data=training_data, fidelity_features=fidelity_features
        )
        # pyre-ignore[45]: Model isn't abstract per the check above.
        self._model = self.botorch_model_class(**formatted_model_inputs)
        # TODO: Instantiate / pass kernel here somewhere.

    def fit(
        self,
        training_data: TrainingData,
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        target_fidelities: Optional[Dict[int, float]] = None,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        if self._model is None or self._should_reconstruct:
            self.construct(
                training_data=training_data, fidelity_features=fidelity_features
            )
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        if state_dict is None or refit:
            # pyre-ignore[16]: Model has no attribute likelihood.
            # All BoTorch `Model`-s expected to work with this setup have likelihood.
            mll = self.mll_class(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predicts outcomes given a model and input tensor.

        Args:
            model: A botorch Model.
            X: A `n x d` tensor of input parameters.

        Returns:
            Tensor: The predicted posterior mean as an `n x o`-dim tensor.
            Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
        """
        return predict_from_model(model=self.model, X=X)

    def best_in_sample_point(
        self,
        bounds: List[Tuple[float, float]],
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
            Xs=self.training_data.Xs,
            # pyre-ignore[6]: `best_in_sample_point` currently expects a `TorchModel`
            # or a `NumpyModel` as `model` kwarg, but only uses them for `predict`
            # function, the signature for which is the same on this `Surrogate`.
            # TODO: When we move `botorch_modular` directory to OSS, we will extend
            # the annotation for `model` kwarg to accept `Surrogate` too.
            model=self,
            bounds=bounds,
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
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        fidelity_features: Optional[List[int]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
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
            raise NotImplementedError("Fixed features not yet implemented.")

        options = options or {}
        acqf_class, acqf_options = pick_best_out_of_sample_point_acqf_class(
            Xs=self.training_data.Xs,
            outcome_constraints=outcome_constraints,
            seed_inner=checked_cast_optional(int, options.get(Keys.SEED_INNER, None)),
            qmc=checked_cast(bool, options.get(Keys.QMC, True)),
        )

        # Avoiding circular import between `Surrogate` and `Acquisition`.
        from ax.models.torch.botorch_modular.acquisition import Acquisition

        acqf = Acquisition(  # TODO: For multi-fidelity, might need diff. class.
            surrogate=self,
            botorch_acqf_class=acqf_class,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            options=acqf_options,
        )
        candidates, acqf_values = acqf.optimize(
            # pyre-ignore[6]: Exp. Tensor, got List[Tuple[float, float]].
            # TODO: Fix typing of `bounds` in `TorchModel`-s.
            bounds=bounds,
            n=1,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=linear_constraints
            ),
            fixed_features=fixed_features,
        )
        return candidates[0], acqf_values[0]

    def pareto_frontier(self) -> Tuple[Tensor, Tensor]:
        """For multi-objective optimization, retrieve Pareto frontier instead
        of best point.

        Returns: A two-tuple of:
            - tensor of points in the feature space,
            - tensor of corresponding (multiple) outcomes.
        """
        pass

    def compute_diagnostics(self) -> Dict[str, Any]:
        """Computes model diagnostics like cross-validation measure of fit, etc.
        """
        return {}

    def update(
        self,
        training_data: TrainingData,
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        """Updates the surrogate model with new data.

        Args:
            training_data: Surrogate training_data containing all the data the model
                should use for inference. NOTE: this should not be just the new data
                since the last time the model was updated, but all available
                data.
            refit: Whether to re-optimize model parameters or just add the new
                data to data used for inference.
        """
        # NOTE: In the future, could have `incremental` kwarg, in which case
        # `training_data` could contain just the new data.
        state_dict = self.model.state_dict
        if not self._should_reconstruct:
            raise NotImplementedError(
                "`update` not yet implemented for models that should "
                "not be re-constructed."
            )
        self.fit(
            training_data=training_data,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            metric_names=metric_names,
            fidelity_features=fidelity_features,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
        )
