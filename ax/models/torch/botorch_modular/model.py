#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.models.torch.botorch import get_rounding_func
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    use_model_list,
    validate_data_format,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchModel
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.containers import TrainingData
from torch import Tensor


class BoTorchModel(TorchModel, Base):
    """**All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Modular `Model` class for combining BoTorch subcomponents
    in Ax. Specified via `Surrogate` and `Acquisition`, which wrap
    BoTorch `Model` and `AcquisitionFunction`, respectively, for
    convenient use in Ax.

    Args:
        acquisition_class: Type of `Acquisition` to be used in
            this model, auto-selected based on experiment and data
            if not specified.
        acquisition_options: Optional dict of kwargs, passed to
            the constructor of BoTorch `AcquisitionFunction`.
        botorch_acqf_class: Type of `AcquisitionFunction` to be
            used in this model, auto-selected based on experiment
            and data if not specified.
        surrogate: An instance of `Surrogate` to be used as part of
            this model; if not specified, type of `Surrogate` and
            underlying BoTorch `Model` will be auto-selected based
            on experiment and data, with kwargs in `surrogate_options`
            applied.
        surrogate_options: Optional dict of kwargs for `Surrogate`
            (used if no pre-instantiated Surrogate via is passed via `surrogate`).
            Can include:
            - model_options: Dict of options to surrogate's underlying
            BoTorch `Model`,
            - submodel_options or submodel_options_per_outcome:
            Options for submodels in `ListSurrogate`, see documentation
            for `ListSurrogate`.
        refit_on_update: Whether to reoptimize model parameters during call
            to `BoTorchModel.update`. If false, training data for the model
            (used for inference) is still swapped for new training data, but
            model parameters are not reoptimized.
        refit_on_cv: Whether to reoptimize model parameters during call to
            `BoTorchmodel.cross_validate`.
        warm_start_refit: Whether to load parameters from either the provided
            state dict or the state dict of the current BoTorch `Model` during
            refitting. If False, model parameters will be reoptimized from
            scratch on refit. NOTE: This setting is ignored during `update` or
            `cross_validate` if the corresponding `refit_on_...` is False.
    """

    acquisition_class: Type[Acquisition]
    acquisition_options: Dict[str, Any]
    surrogate_options: Dict[str, Any]
    _surrogate: Optional[Surrogate]
    _botorch_acqf_class: Optional[Type[AcquisitionFunction]]
    _search_space_digest: Optional[SearchSpaceDigest] = None

    def __init__(
        self,
        acquisition_class: Optional[Type[Acquisition]] = None,
        acquisition_options: Optional[Dict[str, Any]] = None,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        surrogate: Optional[Surrogate] = None,
        surrogate_options: Optional[Dict[str, Any]] = None,
        refit_on_update: bool = True,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
    ) -> None:
        self._surrogate = surrogate
        if surrogate and surrogate_options:
            raise ValueError(  # pragma: no cover
                "`surrogate_options` are only applied when using the default "
                "surrogate, so only one of `surrogate` and `surrogate_options`"
                " arguments is expected."
            )
        self.surrogate_options = surrogate_options or {}
        self.acquisition_class = acquisition_class or Acquisition
        # `_botorch_acqf_class` can be `None` here. If so, `Model.gen` or `Model.
        # evaluate_acquisition_function` will set it with `choose_botorch_acqf_class`.
        self._botorch_acqf_class = botorch_acqf_class
        self.acquisition_options = acquisition_options or {}
        self.refit_on_update = refit_on_update
        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit

    @property
    def surrogate(self) -> Surrogate:
        """Ax ``Surrogate`` object (wrapper for BoTorch ``Model``), associated with
        this model. Raises an error if one is not yet set.
        """
        if not self._surrogate:
            raise ValueError("Surrogate has not yet been set.")
        return not_none(self._surrogate)

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class, associated with this model.
        Raises an error if one is not yet set.
        """
        if not self._botorch_acqf_class:
            raise ValueError("BoTorch `AcquisitionFunction` has not yet been set.")
        return not_none(self._botorch_acqf_class)

    @property
    def Xs(self) -> List[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x d``,
        where `n_i` is the number of training inputs for the i-th model.

        NOTE: This is an accessor for ``self.surrogate.training_data.Xs``
        and returns it unchanged.
        """
        return self.surrogate.training_data.Xs

    @property
    def Ys(self) -> List[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x 1``,
        where `n_i` is the number of training observations for the i-th
        (single-output) model.

        NOTE: This is an accessor for ``self.surrogate.training_data.Ys``
        and returns it unchanged.
        """
        return self.surrogate.training_data.Ys

    @property
    def Yvars(self) -> Optional[List[Tensor]]:
        """An optional list of tensors, each of shape
        ``batch_shape x n_i x 1``, where ``n_i`` is the number of training
        observations of the  observation noise for the i-th  (single-output)
        model. If `None`, the observation noise level is unobserved.

        NOTE: This is an accessor for ``self.surrogate.training_data.Yvars``
        and returns it unchanged.
        """
        return self.surrogate.training_data.Yvars

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        target_fidelities: Optional[Dict[int, float]] = None,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        # Ensure that parts of data all have equal lengths.
        validate_data_format(Xs=Xs, Ys=Ys, Yvars=Yvars, metric_names=metric_names)

        # store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # Choose `Surrogate` and undelying `Model` based on properties of data.
        if not self._surrogate:
            self._autoset_surrogate(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                search_space_digest=search_space_digest,
                metric_names=metric_names,
            )

        self.surrogate.fit(
            training_data=TrainingData(Xs=Xs, Ys=Ys, Yvars=Yvars),
            search_space_digest=search_space_digest,
            metric_names=metric_names,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
        )

    @copy_doc(TorchModel.update)
    def update(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        if not self._surrogate:
            raise ValueError("Cannot update model that has not been fitted.")

        # store search space info  for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # Sometimes the model fit should be restarted from scratch on update, for models
        # that are prone to overfitting. In those cases, `self.warm_start_refit` should
        # be false and `Surrogate.update` will not receive a state dict and will not
        # pass it to the underlying `Surrogate.fit`.
        state_dict = (
            None
            if self.refit_on_update and not self.warm_start_refit
            else self.surrogate.model.state_dict()
        )

        self.surrogate.update(
            training_data=TrainingData(Xs=Xs, Ys=Ys, Yvars=Yvars),
            search_space_digest=search_space_digest,
            metric_names=metric_names,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=self.refit_on_update,
        )

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return self.surrogate.predict(X=X)

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[Tensor, Tensor, TGenMetadata, Optional[List[TCandidateMetadata]]]:
        if self._search_space_digest is None:
            raise RuntimeError("Must `fit` the model before calling `gen`.")
        acq_options, opt_options = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options, model_gen_options=model_gen_options
        )
        # update bounds / target fidelities
        new_ssd_args = {
            **dataclasses.asdict(self._search_space_digest),
            "bounds": bounds,
            "target_fidelities": target_fidelities or {},
        }
        search_space_digest = SearchSpaceDigest(**new_ssd_args)

        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            acq_options=acq_options,
        )
        botorch_rounding_func = get_rounding_func(rounding_func)
        candidates, expected_acquisition_value = acqf.optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=linear_constraints
            ),
            fixed_features=fixed_features,
            rounding_func=botorch_rounding_func,
            optimizer_options=checked_cast(dict, opt_options),
        )
        gen_metadata: TGenMetadata = {
            Keys.EXPECTED_ACQF_VAL: expected_acquisition_value.tolist()
        }
        if objective_weights.nonzero().numel() > 1:  # pyre-ignore [16]
            gen_metadata["objective_thresholds"] = acqf.objective_thresholds
        return (
            candidates.detach().cpu(),
            torch.ones(n, dtype=self.surrogate.dtype),
            gen_metadata,
            None,
        )

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Optional[Tensor]:
        raise NotImplementedError("Coming soon.")

    @copy_doc(TorchModel.evaluate_acquisition_function)
    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            acq_options=acq_options,
        )
        return acqf.evaluate(X=X)

    def cross_validate(
        self,
        Xs_train: List[Tensor],
        Ys_train: List[Tensor],
        Yvars_train: List[Tensor],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
    ) -> Tuple[Tensor, Tensor]:
        current_surrogate = self.surrogate
        # If we should be refitting but not warm-starting the refit, set
        # `state_dict` to None to avoid loading it.
        state_dict = (
            None
            if self.refit_on_cv and not self.warm_start_refit
            else deepcopy(current_surrogate.model.state_dict())
        )

        # Temporarily set `_surrogate` to cloned surrogate to set
        # the training data on cloned surrogate to train set and
        # use it to predict the test point.
        surrogate_clone = self.surrogate.clone_reset()
        self._surrogate = surrogate_clone

        try:
            self.fit(
                Xs=Xs_train,
                Ys=Ys_train,
                Yvars=Yvars_train,
                search_space_digest=search_space_digest,
                metric_names=metric_names,
                state_dict=state_dict,
                refit=self.refit_on_cv,
            )
            X_test_prediction = self.predict(X=X_test)
        finally:
            # Reset the surrogate back to this model's surrogate, make
            # sure the cloned surrogate doesn't stay around if fit or
            # predict fail.
            self._surrogate = current_surrogate
        return X_test_prediction

    def _autoset_surrogate(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
    ) -> None:
        """Sets a default surrogate on this model if one was not explicitly
        provided.
        """
        # To determine whether to use `ListSurrogate`, we need to check for
        # the batched multi-output case, so we first see which model would
        # be chosen given the Yvars and the properties of data.
        botorch_model_class = choose_model_class(
            Yvars=Yvars,
            search_space_digest=search_space_digest,
        )
        if use_model_list(Xs=Xs, botorch_model_class=botorch_model_class):
            # If using `ListSurrogate` / `ModelListGP`, pick submodels for each
            # outcome.
            botorch_submodel_class_per_outcome = {
                metric_name: choose_model_class(
                    Yvars=[Yvar],
                    search_space_digest=search_space_digest,
                )
                for Yvar, metric_name in zip(Yvars, metric_names)
            }
            self._surrogate = ListSurrogate(
                botorch_submodel_class_per_outcome=botorch_submodel_class_per_outcome,
                **self.surrogate_options,
            )
        else:
            # Using regular `Surrogate`, so botorch model picked at the beginning
            # of the function is the one we should use.
            self._surrogate = Surrogate(
                botorch_model_class=botorch_model_class, **self.surrogate_options
            )

    def _instantiate_acquisition(
        self,
        search_space_digest: SearchSpaceDigest,
        objective_weights: Tensor,
        objective_thresholds: Optional[Tensor] = None,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Acquisition:
        """Set a BoTorch acquisition function class for this model if needed and
        instantiate it.

        Returns:
            A BoTorch ``AcquisitionFunction`` instance.
        """
        if not self._botorch_acqf_class:
            self._botorch_acqf_class = choose_botorch_acqf_class(
                objective_thresholds=objective_thresholds,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
            )
        return self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self.botorch_acqf_class,
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            objective_thresholds=objective_thresholds,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            options=acq_options,
        )
