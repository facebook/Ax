#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.models.torch.botorch import get_rounding_func
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    construct_single_training_data,
    construct_training_data_list,
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
        surrogate_fit_options: Optional dict of kwargs, passed to
            `Surrogate.fit`, including:
            - state_dict: `state_dict` for the underlying BoTorch `Model`,
            - refit_on_update: Whether to re-fit the underlying BoTorch `Model`
            when updating it with new data.

    """

    acquisition_class: Type[Acquisition]
    acquisition_options: Dict[str, Any]
    surrogate_options: Dict[str, Any]
    surrogate_fit_options: Dict[str, Any]
    _surrogate: Optional[Surrogate]
    _botorch_acqf_class: Optional[Type[AcquisitionFunction]]

    def __init__(
        self,
        acquisition_class: Optional[Type[Acquisition]] = None,
        acquisition_options: Optional[Dict[str, Any]] = None,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        surrogate: Optional[Surrogate] = None,
        surrogate_options: Optional[Dict[str, Any]] = None,
        surrogate_fit_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._surrogate = surrogate
        if surrogate and surrogate_options:
            raise ValueError(  # pragma: no cover
                "`surrogate_options` are only applied when using the default "
                "surrogate, so only one of `surrogate` and `surrogate_options`"
                " arguments is expected."
            )
        self.surrogate_options = surrogate_options or {}
        self.surrogate_fit_options = surrogate_fit_options or {}
        self.acquisition_class = acquisition_class or Acquisition
        # `_botorch_acqf_class` can be set to `None` here. If so,
        # `Model.gen` will set it with `choose_botorch_acqf_class`.
        self._botorch_acqf_class = (
            botorch_acqf_class or self.acquisition_class.default_botorch_acqf_class
        )
        self.acquisition_options = acquisition_options or {}

    @property
    def surrogate(self) -> Surrogate:
        if not self._surrogate:
            raise ValueError("Surrogate has not yet been set.")
        return not_none(self._surrogate)

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        if not self._botorch_acqf_class:
            raise ValueError("BoTorch `AcquisitionFunction` has not yet been set.")
        return not_none(self._botorch_acqf_class)

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        target_fidelities: Optional[Dict[int, float]] = None,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        # Ensure that parts of data all have equal lengths.
        validate_data_format(Xs=Xs, Ys=Ys, Yvars=Yvars, metric_names=metric_names)

        # Choose `Surrogate` and undelying `Model` based on properties of data.
        if not self._surrogate:
            self._autoset_surrogate(
                Xs=Xs,
                Ys=Ys,
                Yvars=Yvars,
                task_features=task_features,
                fidelity_features=fidelity_features,
                metric_names=metric_names,
            )

        if isinstance(self.surrogate, ListSurrogate):
            training_data = construct_training_data_list(Xs=Xs, Ys=Ys, Yvars=Yvars)
        else:
            training_data = construct_single_training_data(Xs=Xs, Ys=Ys, Yvars=Yvars)

        # Fit the model.
        if self.surrogate_fit_options.get(
            Keys.REFIT_ON_UPDATE, True
        ) and not self.surrogate_fit_options.get(Keys.WARM_START_REFITTING, True):
            self.surrogate_fit_options[Keys.STATE_DICT] = None
        self.surrogate.fit(
            # pyre-ignore[6]: Base `Surrogate` expects only single `TrainingData`,
            # but `ListSurrogate` expects a list of them, to `training_data` here is
            # a union of the two.
            training_data=training_data,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            fidelity_features=fidelity_features,
            target_fidelities=target_fidelities,
            metric_names=metric_names,
            candidate_metadata=candidate_metadata,
            state_dict=self.surrogate_fit_options.get(Keys.STATE_DICT, None),
            refit=self.surrogate_fit_options.get(Keys.REFIT_ON_UPDATE, True),
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
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[Tensor, Tensor, TGenMetadata, Optional[List[TCandidateMetadata]]]:
        acq_options, opt_options = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options, model_gen_options=model_gen_options
        )
        acqf = self._instantiate_acquisition(
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            target_fidelities=target_fidelities,
            acq_options=acq_options,
        )

        botorch_rounding_func = get_rounding_func(rounding_func)
        candidates, expected_acquisition_value = acqf.optimize(
            bounds=self._bounds_as_tensor(bounds=bounds),
            n=n,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=linear_constraints
            ),
            fixed_features=fixed_features,
            rounding_func=botorch_rounding_func,
            optimizer_options=checked_cast(dict, opt_options),
        )
        return (
            candidates.detach().cpu(),
            torch.ones(n, dtype=self.surrogate.dtype),
            {Keys.EXPECTED_ACQF_VAL: expected_acquisition_value.tolist()},
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
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        acqf = self._instantiate_acquisition(
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            target_fidelities=target_fidelities,
            acq_options=acq_options,
        )
        return acqf.evaluate(X=X)

    def _autoset_surrogate(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        task_features: List[int],
        fidelity_features: List[int],
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
            task_features=task_features,
            fidelity_features=fidelity_features,
        )
        if use_model_list(Xs=Xs, botorch_model_class=botorch_model_class):
            # If using `ListSurrogate` / `ModelListGP`, pick submodels for each
            # outcome.
            botorch_submodel_class_per_outcome = {
                metric_name: choose_model_class(
                    Yvars=[Yvar],
                    task_features=task_features,
                    fidelity_features=fidelity_features,
                )
                for Yvar, metric_name in zip(Yvars, metric_names)
            }
            self._surrogate = ListSurrogate(
                botorch_submodel_class_per_outcome=botorch_submodel_class_per_outcome,
                **self.surrogate_options
            )
        else:
            # Using regular `Surrogate`, so botorch model picked at the beginning
            # of the function is the one we should use.
            self._surrogate = Surrogate(
                botorch_model_class=botorch_model_class, **self.surrogate_options
            )

    def _bounds_as_tensor(self, bounds: List[Tuple[float, float]]) -> Tensor:
        bounds_ = torch.tensor(
            bounds, dtype=self.surrogate.dtype, device=self.surrogate.device
        )
        return bounds_.transpose(0, 1)

    def _instantiate_acquisition(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Acquisition:
        if not self._botorch_acqf_class:
            self._botorch_acqf_class = choose_botorch_acqf_class()

        return self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self.botorch_acqf_class,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            target_fidelities=target_fidelities,
            options=acq_options,
        )
