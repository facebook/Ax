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
from ax.models.torch.botorch_modular.surrogate import Surrogate, TrainingData
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_mll_class,
    choose_model_class,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.equality import Base
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor


class BoTorchModel(TorchModel, Base):
    """
    **All classes in 'botorch_modular' directory are under
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
            on experiment and data.
        surrogate_fit_options: Optional dict of kwargs, passed to
            `Surrogate.fit`, like `state_dict` or `refit_on_update`.
    """

    acquisition_class: Type[Acquisition]
    acquisition_options: TConfig
    surrogate_fit_options: Dict[str, Any]
    _surrogate: Optional[Surrogate]
    _botorch_acqf_class: Optional[Type[AcquisitionFunction]]

    def __init__(
        self,
        acquisition_class: Optional[Type[Acquisition]] = None,
        acquisition_options: Optional[TConfig] = None,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        surrogate: Optional[Surrogate] = None,
        surrogate_fit_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._surrogate = surrogate
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

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Argument `Xs` expected.
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
        if not self._surrogate:
            model_class = choose_model_class(
                training_data=TrainingData(Xs=Xs, Ys=Ys, Yvars=Yvars),
                task_features=task_features,
                fidelity_features=fidelity_features,
            )
            mll_class = choose_mll_class(
                model_class=model_class,
                state_dict=self.surrogate_fit_options.get(Keys.STATE_DICT, None),
                refit=self.surrogate_fit_options.get(Keys.REFIT_ON_UPDATE, True),
            )
            self._surrogate = Surrogate(
                botorch_model_class=model_class, mll_class=mll_class
            )
        if self.surrogate_fit_options.get(
            Keys.REFIT_ON_UPDATE, True
        ) and not self.surrogate_fit_options.get(Keys.WARM_START_REFITTING, True):
            self.surrogate_fit_options[Keys.STATE_DICT] = None
        self.surrogate.fit(
            training_data=TrainingData(Xs=Xs, Ys=Ys, Yvars=Yvars),
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

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Argument `X` expected.
    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return self.surrogate.predict(X=X)

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Argument `bounds` expected.
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

        if not self._botorch_acqf_class:
            self._botorch_acqf_class = choose_botorch_acqf_class()

        acqf = self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self.botorch_acqf_class,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            target_fidelities=target_fidelities,
            options=acq_options,
        )

        botorch_rounding_func = get_rounding_func(rounding_func)

        # TODO: Would be good to abstract away bounds reformatting.
        bounds_ = torch.tensor(
            bounds, dtype=self.surrogate.dtype, device=self.surrogate.device
        )
        bounds_ = bounds_.transpose(0, 1)
        candidates, expected_acquisition_value = acqf.optimize(
            bounds=bounds_,
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

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Argument `bounds` expected.
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


def construct_acquisition_and_optimizer_options(
    acqf_options: TConfig, model_gen_options: Optional[TConfig] = None
) -> Tuple[TConfig, TConfig]:
    """Extract acquisition and optimizer options from `model_gen_options`."""
    acq_options = acqf_options.copy()
    opt_options = {}

    if model_gen_options:
        acq_options.update(
            checked_cast(dict, model_gen_options.get(Keys.ACQF_KWARGS, {}))
        )
        # TODO: Add this if all acq. functions accept the `subset_model`
        # kwarg or opt for kwarg filtering.
        # acq_options[SUBSET_MODEL] = model_gen_options.get(SUBSET_MODEL)
        opt_options = checked_cast(
            dict, model_gen_options.get(Keys.OPTIMIZER_KWARGS, {})
        ).copy()
    return acq_options, opt_options
