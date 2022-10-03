#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch import get_rounding_func
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    choose_model_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
    use_model_list,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.utils.datasets import SupervisedDataset
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
    _supports_robust_optimization: bool = True

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
    def Xs(self) -> List[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x d``,
        where `n_i` is the number of training inputs for the i-th model.

        NOTE: This is an accessor for ``self.surrogate.Xs``
        and returns it unchanged.
        """
        return self.surrogate.Xs

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

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
    ) -> None:
        if not len(datasets) == len(metric_names):
            raise ValueError(
                "Length of datasets and metric_names must match, but your inputs "
                f"are of lengths {len(datasets)} and {len(metric_names)}, "
                "respectively."
            )

        # store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # Choose `Surrogate` and undelying `Model` based on properties of data.
        if not self._surrogate:
            self._autoset_surrogate(
                datasets=datasets,
                metric_names=metric_names,
                search_space_digest=search_space_digest,
            )
        original_metric_names = deepcopy(metric_names)
        if len(datasets) > 1 and not isinstance(self.surrogate, ListSurrogate):
            # Note: If the datasets do not confirm to a block design then this
            # will filter the data and drop observations to make sure that it does.
            # This can happen e.g. if only some metrics are observed at some points.
            datasets, metric_names = convert_to_block_design(
                datasets=datasets,
                metric_names=metric_names,
                force=True,
            )

        self.surrogate.fit(
            datasets=datasets,
            metric_names=metric_names,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
            original_metric_names=original_metric_names,
        )

    @copy_doc(TorchModel.update)
    def update(
        self,
        datasets: List[Optional[SupervisedDataset]],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        if not self._surrogate:
            raise UnsupportedError("Cannot update model that has not been fitted.")

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
        if any(dataset is None for dataset in datasets):
            raise UnsupportedError(
                f"{self.__class__.__name__}.update requires data for all outcomes."
            )
        self.surrogate.update(
            datasets=[not_none(dataset) for dataset in datasets],
            metric_names=metric_names,
            search_space_digest=search_space_digest,
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
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        if self._search_space_digest is None:
            raise RuntimeError("Must `fit` the model before calling `gen`.")
        acq_options, opt_options = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options,
            model_gen_options=torch_opt_config.model_gen_options,
        )
        # update bounds / target fidelities
        search_space_digest = not_none(
            dataclasses.replace(
                self._search_space_digest,
                bounds=search_space_digest.bounds,
                target_fidelities=search_space_digest.target_fidelities or {},
            )
        )
        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)
        candidates, expected_acquisition_value = acqf.optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
            rounding_func=botorch_rounding_func,
            optimizer_options=checked_cast(dict, opt_options),
        )
        gen_metadata = self._get_gen_metadata_from_acqf(
            acqf=acqf,
            torch_opt_config=torch_opt_config,
            expected_acquisition_value=expected_acquisition_value,
        )
        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=torch.ones(n, dtype=self.surrogate.dtype),
            gen_metadata=gen_metadata,
        )

    def _get_gen_metadata_from_acqf(
        self,
        acqf: Acquisition,
        torch_opt_config: TorchOptConfig,
        expected_acquisition_value: Tensor,
    ) -> TGenMetadata:
        gen_metadata: TGenMetadata = {
            Keys.EXPECTED_ACQF_VAL: expected_acquisition_value.tolist()
        }
        if torch_opt_config.objective_weights.nonzero().numel() > 1:
            gen_metadata["objective_thresholds"] = acqf.objective_thresholds
            gen_metadata["objective_weights"] = acqf.objective_weights

        if hasattr(acqf.acqf, "outcome_model"):
            outcome_model = acqf.acqf.outcome_model
            if isinstance(
                outcome_model,
                FixedSingleSampleModel,
            ):
                gen_metadata["outcome_model_fixed_draw_weights"] = outcome_model.w
        return gen_metadata

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Optional[Tensor]:
        try:
            return self.surrogate.best_in_sample_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )[0]
        except ValueError:
            return None

    @copy_doc(TorchModel.evaluate_acquisition_function)
    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        return acqf.evaluate(X=X)

    def cross_validate(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
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
        # Remove the robust_digest since we do not want to use perturbations here.
        search_space_digest = dataclasses.replace(
            search_space_digest,
            robust_digest=None,
        )

        try:
            self.fit(
                datasets=datasets,
                metric_names=metric_names,
                search_space_digest=search_space_digest,
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
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
    ) -> None:
        """Sets a default surrogate on this model if one was not explicitly
        provided.
        """
        # To determine whether to use `ListSurrogate`, we need to check for
        # the batched multi-output case, so we first see which model would
        # be chosen given the Yvars and the properties of data.
        botorch_model_class = choose_model_class(
            datasets=datasets,
            search_space_digest=search_space_digest,
        )
        if use_model_list(datasets=datasets, botorch_model_class=botorch_model_class):
            # If using `ListSurrogate` / `ModelListGP`, pick submodels for each
            # outcome.
            botorch_submodel_class_per_outcome = {
                metric_name: choose_model_class(
                    datasets=[dataset],
                    search_space_digest=search_space_digest,
                )
                for dataset, metric_name in zip(datasets, metric_names)
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
        torch_opt_config: TorchOptConfig,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Acquisition:
        """Set a BoTorch acquisition function class for this model if needed and
        instantiate it.

        Returns:
            A BoTorch ``AcquisitionFunction`` instance.
        """
        if not self._botorch_acqf_class:
            if torch_opt_config.risk_measure is not None:  # pragma: no cover
                # TODO[T131759261]: Implement selection of acqf for robust opt.
                # This will depend on the properties of the robust search space and
                # the risk measure being used.
                raise NotImplementedError
            self._botorch_acqf_class = choose_botorch_acqf_class(
                pending_observations=torch_opt_config.pending_observations,
                outcome_constraints=torch_opt_config.outcome_constraints,
                linear_constraints=torch_opt_config.linear_constraints,
                fixed_features=torch_opt_config.fixed_features,
                objective_thresholds=torch_opt_config.objective_thresholds,
                objective_weights=torch_opt_config.objective_weights,
            )
        return self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self.botorch_acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acq_options,
        )
