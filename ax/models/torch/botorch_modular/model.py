#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.torch.botorch import (
    get_feature_importances_from_botorch_model,
    get_rounding_func,
)
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    construct_acquisition_and_optimizer_options,
    convert_to_block_design,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelList
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.datasets import SupervisedDataset
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

T = TypeVar("T")


def single_surrogate_only(f: Callable[..., T]) -> Callable[..., T]:
    """
    For use as a decorator on functions only implemented for BotorchModels with a
    single Surrogate.
    """

    @wraps(f)
    def impl(self: "BoTorchModel", *args: List[Any], **kwargs: Dict[str, Any]) -> T:
        if len(self._surrogates) != 1:
            raise NotImplementedError(
                f"{f.__name__} not implemented for multi-surrogate case. Found "
                f"{self.surrogates=}."
            )
        return f(self, *args, **kwargs)

    return impl


@dataclass(frozen=True)
class SurrogateSpec:
    """
    Fields in the SurrogateSpec dataclass correspond to arguments in
    ``Surrogate.__init__``, except for ``outcomes`` which is used to specify which
    outcomes the Surrogate is responsible for modeling.
    When ``BotorchModel.fit`` is called, these fields will be used to construct the
    requisite Surrogate objects.
    If ``outcomes`` is left empty then no outcomes will be fit to the Surrogate.
    """

    botorch_model_class: Optional[Type[Model]] = None
    botorch_model_kwargs: Dict[str, Any] = field(default_factory=dict)

    mll_class: Type[MarginalLogLikelihood] = ExactMarginalLogLikelihood
    mll_kwargs: Dict[str, Any] = field(default_factory=dict)

    covar_module_class: Optional[Type[Kernel]] = None
    covar_module_kwargs: Optional[Dict[str, Any]] = None

    likelihood_class: Optional[Type[Likelihood]] = None
    likelihood_kwargs: Optional[Dict[str, Any]] = None

    input_transform: Optional[InputTransform] = None
    outcome_transform: Optional[OutcomeTransform] = None

    allow_batched_models: bool = True

    outcomes: List[str] = field(default_factory=list)


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
        surrogate_specs: Optional Mapping of names onto SurrogateSpecs, which specify
            how to initialize specific Surrogates to model specific outcomes. If None
            is provided a single Surrogate will be created and set up automatically
            based on the data provided.
        surrogate: In liu of SurrogateSpecs, an instance of `Surrogate` may be
            provided to be used as the sole Surrogate for all outcomes
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

    surrogate_specs: Dict[str, SurrogateSpec]
    _surrogates: Dict[str, Surrogate]

    _botorch_acqf_class: Optional[Type[AcquisitionFunction]]
    _search_space_digest: Optional[SearchSpaceDigest] = None
    _supports_robust_optimization: bool = True

    def __init__(
        self,
        surrogate_specs: Optional[Mapping[str, SurrogateSpec]] = None,
        surrogate: Optional[Surrogate] = None,
        acquisition_class: Optional[Type[Acquisition]] = None,
        acquisition_options: Optional[Dict[str, Any]] = None,
        botorch_acqf_class: Optional[Type[AcquisitionFunction]] = None,
        refit_on_update: bool = True,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
    ) -> None:
        # Ensure only surrogate_specs or surrogate is provided
        if surrogate_specs and surrogate:
            raise UserInputError(  # pragma: no cover
                "Only one of `surrogate_specs` and `surrogate` arguments is expected."
            )

        # Ensure each outcome is only modeled by one Surrogate in the SurrogateSpecs
        if surrogate_specs is not None:
            outcomes_by_surrogate_label = {
                label: spec.outcomes for label, spec in surrogate_specs.items()
            }
            if sum(
                len(outcomes) for outcomes in outcomes_by_surrogate_label.values()
            ) != len(set(*outcomes_by_surrogate_label.values())):
                raise UserInputError(  # pragma: no cover
                    "Each outcome may be modeled by only one Surrogate, found "
                    f"{outcomes_by_surrogate_label}"
                )

        # Ensure user does not use reserved Surrogate labels
        if (
            surrogate_specs is not None
            and len(
                {Keys.ONLY_SURROGATE, Keys.AUTOSET_SURROGATE} - surrogate_specs.keys()
            )
            < 2
        ):
            raise UserInputError(  # pragma: no cover
                f"SurrogateSpecs may not be labeled {Keys.ONLY_SURROGATE} or "
                f"{Keys.AUTOSET_SURROGATE}, these are reserved."
            )

        self._surrogates = {}
        self.surrogate_specs = {}
        if surrogate_specs is not None:
            self.surrogate_specs: Dict[str, SurrogateSpec] = {
                label: spec for label, spec in surrogate_specs.items()
            }
        elif surrogate is not None:
            self._surrogates = {Keys.ONLY_SURROGATE: surrogate}

        self.acquisition_class = acquisition_class or Acquisition
        self.acquisition_options = acquisition_options or {}
        self._botorch_acqf_class = botorch_acqf_class

        self.refit_on_update = refit_on_update
        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit

    @property
    def surrogates(self) -> Dict[str, Surrogate]:
        """Surrogates by label"""
        return self._surrogates

    @property
    @single_surrogate_only
    def surrogate(self) -> Surrogate:
        """Surrogate, if there is only one."""

        return next(iter(self.surrogates.values()))

    @property
    @single_surrogate_only
    def Xs(self) -> List[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x d``,
        where `n_i` is the number of training inputs for the i-th model.

        NOTE: This is an accessor for ``self.surrogate.Xs``
        and returns it unchanged.
        """

        return self.surrogate.Xs

    @property
    def botorch_acqf_class(self) -> Type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class, associated with this model.
        Raises an error if one is not yet set.
        """
        if not self._botorch_acqf_class:
            raise ValueError("BoTorch `AcquisitionFunction` has not yet been set.")
        return self._botorch_acqf_class

    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        # state dict by surrogate label
        state_dicts: Optional[Mapping[str, Dict[str, Tensor]]] = None,
        refit: bool = True,
        **kwargs: Any,
    ) -> None:
        """Fit model to m outcomes.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome).
            metric_names: A list of metric names, with the i-th metric
                corresponding to the i-th dataset.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dicts: Optional state dict to load by model label as passed in via
                surrogate_specs. If using a single, pre-instantiated model use
                `Keys.ONLY_SURROGATE.
            refit: Whether to re-optimize model parameters.
        """

        if len(datasets) != len(metric_names):
            raise ValueError(
                "Length of datasets and metric_names must match, but your inputs "
                f"are of lengths {len(datasets)} and {len(metric_names)}, "
                "respectively."
            )

        # Store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # Step 0. If the user passed in a preconstructed surrogate we won't have a
        # SurrogateSpec and must assume we're fitting all metrics
        if Keys.ONLY_SURROGATE in self._surrogates.keys():
            self._surrogates[Keys.ONLY_SURROGATE].fit(
                datasets=datasets,
                metric_names=metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=state_dicts.get(Keys.ONLY_SURROGATE)
                if state_dicts
                else None,
                refit=refit,
                **kwargs,
            )
            return

        # Step 1. Initialize a Surrogate for every SurrogateSpec
        self._surrogates = {
            label: Surrogate(
                # if None, Surrogate will autoset class per outcome at construct time
                botorch_model_class=spec.botorch_model_class,
                model_options=spec.botorch_model_kwargs,
                mll_class=spec.mll_class,
                mll_options=spec.mll_kwargs,
                covar_module_class=spec.covar_module_class,
                covar_module_options=spec.covar_module_kwargs,
                likelihood_class=spec.likelihood_class,
                likelihood_options=spec.likelihood_kwargs,
                input_transform=spec.input_transform,
                outcome_transform=spec.outcome_transform,
                allow_batched_models=spec.allow_batched_models,
            )
            for label, spec in self.surrogate_specs.items()
        }

        # Step 1.5. If any outcomes are not explicitly assigned to a Surrogate, create
        # a new Surrogate for all these metrics (which will autoset its botorch model
        # class per outcome) UNLESS there is only one SurrogateSpec with no outcomes
        # assigned to it, in which case that will be used for all metrics.
        assigned_metric_names = {
            item
            for sublist in [spec.outcomes for spec in self.surrogate_specs.values()]
            for item in sublist
        }
        unassigned_metric_names = [
            name for name in metric_names if name not in assigned_metric_names
        ]
        if len(unassigned_metric_names) > 0 and len(self.surrogates) != 1:
            self._surrogates[Keys.AUTOSET_SURROGATE] = Surrogate()

        # Step 2. Fit each Surrogate iteratively using its assigned outcomes
        datasets_by_metric_name = dict(zip(metric_names, datasets))
        for label, surrogate in self.surrogates.items():
            if label == Keys.AUTOSET_SURROGATE or len(self.surrogates) == 1:
                subset_metric_names = unassigned_metric_names
            else:
                subset_metric_names = self.surrogate_specs[label].outcomes

            subset_datasets = [
                datasets_by_metric_name[metric_name]
                for metric_name in subset_metric_names
            ]
            if (
                len(subset_datasets) > 1
                # if Surrogate's model is none a ModelList will be autoset
                and surrogate._model is not None
                and not isinstance(surrogate.model, ModelList)
            ):
                # Note: If the datasets do not confirm to a block design then this
                # will filter the data and drop observations to make sure that it does.
                # This can happen e.g. if only some metrics are observed at some points
                subset_datasets, metric_names = convert_to_block_design(
                    datasets=subset_datasets,
                    metric_names=metric_names,
                    force=True,
                )

            surrogate.fit(
                datasets=subset_datasets,
                metric_names=subset_metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=(state_dicts or {}).get(label),
                refit=refit,
                **kwargs,
            )

    @copy_doc(TorchModel.update)
    def update(
        self,
        datasets: List[Optional[SupervisedDataset]],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        **kwargs: Any,
    ) -> None:
        if len(self.surrogates) == 0:
            raise UnsupportedError("Cannot update model that has not been fitted.")

        # store search space info  for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        for label, surrogate in self.surrogates.items():
            # Sometimes the model fit should be restarted from scratch on update, for
            # models that are prone to overfitting. In those cases,
            # `self.warm_start_refit` should be false and `Surrogate.update` will not
            # receive a state dict and will not pass it to the underlying
            # `Surrogate.fit`.

            state_dict = (
                None
                if self.refit_on_update and not self.warm_start_refit
                else surrogate.model.state_dict()
            )
            if any(dataset is None for dataset in datasets):
                raise UnsupportedError(
                    f"{self.__class__.__name__}.update requires data for all outcomes."
                )

            # Only update each Surrogate on its own metrics unless it was the
            # preconstructed Surrogate
            datasets_by_metric_name = dict(zip(metric_names, datasets))
            subset_metric_names = (
                self.surrogate_specs[label].outcomes
                if label not in (Keys.ONLY_SURROGATE, Keys.AUTOSET_SURROGATE)
                else metric_names
            )
            subset_datasets = [
                datasets_by_metric_name[metric_name]
                for metric_name in subset_metric_names
            ]

            surrogate.update(
                datasets=subset_datasets,
                metric_names=subset_metric_names,
                search_space_digest=search_space_digest,
                candidate_metadata=candidate_metadata,
                state_dict=state_dict,
                refit=self.refit_on_update,
                **kwargs,
            )

    @single_surrogate_only
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict if only one Surrogate, Error if there are many"""

        return self.surrogate.predict(X=X)

    def predict_from_surrogate(
        self, surrogate_label: str, X: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Predict from the Surrogate with the given label."""
        return self.surrogates[surrogate_label].predict(X=X)

    @copy_doc(TorchModel.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        acq_options, opt_options = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options,
            model_gen_options=torch_opt_config.model_gen_options,
        )
        # update bounds / target fidelities
        search_space_digest = dataclasses.replace(
            self.search_space_digest,
            bounds=search_space_digest.bounds,
            target_fidelities=search_space_digest.target_fidelities or {},
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
            weights=torch.ones(n, dtype=self.dtype),
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
    @single_surrogate_only
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

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        # Will fail if metric_names exist across multiple models
        surrogate_labels = (
            [
                label
                for label, spec in self.surrogate_specs.items()
                if any(metric in spec.outcomes for metric in metric_names)
            ]
            if len(self.surrogates) > 1
            else [*self.surrogates.keys()]
        )
        if len(surrogate_labels) != 1:
            raise UserInputError(
                "May not cross validate multiple Surrogates at once. Please input "
                f"metric_names that exist on one Surrogate. {metric_names} spans "
                f"{surrogate_labels}"
            )
        surrogate_label = surrogate_labels[0]

        current_surrogates = self.surrogates
        # If we should be refitting but not warm-starting the refit, set
        # `state_dicts` to None to avoid loading it.
        state_dicts = (
            None
            if self.refit_on_cv and not self.warm_start_refit
            else {
                label: deepcopy(surrogate.model.state_dict())
                for label, surrogate in current_surrogates.items()
            }
        )

        # Temporarily set `_surrogates` to cloned surrogates to set
        # the training data on cloned surrogates to train set and
        # use it to predict the test point.
        surrogate_clones = {
            label: surrogate.clone_reset()
            for label, surrogate in self.surrogates.items()
        }
        self._surrogates = surrogate_clones
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
                state_dicts=state_dicts,
                refit=self.refit_on_cv,
                **kwargs,
            )
            X_test_prediction = self.predict_from_surrogate(
                surrogate_label=surrogate_label, X=X_test
            )
        finally:
            # Reset the surrogates back to this model's surrogate, make
            # sure the cloned surrogate doesn't stay around if fit or
            # predict fail.
            self._surrogates = current_surrogates
        return X_test_prediction

    @property
    def dtype(self) -> torch.dtype:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        dtypes = {
            label: surrogate.dtype for label, surrogate in self.surrogates.items()
        }

        dtypes_list = list(dtypes.values())
        if dtypes_list.count(dtypes_list[0]) != len(dtypes_list):
            raise NotImplementedError(  # pragma: no cover
                f"Expected all Surrogates to have same dtype, found {dtypes}"
            )

        return dtypes_list[0]

    @property
    def device(self) -> torch.device:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """

        devices = {
            label: surrogate.device for label, surrogate in self.surrogates.items()
        }

        devices_list = list(devices.values())
        if devices_list.count(devices_list[0]) != len(devices_list):
            raise NotImplementedError(  # pragma: no cover
                f"Expected all Surrogates to have same device, found {devices}"
            )

        return devices_list[0]

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
            surrogates=self.surrogates,
            botorch_acqf_class=self.botorch_acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acq_options,
        )

    def feature_importances(self) -> np.ndarray:
        """Compute feature importances from the model.

        Caveat: This assumes the following:
            1. There is a single surrogate model (potentially a `ModelList`).
            2. We can get model lengthscales from `covar_module.base_kernel.lengthscale`

        Returns:
            The feature importances as a numpy array of size len(metrics) x 1 x dim
            where each row sums to 1.
        """
        if list(self.surrogates.keys()) != [Keys.ONLY_SURROGATE]:
            raise NotImplementedError("Only support a single surrogate model for now")
        surrogate = self.surrogates[Keys.ONLY_SURROGATE]
        return get_feature_importances_from_botorch_model(model=surrogate.model)

    @property
    def search_space_digest(self) -> SearchSpaceDigest:
        if self._search_space_digest is None:
            raise RuntimeError(
                "`search_space_digest` is not initialized. Must `fit` the model first."
            )
        return self._search_space_digest

    @search_space_digest.setter
    def search_space_digest(self, value: SearchSpaceDigest) -> None:
        raise RuntimeError("Setting search_space_digest manually is disallowed.")

    @property
    def outcomes_by_surrogate_label(self) -> Dict[str, List[str]]:
        """Retuns a dictionary mapping from surrogate label to a list of outcomes."""
        outcomes_by_surrogate_label = {}
        for k, v in self.surrogates.items():
            outcomes_by_surrogate_label[k] = v.outcomes
        return outcomes_by_surrogate_label
