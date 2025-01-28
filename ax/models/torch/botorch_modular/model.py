#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.torch.botorch import (
    get_feature_importances_from_botorch_model,
    get_rounding_func,
)
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.models.torch.botorch_modular.utils import (
    check_outcome_dataset_match,
    choose_botorch_acqf_class,
    construct_acquisition_and_optimizer_options,
    ModelConfig,
)
from ax.models.torch.utils import _to_inequality_constraints
from ax.models.torch_base import TorchGenResults, TorchModel, TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance
from torch import Tensor


class BoTorchModel(TorchModel, Base):
    """**All classes in 'botorch_modular' directory are under
    construction, incomplete, and should be treated as alpha
    versions only.**

    Modular ``Model`` class for combining BoTorch subcomponents
    in Ax. Specified via ``Surrogate`` and ``Acquisition``, which wrap
    BoTorch ``Model`` and ``AcquisitionFunction``, respectively, for
    convenient use in Ax.

    Args:
        acquisition_class: Type of ``Acquisition`` to be used in
            this model, auto-selected based on experiment and data
            if not specified.
        acquisition_options: Optional dict of kwargs, passed to
            the constructor of BoTorch ``AcquisitionFunction``.
        botorch_acqf_class: Type of ``AcquisitionFunction`` to be
            used in this model, auto-selected based on experiment
            and data if not specified.
        surrogate_spec: An optional ``SurrogateSpec`` object specifying how to
            construct the ``Surrogate`` and the underlying BoTorch ``Model``.
        surrogate_specs: DEPRECATED. Please use ``surrogate_spec`` instead.
        surrogate: In lieu of ``SurrogateSpec``, an instance of ``Surrogate`` may
            be provided. In most cases, ``surrogate_spec`` should be used instead.
        refit_on_cv: Whether to reoptimize model parameters during call to
            ``BoTorchmodel.cross_validate``.
        warm_start_refit: Whether to load parameters from either the provided
            state dict or the state dict of the current BoTorch ``Model`` during
            refitting. If False, model parameters will be reoptimized from
            scratch on refit. NOTE: This setting is ignored during
            ``cross_validate`` if ``refit_on_cv`` is False.
    """

    acquisition_class: type[Acquisition]
    acquisition_options: dict[str, Any]

    surrogate_spec: SurrogateSpec | None
    _surrogate: Surrogate | None

    _botorch_acqf_class: type[AcquisitionFunction] | None
    _search_space_digest: SearchSpaceDigest | None = None
    _supports_robust_optimization: bool = True

    def __init__(
        self,
        surrogate_spec: SurrogateSpec | None = None,
        surrogate_specs: Mapping[str, SurrogateSpec] | None = None,
        surrogate: Surrogate | None = None,
        acquisition_class: type[Acquisition] | None = None,
        acquisition_options: dict[str, Any] | None = None,
        botorch_acqf_class: type[AcquisitionFunction] | None = None,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
    ) -> None:
        # Check that only one surrogate related option is provided.
        if bool(surrogate_spec) + bool(surrogate_specs) + bool(surrogate) > 1:
            raise UserInputError(
                "Only one of `surrogate_spec`, `surrogate_specs`, and `surrogate` "
                "can be specified. Please use `surrogate_spec`."
            )
        if surrogate_specs is not None:
            if len(surrogate_specs) > 1:
                raise DeprecationWarning(
                    "Support for multiple `Surrogate`s has been deprecated. "
                    "Please use the `surrogate_spec` input in the future to "
                    "specify a single `Surrogate`."
                )
            warnings.warn(
                "The `surrogate_specs` argument is deprecated in favor of "
                "`surrogate_spec`, which accepts a single `SurrogateSpec` object. "
                "Please use `surrogate_spec` in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            surrogate_spec = next(iter(surrogate_specs.values()))
        self.surrogate_spec = surrogate_spec
        self._surrogate = surrogate

        self.acquisition_class = acquisition_class or Acquisition
        self.acquisition_options = acquisition_options or {}
        self._botorch_acqf_class = botorch_acqf_class

        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit

    @property
    def surrogate(self) -> Surrogate:
        """Returns the ``Surrogate``, if it has been constructed."""
        if self._surrogate is None:
            raise ValueError("Surrogate has not yet been constructed.")
        return self._surrogate

    @property
    def Xs(self) -> list[Tensor]:
        """A list of tensors, each of shape ``batch_shape x n_i x d``,
        where `n_i` is the number of training inputs for the i-th model.

        NOTE: This is an accessor for ``self.surrogate.Xs``
        and returns it unchanged.
        """
        return self.surrogate.Xs

    @property
    def botorch_acqf_class(self) -> type[AcquisitionFunction]:
        """BoTorch ``AcquisitionFunction`` class, associated with this model.
        Raises an error if one is not yet set.
        """
        if not self._botorch_acqf_class:
            raise ValueError("BoTorch `AcquisitionFunction` has not yet been set.")
        return self._botorch_acqf_class

    def fit(
        self,
        datasets: Sequence[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
        state_dict: OrderedDict[str, Tensor] | None = None,
        refit: bool = True,
        **additional_model_inputs: Any,
    ) -> None:
        """Fit model to m outcomes.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one or more outcomes.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
            state_dict: An optional model statedict for the underlying ``Surrogate``.
                Primarily used in ``BoTorchModel.cross_validate``.
            refit: Whether to re-optimize model parameters.
            additional_model_inputs: Additional kwargs to pass to the
                model input constructor in ``Surrogate.fit``.
        """
        outcome_names = sum((ds.outcome_names for ds in datasets), [])
        check_outcome_dataset_match(
            outcome_names=outcome_names, datasets=datasets, exact_match=True
        )  # Checks for duplicate outcome names

        # Store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest

        # If a surrogate has not been constructed, construct it.
        if self._surrogate is None:
            surrogate_spec = (
                SurrogateSpec(model_configs=[ModelConfig(name="default")])
                if self.surrogate_spec is None
                else self.surrogate_spec
            )
            self._surrogate = Surrogate(
                surrogate_spec=surrogate_spec, refit_on_cv=self.refit_on_cv
            )

        # Fit the surrogate.
        for config in self.surrogate.surrogate_spec.model_configs:
            config.model_options.update(additional_model_inputs)
        for (
            config_list
        ) in self.surrogate.surrogate_spec.metric_to_model_configs.values():
            for config in config_list:
                config.model_options.update(additional_model_inputs)
        self.surrogate.fit(
            datasets=datasets,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            state_dict=state_dict,
            refit=refit,
        )

    def predict(
        self, X: Tensor, use_posterior_predictive: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Predicts, potentially from multiple surrogates.

        Args:
            X: (n x d) Tensor of input locations.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).

        Returns: Tuple of tensors: (n x m) mean, (n x m x m) covariance.
        """
        return self.surrogate.predict(
            X=X, use_posterior_predictive=use_posterior_predictive
        )

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
        # update bounds / target values
        search_space_digest = dataclasses.replace(
            self.search_space_digest,
            bounds=search_space_digest.bounds,
            target_values=search_space_digest.target_values or {},
        )

        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)
        candidates, expected_acquisition_value, weights = acqf.optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=_to_inequality_constraints(
                linear_constraints=torch_opt_config.linear_constraints
            ),
            fixed_features=torch_opt_config.fixed_features,
            rounding_func=botorch_rounding_func,
            optimizer_options=assert_is_instance(
                opt_options,
                dict,
            ),
        )
        gen_metadata = self._get_gen_metadata_from_acqf(
            acqf=acqf,
            torch_opt_config=torch_opt_config,
            expected_acquisition_value=expected_acquisition_value,
        )
        # log what model was used
        metric_to_model_config_name = {
            metric_name: model_config.name or str(model_config)
            for metric_name, model_config in (
                self.surrogate.metric_to_best_model_config.items()
            )
        }
        gen_metadata["metric_to_model_config_name"] = metric_to_model_config_name
        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=weights,
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
    ) -> Tensor | None:
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
        acq_options: dict[str, Any] | None = None,
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
        datasets: Sequence[SupervisedDataset],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        use_posterior_predictive: bool = False,
        **additional_model_inputs: Any,
    ) -> tuple[Tensor, Tensor]:
        current_surrogate = self.surrogate
        # If we should be refitting but not warm-starting the refit, set
        # `state_dict` to None to avoid loading it.
        state_dict = (
            None
            if self.refit_on_cv and not self.warm_start_refit
            else current_surrogate.model.state_dict()
        )

        # Temporarily set `_surrogate` to cloned surrogate to set
        # the training data on cloned surrogate to train set and
        # use it to predict the test point.
        self._surrogate = current_surrogate.clone_reset()

        # Remove the `robust_digest` since we do not want to use perturbations here.
        search_space_digest = dataclasses.replace(
            search_space_digest,
            robust_digest=None,
        )

        try:
            self.fit(
                datasets=datasets,
                search_space_digest=search_space_digest,
                # pyre-fixme [6]: state_dict() has a generic dict[str, Any] return type
                # but it is actually an OrderedDict[str, Tensor].
                state_dict=state_dict,
                refit=self.refit_on_cv,
                **additional_model_inputs,
            )
            X_test_prediction = self.predict(
                X=X_test,
                use_posterior_predictive=use_posterior_predictive,
            )
        finally:
            # Reset the surrogates back to this model's surrogate, make
            # sure the cloned surrogate doesn't stay around if fit or
            # predict fail.
            self._surrogate = current_surrogate
        return X_test_prediction

    @property
    def dtype(self) -> torch.dtype:
        """Torch data type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.dtype

    @property
    def device(self) -> torch.device:
        """Torch device type of the tensors in the training data used in the model,
        of which this ``Acquisition`` is a subcomponent.
        """
        return self.surrogate.device

    def _instantiate_acquisition(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: dict[str, Any] | None = None,
    ) -> Acquisition:
        """Set a BoTorch acquisition function class for this model if needed and
        instantiate it.

        Returns:
            A BoTorch ``AcquisitionFunction`` instance.
        """
        if not self._botorch_acqf_class:
            if torch_opt_config.risk_measure is not None:
                raise UnsupportedError(
                    "Automated selection of `botorch_acqf_class` is not supported "
                    "for robust optimization with risk measures. Please specify "
                    "`botorch_acqf_class` as part of `model_kwargs`."
                )
            self._botorch_acqf_class = choose_botorch_acqf_class(
                torch_opt_config=torch_opt_config
            )

        return self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self.botorch_acqf_class,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acq_options,
        )

    def feature_importances(self) -> npt.NDArray:
        """Compute feature importances from the model.

        This assumes that we can get model lengthscales from either
        ``covar_module.base_kernel.lengthscale`` or ``covar_module.lengthscale``.

        Returns:
            The feature importances as a numpy array of size len(metrics) x 1 x dim
            where each row sums to 1.
        """
        return get_feature_importances_from_botorch_model(model=self.surrogate.model)

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
