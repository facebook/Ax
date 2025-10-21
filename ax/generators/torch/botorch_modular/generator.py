#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
from collections import OrderedDict
from collections.abc import Sequence
from logging import Logger
from typing import Any

import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.exceptions.model import ModelError
from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.multi_acquisition import MultiAcquisition
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import (
    choose_botorch_acqf_class,
    construct_acquisition_and_optimizer_options,
    ModelConfig,
)
from ax.generators.torch.utils import (
    _to_inequality_constraints,
    get_feature_importances_from_botorch_model,
    get_rounding_func,
)
from ax.generators.torch_base import TorchGenerator, TorchGenResults, TorchOptConfig
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.settings import validate_input_scaling
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

logger: Logger = get_logger(__name__)


class BoTorchGenerator(TorchGenerator, Base):
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
            the the Ax ``Acquisition`` class.
        botorch_acqf_class: Type of ``AcquisitionFunction`` to be
            used in this model, auto-selected based on experiment
            and data if not specified.
        botorch_acqf_options: Optional dict of kwargs, passed to the botorch
            ``AcquisitionFunction``.
        botorch_acqf_classes_with_options: List of tuples of
            ``AcquisitionFunction`` classes and dicts of kwargs, passed to
            the botorch ``AcquisitionFunction``. This is used to specify
            multiple acquisition functions to be used with MultiAcquisition.
        surrogate_spec: An optional ``SurrogateSpec`` object specifying how to
            construct the ``Surrogate`` and the underlying BoTorch ``Model``.
        surrogate: In lieu of ``SurrogateSpec``, an instance of ``Surrogate`` may
            be provided. In most cases, ``surrogate_spec`` should be used instead.
        refit_on_cv: Whether to reoptimize model parameters during call to
            ``BoTorchGenerator.cross_validate``.
        warm_start_refit: Whether to load parameters from either the provided
            state dict or the state dict of the current BoTorch ``Model`` during
            refitting. If False, model parameters will be reoptimized from
            scratch on refit. NOTE: This setting is ignored during
            ``cross_validate`` if ``refit_on_cv`` is False. This is also used in
            Surrogate.model_selection.
        use_p_feasible: Whether we consider dispatching to
            ``qLogProbabilityOfFeasibility`` in ``choose_botorch_acqf_class``.
    """

    acquisition_class: type[Acquisition]
    acquisition_options: dict[str, Any]

    surrogate_spec: SurrogateSpec | None
    _surrogate: Surrogate | None

    _user_specified_botorch_acqf_class: type[AcquisitionFunction] | None
    _botorch_acqf_class: type[AcquisitionFunction] | None
    _botorch_acqf_options: dict[str, Any]
    _supports_robust_optimization: bool = True
    _acquisition: Acquisition | None = None

    def __init__(
        self,
        surrogate_spec: SurrogateSpec | None = None,
        surrogate: Surrogate | None = None,
        acquisition_class: type[Acquisition] | None = None,
        acquisition_options: dict[str, Any] | None = None,
        botorch_acqf_class: type[AcquisitionFunction] | None = None,
        botorch_acqf_options: dict[str, Any] | None = None,
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ]
        | None = None,
        refit_on_cv: bool = False,
        warm_start_refit: bool = True,
        use_p_feasible: bool = True,
    ) -> None:
        # Check that only one surrogate related option is provided.
        if surrogate_spec is not None and surrogate is not None:
            raise UserInputError(
                "Only one of `surrogate_spec` or `surrogate` "
                "can be specified. Please use `surrogate_spec`."
            )
        self.surrogate_spec = surrogate_spec
        self._surrogate = surrogate

        if botorch_acqf_class is not None:
            if botorch_acqf_classes_with_options is not None:
                raise UserInputError(
                    "Only one of `botorch_acqf_class` or "
                    "`botorch_acqf_classes_with_options` can be specified."
                )

        if (
            botorch_acqf_classes_with_options is not None
            and len(botorch_acqf_classes_with_options) >= 2
        ):
            if (
                acquisition_class is not None
                and acquisition_class is not MultiAcquisition
            ):
                raise UserInputError(
                    "Multiple classes in `botorch_acqf_classes_with_options`"
                    "must be used with MultiAcquisition."
                )
            acquisition_class = MultiAcquisition

        self.acquisition_class = acquisition_class or Acquisition

        self.acquisition_options = acquisition_options or {}
        self._user_specified_botorch_acqf_class = botorch_acqf_class
        self._botorch_acqf_class = botorch_acqf_class
        self._botorch_acqf_classes_with_options = botorch_acqf_classes_with_options
        self._botorch_acqf_options = botorch_acqf_options or {}

        self.refit_on_cv = refit_on_cv
        self.warm_start_refit = warm_start_refit
        self.use_p_feasible = use_p_feasible

    @property
    def surrogate(self) -> Surrogate:
        """Returns the ``Surrogate``, if it has been constructed."""
        if self._surrogate is None:
            raise ModelError("Surrogate has not yet been constructed.")
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
                Primarily used in ``BoTorchGenerator.cross_validate``.
            refit: Whether to re-optimize model parameters.
            additional_model_inputs: Additional kwargs to pass to the
                model input constructor in ``Surrogate.fit``.
        """
        # If a surrogate has not been constructed, construct it.
        if self._surrogate is None:
            surrogate_spec = (
                SurrogateSpec(model_configs=[ModelConfig(name="default")])
                if self.surrogate_spec is None
                else self.surrogate_spec
            )
            self._surrogate = Surrogate(
                surrogate_spec=surrogate_spec,
                refit_on_cv=self.refit_on_cv,
                warm_start_refit=self.warm_start_refit,
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
            repeat_model_selection_if_dataset_changed=True,
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

    @copy_doc(TorchGenerator.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        (
            acq_options,
            botorch_acqf_options,
            opt_options,
            botorch_acqf_classes_with_options,
        ) = construct_acquisition_and_optimizer_options(
            acqf_options=self.acquisition_options,
            botorch_acqf_options=self._botorch_acqf_options,
            model_gen_options=torch_opt_config.model_gen_options,
            botorch_acqf_classes_with_options=self._botorch_acqf_classes_with_options,
        )
        self._acquisition = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_options=botorch_acqf_options,
            acq_options=acq_options,
            botorch_acqf_classes_with_options=botorch_acqf_classes_with_options,
            n=n,
        )
        acqf = none_throws(self._acquisition)

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

        gen_metadata["num_pruned_dims"] = acqf.num_pruned_dims
        gen_metadata["models_used"] = acqf.models_used
        return gen_metadata

    @copy_doc(TorchGenerator.best_point)
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

    @copy_doc(TorchGenerator.evaluate_acquisition_function)
    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: dict[str, Any] | None = None,
        botorch_acqf_options: dict[str, Any] | None = None,
    ) -> Tensor:
        acqf = self._instantiate_acquisition(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_options=botorch_acqf_options or self._botorch_acqf_options,
            acq_options=acq_options,
        )
        return acqf.evaluate(X=X)

    @copy_doc(TorchGenerator.cross_validate)
    def cross_validate(
        self,
        datasets: Sequence[SupervisedDataset],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        use_posterior_predictive: bool = False,
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
            # Since each CV fold removes points from the training data, the
            # remaining observations will not pass the input scaling checks.
            # To avoid confusing users with warnings, we disable these checks.
            with validate_input_scaling(False):
                self.surrogate.fit(
                    datasets=datasets,
                    search_space_digest=search_space_digest,
                    state_dict=state_dict,
                    refit=self.refit_on_cv,
                    repeat_model_selection_if_dataset_changed=False,
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
        botorch_acqf_options: dict[str, Any],
        acq_options: dict[str, Any] | None = None,
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ]
        | None = None,
        n: int | None = None,
    ) -> Acquisition:
        """Set a BoTorch acquisition function class for this model if needed and
        instantiate it.

        Returns:
            A BoTorch ``AcquisitionFunction`` instance.
        """
        if (
            self._user_specified_botorch_acqf_class is None
            and self._botorch_acqf_classes_with_options is None
        ):
            if torch_opt_config.risk_measure is not None:
                raise UnsupportedError(
                    "Automated selection of `botorch_acqf_class` is not supported "
                    "for robust optimization with risk measures. Please specify "
                    "`botorch_acqf_class` as part of `model_kwargs`."
                )
            self._botorch_acqf_class = choose_botorch_acqf_class(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
                datasets=self.surrogate.training_data,
                use_p_feasible=self.use_p_feasible,
            )
        return self.acquisition_class(
            surrogate=self.surrogate,
            botorch_acqf_class=self._botorch_acqf_class,
            botorch_acqf_options=botorch_acqf_options,
            botorch_acqf_classes_with_options=botorch_acqf_classes_with_options,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            options=acq_options,
            n=n,
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
        """Returns the ``SearchSpaceDigest`` that was used while fitting the underlying
        surrogate. If the surrogate has not been fit, raises a ``ModelError``.
        """
        if self.surrogate._last_search_space_digest is None:
            raise ModelError(
                "`search_space_digest` is not initialized. Must `fit` the model first."
            )
        return self.surrogate._last_search_space_digest
