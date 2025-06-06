#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, Optional

import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.exceptions.core import AxWarning, DataRequiredError
from ax.generators.torch.botorch_defaults import (
    get_and_fit_model,
    get_qLogNEI,
    recommend_best_observed_point,
    scipy_optimizer,
    TAcqfConstructor,
)
from ax.generators.torch.utils import (
    _datasets_to_legacy_inputs,
    _get_X_pending_and_observed,
    _to_inequality_constraints,
    predict_from_model,
    subset_model,
)
from ax.generators.torch_base import TorchGenerator, TorchGenResults, TorchOptConfig
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelList
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import is_ensemble, normalize_indices
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor
from torch.nn import ModuleList  # @manual


# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
TModelConstructor = Callable[
    [
        list[Tensor],
        list[Tensor],
        list[Tensor],
        list[int],
        list[int],
        list[str],
        Optional[dict[str, Tensor]],
        Any,
    ],
    Model,
]
TModelPredictor = Callable[[Model, Tensor, bool], tuple[Tensor, Tensor]]


# pyre-fixme[33]: Aliased annotation cannot contain `Any`.
TOptimizer = Callable[
    [
        AcquisitionFunction,
        Tensor,
        int,
        Optional[list[tuple[Tensor, Tensor, float]]],
        Optional[list[tuple[Tensor, Tensor, float]]],
        Optional[dict[int, float]],
        Optional[Callable[[Tensor], Tensor]],
        Any,
    ],
    tuple[Tensor, Tensor],
]
TBestPointRecommender = Callable[
    [
        TorchGenerator,
        list[tuple[float, float]],
        Tensor,
        Optional[tuple[Tensor, Tensor]],
        Optional[tuple[Tensor, Tensor]],
        Optional[dict[int, float]],
        Optional[TConfig],
        Optional[dict[int, float]],
    ],
    Optional[Tensor],
]


class LegacyBoTorchGenerator(TorchGenerator):
    r"""
    Customizable botorch model.

    By default, this uses a noisy Log Expected Improvement (qLogNEI) acquisition
    function on top of a model made up of separate GPs, one for each outcome. This
    behavior can be modified by providing custom implementations of the following
    components:

    - a `model_constructor` that instantiates and fits a model on data
    - a `model_predictor` that predicts outcomes using the fitted model
    - a `acqf_constructor` that creates an acquisition function from a fitted model
    - a `acqf_optimizer` that optimizes the acquisition function
    - a `best_point_recommender` that recommends a current "best" point (i.e.,
        what the model recommends if the learning process ended now)

    Args:
        model_constructor: A callable that instantiates and fits a model on data,
            with signature as described below.
        model_predictor: A callable that predicts using the fitted model, with
            signature as described below.
        acqf_constructor: A callable that creates an acquisition function from a
            fitted model, with signature as described below.
        acqf_optimizer: A callable that optimizes the acquisition function, with
            signature as described below.
        best_point_recommender: A callable that recommends the best point, with
            signature as described below.
        refit_on_cv: If True, refit the model for each fold when performing
            cross-validation.
        warm_start_refitting: If True, start model refitting from previous
            model parameters in order to speed up the fitting process.
        prior: An optional dictionary that contains the specification of GP model prior.
            Currently, the keys include:
            - covar_module_prior: prior on covariance matrix e.g.
                {"lengthscale_prior": GammaPrior(3.0, 6.0)}.
            - type: type of prior on task covariance matrix e.g.`LKJCovariancePrior`.
            - sd_prior: A scalar prior over nonnegative numbers, which is used for the
                default LKJCovariancePrior task_covar_prior.
            - eta: The eta parameter on the default LKJ task_covar_prior.


    Call signatures:

    ::

        model_constructor(
            Xs,
            Ys,
            Yvars,
            task_features,
            fidelity_features,
            metric_names,
            state_dict,
            **kwargs,
        ) -> model

    Here `Xs`, `Ys`, `Yvars` are lists of tensors (one element per outcome),
    `task_features` identifies columns of Xs that should be modeled as a task,
    `fidelity_features` is a list of ints that specify the positions of fidelity
    parameters in 'Xs', `metric_names` provides the names of each `Y` in `Ys`,
    `state_dict` is a pytorch module state dict, and `model` is a BoTorch `Model`.
    Optional kwargs are being passed through from the `LegacyBoTorchGenerator`
    constructor. This callable is assumed to return a fitted BoTorch model that has
    the same dtype and lives on the same device as the input tensors.

    ::

        model_predictor(model, X) -> [mean, cov]

    Here `model` is a fitted botorch model, `X` is a tensor of candidate points,
    and `mean` and `cov` are the posterior mean and covariance, respectively.

    ::

        acqf_constructor(
            model,
            objective_weights,
            outcome_constraints,
            X_observed,
            X_pending,
            **kwargs,
        ) -> acq_function


    Here `model` is a botorch `Model`, `objective_weights` is a tensor of weights
    for the model outputs, `outcome_constraints` is a tuple of tensors describing
    the (linear) outcome constraints, `X_observed` are previously observed points,
    and `X_pending` are points whose evaluation is pending. `acq_function` is a
    BoTorch acquisition function crafted from these inputs. For additional
    details on the arguments, see `get_qLogNEI`.

    ::

        acqf_optimizer(
            acq_function,
            bounds,
            n,
            inequality_constraints,
            equality_constraints,
            fixed_features,
            rounding_func,
            **kwargs,
        ) -> candidates

    Here `acq_function` is a BoTorch `AcquisitionFunction`, `bounds` is a tensor
    containing bounds on the parameters, `n` is the number of candidates to be
    generated, `inequality_constraints` are inequality constraints on parameter
    values, `fixed_features` specifies features that should be fixed during
    generation, and `rounding_func` is a callback that rounds an optimization
    result appropriately. `candidates` is a tensor of generated candidates.
    For additional details on the arguments, see `scipy_optimizer`.

    ::

        best_point_recommender(
            model,
            bounds,
            objective_weights,
            outcome_constraints,
            linear_constraints,
            fixed_features,
            model_gen_options,
            target_fidelities,
        ) -> candidates

    Here `model` is a TorchGenerator, `bounds` is a list of tuples containing bounds
    on the parameters, `objective_weights` is a tensor of weights for the model outputs,
    `outcome_constraints` is a tuple of tensors describing the (linear) outcome
    constraints, `linear_constraints` is a tuple of tensors describing constraints
    on the design, `fixed_features` specifies features that should be fixed during
    generation, `model_gen_options` is a config dictionary that can contain
    model-specific options, and `target_fidelities` is a map from fidelity feature
    column indices to their respective target fidelities, used for multi-fidelity
    optimization problems. % TODO: refer to an example.
    """

    dtype: torch.dtype | None
    device: torch.device | None
    Xs: list[Tensor]
    Ys: list[Tensor]
    Yvars: list[Tensor]
    _model: Model | None
    _search_space_digest: SearchSpaceDigest | None = None

    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model,
        model_predictor: TModelPredictor = predict_from_model,
        acqf_constructor: TAcqfConstructor = get_qLogNEI,
        # pyre-fixme[9]: acqf_optimizer declared/used type mismatch
        acqf_optimizer: TOptimizer = scipy_optimizer,
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        refit_on_cv: bool = False,
        warm_start_refitting: bool = True,
        use_input_warping: bool = False,
        use_loocv_pseudo_likelihood: bool = False,
        prior: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "The legacy `LegacyBoTorchGenerator` and its subclasses, including the "
            f"current class `{self.__class__.__name__}`, slated for deprecation. "
            "These models will not be supported going forward and may be "
            "fully removed in a future release. Please consider using the "
            "Modular BoTorch Generator (MBG) setup (ax/models/torch/botorch_modular) "
            "instead. If you run into a use case that is not supported by MBG, "
            "please raise this with an issue at https://github.com/facebook/Ax",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model_constructor = model_constructor
        self.model_predictor = model_predictor
        self.acqf_constructor = acqf_constructor
        self.acqf_optimizer = acqf_optimizer
        self.best_point_recommender = best_point_recommender
        # pyre-fixme[4]: Attribute must be annotated.
        self._kwargs = kwargs
        self.refit_on_cv = refit_on_cv
        self.warm_start_refitting = warm_start_refitting
        self.use_input_warping = use_input_warping
        self.use_loocv_pseudo_likelihood = use_loocv_pseudo_likelihood
        self.prior = prior
        self._model: Model | None = None
        self.Xs = []
        self.Ys = []
        self.Yvars = []
        self.dtype = None
        self.device = None
        self.task_features: list[int] = []
        self.fidelity_features: list[int] = []
        self.metric_names: list[str] = []

    @copy_doc(TorchGenerator.fit)
    def fit(
        self,
        datasets: Sequence[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
    ) -> None:
        if len(datasets) == 0:
            raise DataRequiredError(
                "LegacyBoTorchGenerator.fit requires non-empty data sets."
            )
        self.Xs, self.Ys, self.Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        self.metric_names = sum((ds.outcome_names for ds in datasets), [])
        # Store search space info for later use (e.g. during generation)
        self._search_space_digest = search_space_digest
        self.dtype = self.Xs[0].dtype
        self.device = self.Xs[0].device
        self.task_features = none_throws(
            normalize_indices(search_space_digest.task_features, d=self.Xs[0].size(-1))
        )
        self.fidelity_features = none_throws(
            normalize_indices(
                search_space_digest.fidelity_features, d=self.Xs[0].size(-1)
            )
        )
        extra_kwargs = {} if self.prior is None else {"prior": self.prior}
        self._model = self.model_constructor(  # pyre-ignore [28]
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            task_features=self.task_features,
            fidelity_features=self.fidelity_features,
            metric_names=self.metric_names,
            use_input_warping=self.use_input_warping,
            use_loocv_pseudo_likelihood=self.use_loocv_pseudo_likelihood,
            **extra_kwargs,
            **self._kwargs,
        )

    @copy_doc(TorchGenerator.predict)
    def predict(
        self, X: Tensor, use_posterior_predictive: bool = False
    ) -> tuple[Tensor, Tensor]:
        if use_posterior_predictive:
            warnings.warn(
                f"{self.__class__.__name__} does not support posterior-predictive "
                "predictions. Ignoring `use_posterior_predictive`. ",
                AxWarning,
                stacklevel=2,
            )
        return self.model_predictor(model=self.model, X=X)  # pyre-ignore [28]

    @copy_doc(TorchGenerator.gen)
    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        options = torch_opt_config.model_gen_options or {}
        acf_options = options.get(Keys.ACQF_KWARGS, {})
        optimizer_options = options.get(Keys.OPTIMIZER_KWARGS, {})

        if search_space_digest.fidelity_features:
            raise NotImplementedError(
                "Base LegacyBoTorchGenerator does not support fidelity_features."
            )
        X_pending, X_observed = _get_X_pending_and_observed(
            Xs=self.Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            fit_out_of_design=torch_opt_config.fit_out_of_design,
        )
        model = self.model
        # subset model only to the outcomes we need for the optimization	357
        if options.get(Keys.SUBSET_MODEL, True):
            subset_model_results = subset_model(
                model=model,
                objective_weights=torch_opt_config.objective_weights,
                outcome_constraints=torch_opt_config.outcome_constraints,
            )
            model = subset_model_results.model
            objective_weights = subset_model_results.objective_weights
            outcome_constraints = subset_model_results.outcome_constraints
        else:
            objective_weights = torch_opt_config.objective_weights
            outcome_constraints = torch_opt_config.outcome_constraints

        bounds_ = torch.tensor(
            search_space_digest.bounds, dtype=self.dtype, device=self.device
        )
        bounds_ = bounds_.transpose(0, 1)

        botorch_rounding_func = get_rounding_func(torch_opt_config.rounding_func)

        from botorch.exceptions.errors import UnsupportedError

        # pyre-fixme[53]: Captured variable `X_observed` is not annotated.
        # pyre-fixme[53]: Captured variable `X_pending` is not annotated.
        # pyre-fixme[53]: Captured variable `acf_options` is not annotated.
        # pyre-fixme[53]: Captured variable `botorch_rounding_func` is not annotated.
        # pyre-fixme[53]: Captured variable `bounds_` is not annotated.
        # pyre-fixme[53]: Captured variable `model` is not annotated.
        # pyre-fixme[53]: Captured variable `objective_weights` is not annotated.
        # pyre-fixme[53]: Captured variable `optimizer_options` is not annotated.
        # pyre-fixme[53]: Captured variable `outcome_constraints` is not annotated.
        def make_and_optimize_acqf(override_qmc: bool = False) -> tuple[Tensor, Tensor]:
            add_kwargs = {"qmc": False} if override_qmc else {}
            acquisition_function = self.acqf_constructor(
                model=model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
                **acf_options,
                **add_kwargs,
            )
            acquisition_function = assert_is_instance(
                acquisition_function, AcquisitionFunction
            )
            # pyre-ignore: [28]
            candidates, expected_acquisition_value = self.acqf_optimizer(
                acq_function=assert_is_instance(
                    acquisition_function, AcquisitionFunction
                ),
                bounds=bounds_,
                n=n,
                inequality_constraints=_to_inequality_constraints(
                    linear_constraints=torch_opt_config.linear_constraints
                ),
                fixed_features=torch_opt_config.fixed_features,
                rounding_func=botorch_rounding_func,
                **optimizer_options,
            )
            return candidates, expected_acquisition_value

        try:
            candidates, expected_acquisition_value = make_and_optimize_acqf()
        except UnsupportedError as e:  # untested
            if "SobolQMCSampler only supports dimensions" in str(e):
                # dimension too large for Sobol, let's use IID
                candidates, expected_acquisition_value = make_and_optimize_acqf(
                    override_qmc=True
                )
            else:
                raise e

        gen_metadata = {}
        if expected_acquisition_value.numel() > 0:
            gen_metadata["expected_acquisition_value"] = (
                expected_acquisition_value.tolist()
            )

        return TorchGenResults(
            points=candidates.detach().cpu(),
            weights=torch.ones(n, dtype=self.dtype),
            gen_metadata=gen_metadata,
        )

    @copy_doc(TorchGenerator.best_point)
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Tensor | None:
        if torch_opt_config.is_moo:
            raise NotImplementedError(
                "Best observed point is incompatible with MOO problems."
            )
        target_fidelities = {
            k: v
            for k, v in search_space_digest.target_values.items()
            if k in search_space_digest.fidelity_features
        }
        return self.best_point_recommender(  # pyre-ignore [28]
            model=self,
            bounds=search_space_digest.bounds,
            objective_weights=torch_opt_config.objective_weights,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            model_gen_options=torch_opt_config.model_gen_options,
            target_fidelities=target_fidelities,
        )

    @copy_doc(TorchGenerator.cross_validate)
    def cross_validate(
        self,
        datasets: Sequence[SupervisedDataset],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        use_posterior_predictive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if self._model is None:
            raise RuntimeError("Cannot cross-validate model that has not been fitted.")
        if self.refit_on_cv:
            state_dict = None
        else:
            state_dict = deepcopy(self.model.state_dict())
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        model = self.model_constructor(  # pyre-ignore: [28]
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            task_features=self.task_features,
            state_dict=state_dict,
            fidelity_features=self.fidelity_features,
            metric_names=self.metric_names,
            refit_model=self.refit_on_cv,
            use_input_warping=self.use_input_warping,
            use_loocv_pseudo_likelihood=self.use_loocv_pseudo_likelihood,
            **self._kwargs,
        )
        # pyre-ignore: [28]
        return self.model_predictor(
            model=model, X=X_test, use_posterior_predictive=use_posterior_predictive
        )

    def feature_importances(self) -> npt.NDArray:
        return get_feature_importances_from_botorch_model(model=self._model)

    @property
    def search_space_digest(self) -> SearchSpaceDigest:
        if self._search_space_digest is None:
            raise RuntimeError(
                "`search_space_digest` is not initialized. Please fit the model first."
            )
        return self._search_space_digest

    @search_space_digest.setter
    def search_space_digest(self, value: SearchSpaceDigest) -> None:
        raise RuntimeError("Setting search_space_digest manually is disallowed.")

    @property
    def model(self) -> Model:
        if self._model is None:
            raise RuntimeError(
                "`model` is not initialized. Please fit the model first."
            )
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model  # there are a few places that set model directly


def get_rounding_func(
    rounding_func: Callable[[Tensor], Tensor] | None,
) -> Callable[[Tensor], Tensor] | None:
    if rounding_func is None:
        botorch_rounding_func = rounding_func
    else:
        # make sure rounding_func is properly applied to q- and t-batches
        def botorch_rounding_func(X: Tensor) -> Tensor:
            batch_shape, d = X.shape[:-1], X.shape[-1]
            X_round = torch.stack(
                [rounding_func(x) for x in X.view(-1, d)]  # pyre-ignore: [16]
            )
            return X_round.view(*batch_shape, d)

    return botorch_rounding_func


def get_feature_importances_from_botorch_model(
    model: Model | ModuleList | None,
) -> npt.NDArray:
    """Get feature importances from a list of BoTorch models.

    Args:
        models: BoTorch model to get feature importances from.

    Returns:
        The feature importances as a numpy array where each row sums to 1.
    """
    if model is None:
        raise RuntimeError(
            "Cannot calculate feature_importances without a fitted model."
            "Call `fit` first."
        )
    elif isinstance(model, ModelList):
        models = model.models
    else:
        models = [model]
    lengthscales = []
    for m in models:
        try:
            # this can be a ModelList of a SAAS and STGP, so this is a necessary way
            # to get the lengthscale
            if hasattr(m.covar_module, "base_kernel"):
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of...
                ls = m.covar_module.base_kernel.lengthscale
            else:
                # pyre-fixme[16]: Undefined attribute: Item `torch._tensor.Tensor` of...
                ls = m.covar_module.lengthscale
        except AttributeError:
            ls = None
        # pyre-fixme[29]: Call error: `typing.Union[BoundMethod[typing.Callable(torch...
        if ls is None or ls.shape[-1] != m.train_inputs[0].shape[-1]:
            # TODO: We could potentially set the feature importances to NaN in this
            # case, but this require knowing the batch dimension of this model.
            # Consider supporting in the future.
            raise NotImplementedError(
                "Failed to extract lengthscales from `m.covar_module` "
                "and `m.covar_module.base_kernel`"
            )
        if ls.ndim == 2:
            ls = ls.unsqueeze(0)
        # pyre-fixme[6]: Incompatible parameter type: In call `is_ensemble`, for 1st ...
        if is_ensemble(m):  # Take the median over the model batch dimension
            ls = torch.quantile(ls, q=0.5, dim=0, keepdim=True)
        lengthscales.append(ls)
    lengthscales = torch.cat(lengthscales, dim=0)
    feature_importances = (1 / lengthscales).detach().cpu()  # pyre-ignore
    # Make sure the sum of feature importances is 1.0 for each metric
    feature_importances /= feature_importances.sum(dim=-1, keepdim=True)
    return feature_importances.numpy()
