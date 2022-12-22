#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Models and utilities for fully bayesian inference.

TODO: move some of this into botorch.

References

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.

.. [Eriksson2021nas]
    D. Eriksson, P. Chuang, S. Daulton, et al. Latency-Aware Neural
    Architecture Search with Multi-Objective Bayesian Optimization.
    ICML AutoML Workshop, 2021.

"""

import math
import sys
import time
import types
import warnings

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from ax.exceptions.core import AxError
from ax.models.torch.botorch import (
    BotorchModel,
    TAcqfConstructor,
    TBestPointRecommender,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    get_NEI,
    MIN_OBSERVED_NOISE_LEVEL,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import get_NEHVI, pareto_frontier_evaluator
from ax.models.torch.frontier_utils import TFrontierEvaluator
from ax.models.torch.fully_bayesian_model_utils import (
    _get_single_task_gpytorch_model,
    load_mcmc_samples_to_model,
    pyro_sample_input_warping,
    pyro_sample_mean,
    pyro_sample_noise,
    pyro_sample_outputscale,
    pyro_sample_saas_lengthscales,
)
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast
from botorch.acquisition import AcquisitionFunction
from botorch.models.fully_bayesian import _psd_safe_pyro_mvn_sample
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import Tensor

logger: Logger = get_logger(__name__)


SAAS_DEPRECATION_MSG = (
    "Passing `use_saas` is no longer supported and has no effect. "
    "SAAS priors are used by default. "
    "This will become an error in the future."
)


def predict_from_model_mcmc(model: Model, X: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Predicts outcomes given a model and input tensor.

    This method integrates over the hyperparameter posterior.

    Args:
        model: A batched botorch Model where the batch dimension corresponds
            to sampled hyperparameters.
        X: A `n x d` tensor of input parameters.

    Returns:
        Tensor: The predicted posterior mean as an `n x o`-dim tensor.
        Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
    """
    with torch.no_grad():
        # compute the batch (independent posterior over the inputs)
        posterior = checked_cast(GPyTorchPosterior, model.posterior(X.unsqueeze(-3)))
    # the mean and variance both have shape: n x num_samples x m (after squeezing)
    mean = posterior.mean.cpu().detach()
    # TODO: Allow Posterior to (optionally) return the full covariance matrix
    variance = posterior.variance.cpu().detach().clamp_min(0)
    # marginalize over samples
    t1 = variance.sum(dim=0) / variance.shape[0]
    t2 = mean.pow(2).sum(dim=0) / variance.shape[0]
    t3 = -(mean.sum(dim=0) / variance.shape[0]).pow(2)
    variance = t1 + t2 + t3
    mean = mean.mean(dim=0)
    cov = torch.diag_embed(variance)
    return mean, cov


def compute_dists(X: Tensor, Z: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances.

    TODO: use gpytorch `Distance` module. This will require some care to make sure
    jit compilation works as expected.
    """
    mean = X.mean(dim=0)
    X_ = (X - mean).div(lengthscale)
    Z_ = (Z - mean).div(lengthscale)
    x1 = X_
    x2 = Z_
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    # x1 and x2 should be identical in all dims except -2 at this point
    x2 = x2 - adjustment
    x1_eq_x2 = torch.equal(x1, x2)

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2_norm, x2_pad = x1_norm, x1_pad
    else:
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    dist = res.clamp_min_(1e-30).sqrt()
    return dist


def matern_kernel(X: Tensor, Z: Tensor, lengthscale: Tensor, nu: float = 2.5) -> Tensor:
    """Scaled Matern kernel."""
    dist = compute_dists(X=X, Z=Z, lengthscale=lengthscale)
    exp_component = torch.exp(-math.sqrt(nu * 2) * dist)

    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = (math.sqrt(3) * dist).add(1)
    elif nu == 2.5:
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        constant_component = (math.sqrt(5) * dist).add(1).add(5.0 / 3.0 * (dist**2))
    else:
        raise AxError(f"Unsupported value of nu: {nu}")
    return constant_component * exp_component


def rbf_kernel(X: Tensor, Z: Tensor, lengthscale: Tensor) -> Tensor:
    """Scaled RBF kernel."""
    dist = compute_dists(X=X, Z=Z, lengthscale=lengthscale)
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    return torch.exp(-0.5 * (dist**2))


def single_task_pyro_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    use_input_warping: bool = False,
    eps: float = 1e-7,
    gp_kernel: str = "matern",
    task_feature: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    r"""Instantiates a single task pyro model for running fully bayesian inference.

    Args:
        X: A `n x d` tensor of input parameters.
        Y: A `n x 1` tensor of output.
        Yvar: A `n x 1` tensor of observed noise.
        use_input_warping: A boolean indicating whether to use input warping
        task_feature: Column index of task feature in X.
        gp_kernel: kernel name. Currently only two kernels are supported: "matern" for
            Matern Kernel and "rbf" for RBFKernel.
        rank: num of latent task features to learn for task covariance.
    """
    Y = Y.view(-1)
    Yvar = Yvar.view(-1)
    tkwargs = {"dtype": X.dtype, "device": X.device}
    dim = X.shape[-1]
    # TODO: test alternative outputscale priors
    outputscale = pyro_sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
    mean = pyro_sample_mean(**tkwargs)
    if torch.isnan(Yvar).all():
        # infer noise level
        noise = MIN_OBSERVED_NOISE_LEVEL + pyro_sample_noise(**tkwargs)
    else:
        noise = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)
    # pyre-fixme[6]: For 2nd param expected `float` but got `Union[device, dtype]`.
    lengthscale = pyro_sample_saas_lengthscales(dim=dim, **tkwargs)

    # transform inputs through kumaraswamy cdf
    if use_input_warping:
        c0, c1 = pyro_sample_input_warping(dim=dim, **tkwargs)
        # unnormalize X from [0, 1] to [eps, 1-eps]
        X = (X * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
        X_tf = 1 - torch.pow((1 - torch.pow(X, c1)), c0)
    else:
        X_tf = X
    # compute kernel
    if gp_kernel == "matern":
        k = matern_kernel(X=X_tf, Z=X_tf, lengthscale=lengthscale)
    elif gp_kernel == "rbf":
        k = rbf_kernel(X=X_tf, Z=X_tf, lengthscale=lengthscale)
    else:
        raise ValueError(f"Expected kernel to be 'rbf' or 'matern', got {gp_kernel}")

    # add noise
    k = outputscale * k + noise * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)

    _psd_safe_pyro_mvn_sample(
        name="Y",
        loc=mean.view(-1).expand(X.shape[0]),
        covariance_matrix=k,
        obs=Y,
    )


def _get_model_mcmc_samples(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    num_samples: int = 256,
    warmup_steps: int = 512,
    thinning: int = 16,
    max_tree_depth: int = 6,
    disable_progbar: bool = False,
    gp_kernel: str = "matern",
    verbose: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    pyro_model: Callable = single_task_pyro_model,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    get_gpytorch_model: Callable = _get_single_task_gpytorch_model,
    rank: Optional[int] = 1,
    **kwargs: Any,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
    #  to avoid runtime subscripting errors.
) -> Tuple[ModelListGP, List[Dict]]:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data and
    fit the model based on MCMC in pyro.

    Args:
        pyro_model: callable to instantiate a pyro model for running MCMC
        get_gpytorch_model: callable to instantiate a coupled GPyTorchModel to load the
        returned MCMC samples.
    """
    model = get_gpytorch_model(
        Xs=Xs,
        Ys=Ys,
        Yvars=Yvars,
        task_features=task_features,
        fidelity_features=fidelity_features,
        state_dict=state_dict,
        num_samples=num_samples,
        thinning=thinning,
        use_input_warping=use_input_warping,
        gp_kernel=gp_kernel,
        **kwargs,
    )
    if state_dict is not None:
        # Expected `OrderedDict[typing.Any, typing.Any]` for 1st
        #  param but got `Dict[str, Tensor]`.
        model.load_state_dict(state_dict)

    mcmc_samples_list = []
    if len(task_features) > 0:
        task_feature = task_features[0]
    else:
        task_feature = None
    if state_dict is None or refit_model:
        for X, Y, Yvar in zip(Xs, Ys, Yvars):
            mcmc_samples = run_inference(
                pyro_model=pyro_model,
                X=X,
                Y=Y,
                Yvar=Yvar,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                thinning=thinning,
                use_input_warping=use_input_warping,
                max_tree_depth=max_tree_depth,
                disable_progbar=disable_progbar,
                gp_kernel=gp_kernel,
                verbose=verbose,
                task_feature=task_feature,
                rank=rank,
            )
            mcmc_samples_list.append(mcmc_samples)
    return model, mcmc_samples_list


def get_and_fit_model_mcmc(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    metric_names: List[str],
    state_dict: Optional[Dict[str, Tensor]] = None,
    refit_model: bool = True,
    use_input_warping: bool = False,
    use_loocv_pseudo_likelihood: bool = False,
    num_samples: int = 256,
    warmup_steps: int = 512,
    thinning: int = 16,
    max_tree_depth: int = 6,
    disable_progbar: bool = False,
    gp_kernel: str = "matern",
    verbose: bool = False,
    **kwargs: Any,
) -> GPyTorchModel:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data and
    fit the model based on MCMC in pyro. The batch dimension corresponds to sampled
    hyperparameters from MCMC.
    """
    model, mcmc_samples_list = _get_model_mcmc_samples(
        Xs=Xs,
        Ys=Ys,
        Yvars=Yvars,
        task_features=task_features,
        fidelity_features=fidelity_features,
        metric_names=metric_names,
        state_dict=state_dict,
        refit_model=refit_model,
        use_input_warping=use_input_warping,
        use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        thinning=thinning,
        max_tree_depth=max_tree_depth,
        disable_progbar=disable_progbar,
        gp_kernel=gp_kernel,
        verbose=verbose,
        pyro_model=single_task_pyro_model,
        get_gpytorch_model=_get_single_task_gpytorch_model,
    )
    for i, mcmc_samples in enumerate(mcmc_samples_list):
        load_mcmc_samples_to_model(model=model.models[i], mcmc_samples=mcmc_samples)
    return model


def run_inference(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    pyro_model: Callable,
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    num_samples: int = 256,
    warmup_steps: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    max_tree_depth: int = 6,
    disable_progbar: bool = False,
    gp_kernel: str = "matern",
    verbose: bool = False,
    task_feature: Optional[int] = None,
    rank: Optional[int] = None,
    jit_compile: bool = False,
) -> Dict[str, Tensor]:
    start = time.time()
    try:
        # @manual=fbsource//third-party/pypi/pyro-ppl:pyro-ppl
        from pyro.infer.mcmc import MCMC, NUTS

        # @manual=fbsource//third-party/pypi/pyro-ppl:pyro-ppl
        from pyro.infer.mcmc.util import print_summary
    except ImportError:  # pragma: no cover
        raise RuntimeError("Cannot call run_inference without pyro installed!")
    kernel = NUTS(
        pyro_model,
        jit_compile=jit_compile,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=disable_progbar,
    )
    mcmc.run(
        X,
        Y,
        Yvar,
        use_input_warping=use_input_warping,
        gp_kernel=gp_kernel,
        task_feature=task_feature,
        rank=rank,
    )

    # compute the true lengthscales and get rid of the temporary variables
    samples = mcmc.get_samples()
    inv_length_sq = (
        samples["kernel_tausq"].unsqueeze(-1) * samples["_kernel_inv_length_sq"]
    )
    samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()  # pyre-ignore [16]
    del samples["kernel_tausq"], samples["_kernel_inv_length_sq"]
    # this prints the summary
    if verbose:
        orig_std_out = sys.stdout.write
        sys.stdout.write = logger.info  # pyre-fixme[8]
        print_summary(samples, prob=0.9, group_by_chain=False)
        sys.stdout.write = orig_std_out
        logger.info(f"MCMC elapsed time: {time.time() - start}")
    # thin
    for k, v in samples.items():
        samples[k] = v[::thinning]  # apply thinning
    return samples


def get_fully_bayesian_acqf(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    # pyre-fixme[9]: acqf_constructor has type `Callable[[Model, Tensor,
    #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor], Any],
    #  AcquisitionFunction]`; used as `Callable[[Model, Tensor,
    #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor],
    #  **(Any)], AcquisitionFunction]`.
    acqf_constructor: TAcqfConstructor = get_NEI,
    **kwargs: Any,
) -> AcquisitionFunction:
    kwargs["marginalize_dim"] = -3
    # pyre-ignore [28]
    acqf = acqf_constructor(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        X_pending=X_pending,
        **kwargs,
    )
    base_forward = acqf.forward

    # pyre-fixme[53]: Captured variable `base_forward` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, X):
        # unsqueeze dim for GP hyperparameter samples
        return base_forward(X.unsqueeze(-3)).mean(dim=-1)

    acqf.forward = types.MethodType(forward, acqf)  # pyre-ignore[8]
    return acqf


def get_fully_bayesian_acqf_nehvi(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
    X_pending: Optional[Tensor] = None,
    **kwargs: Any,
) -> AcquisitionFunction:
    return get_fully_bayesian_acqf(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        X_pending=X_pending,
        acqf_constructor=get_NEHVI,  # pyre-ignore [6]
        **kwargs,
    )


class FullyBayesianBotorchModelMixin:
    model: Optional[Model] = None

    def feature_importances(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "Cannot calculate feature_importances without a fitted model"
            )
        elif isinstance(self.model, ModelListGP):
            models = self.model.models
        else:
            models = [self.model]
        lengthscales = []
        for m in models:
            ls = m.covar_module.base_kernel.lengthscale
            lengthscales.append(ls)
        lengthscales = torch.stack(lengthscales, dim=0)
        # take mean over MCMC samples
        lengthscales = torch.quantile(lengthscales, 0.5, dim=1)
        # pyre-ignore [16]
        # pyre-fixme[58]: `/` is not supported for operand types `int` and `Tensor`.
        return (1 / lengthscales).detach().cpu().numpy()


class FullyBayesianBotorchModel(FullyBayesianBotorchModelMixin, BotorchModel):
    r"""Fully Bayesian Model that uses NUTS to sample from hyperparameter posterior.

    This includes support for using sparse axis-aligned subspace priors (SAAS). See
    [Eriksson2021saasbo]_ for details.
    """

    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model_mcmc,
        model_predictor: TModelPredictor = predict_from_model_mcmc,
        acqf_constructor: TAcqfConstructor = get_fully_bayesian_acqf,
        # pyre-fixme[9]: acqf_optimizer declared/used type mismatch
        acqf_optimizer: TOptimizer = scipy_optimizer,
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = True,
        use_input_warping: bool = False,
        # use_saas is deprecated. TODO: remove
        use_saas: Optional[bool] = None,
        num_samples: int = 256,
        warmup_steps: int = 512,
        thinning: int = 16,
        max_tree_depth: int = 6,
        disable_progbar: bool = False,
        gp_kernel: str = "matern",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Fully Bayesian Botorch Model.

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
            refit_on_update: If True, refit the model after updating the training
                data using the `update` method.
            warm_start_refitting: If True, start model refitting from previous
                model parameters in order to speed up the fitting process.
            use_input_warping: A boolean indicating whether to use input warping
            use_saas: [deprecated] A boolean indicating whether to use the SAAS model
            num_samples: The number of MCMC samples. Note that with thinning,
                num_samples/thinning samples are retained.
            warmup_steps: The number of burn-in steps for NUTS.
            thinning: The amount of thinning. Every nth sample is retained.
            max_tree_depth: The max_tree_depth for NUTS.
            disable_progbar: A boolean indicating whether to print the progress
                bar and diagnostics during MCMC.
            gp_kernel: The type of ARD base kernel. "matern" corresponds to a Matern-5/2
                kernel and "rbf" corresponds to an RBF kernel.
            verbose: A boolean indicating whether to print summary stats from MCMC.
        """
        # use_saas is deprecated. TODO: remove
        if use_saas is not None:
            warnings.warn(SAAS_DEPRECATION_MSG, DeprecationWarning)
        BotorchModel.__init__(
            self,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            best_point_recommender=best_point_recommender,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            warm_start_refitting=warm_start_refitting,
            use_input_warping=use_input_warping,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            thinning=thinning,
            max_tree_depth=max_tree_depth,
            disable_progbar=disable_progbar,
            gp_kernel=gp_kernel,
            verbose=verbose,
        )


class FullyBayesianMOOBotorchModel(
    FullyBayesianBotorchModelMixin, MultiObjectiveBotorchModel
):
    r"""Fully Bayesian Model that uses qNEHVI.

    This includes support for using qNEHVI + SAASBO as in [Eriksson2021nas]_.
    """

    @copy_doc(FullyBayesianBotorchModel.__init__)
    def __init__(
        self,
        model_constructor: TModelConstructor = get_and_fit_model_mcmc,
        model_predictor: TModelPredictor = predict_from_model_mcmc,
        # pyre-fixme[9]: acqf_constructor has type `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor], Any],
        #  AcquisitionFunction]`; used as `Callable[[Model, Tensor,
        #  Optional[Tuple[Tensor, Tensor]], Optional[Tensor], Optional[Tensor],
        #  **(Any)], AcquisitionFunction]`.
        acqf_constructor: TAcqfConstructor = get_fully_bayesian_acqf_nehvi,
        # pyre-fixme[9]: acqf_optimizer has type `Callable[[AcquisitionFunction,
        #  Tensor, int, Optional[Dict[int, float]], Optional[Callable[[Tensor],
        #  Tensor]], Any], Tensor]`; used as `Callable[[AcquisitionFunction, Tensor,
        #  int, Optional[Dict[int, float]], Optional[Callable[[Tensor], Tensor]],
        #  **(Any)], Tensor]`.
        acqf_optimizer: TOptimizer = scipy_optimizer,
        # TODO: Remove best_point_recommender for botorch_moo. Used in modelbridge._gen.
        best_point_recommender: TBestPointRecommender = recommend_best_observed_point,
        frontier_evaluator: TFrontierEvaluator = pareto_frontier_evaluator,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        warm_start_refitting: bool = False,
        use_input_warping: bool = False,
        num_samples: int = 256,
        warmup_steps: int = 512,
        thinning: int = 16,
        max_tree_depth: int = 6,
        # use_saas is deprecated. TODO: remove
        use_saas: Optional[bool] = None,
        disable_progbar: bool = False,
        gp_kernel: str = "matern",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # use_saas is deprecated. TODO: remove
        if use_saas is not None:
            warnings.warn(SAAS_DEPRECATION_MSG, DeprecationWarning)
        MultiObjectiveBotorchModel.__init__(
            self,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            best_point_recommender=best_point_recommender,
            frontier_evaluator=frontier_evaluator,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            warm_start_refitting=warm_start_refitting,
            use_input_warping=use_input_warping,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            thinning=thinning,
            max_tree_depth=max_tree_depth,
            disable_progbar=disable_progbar,
            gp_kernel=gp_kernel,
            verbose=verbose,
        )
