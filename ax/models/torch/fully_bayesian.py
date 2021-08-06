#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from ax.exceptions.core import AxError
from ax.models.torch.botorch import (
    BotorchModel,
    TModelConstructor,
    TModelPredictor,
    TAcqfConstructor,
    TOptimizer,
    TBestPointRecommender,
)
from ax.models.torch.botorch_defaults import (
    _get_model,
    get_NEI,
    recommend_best_observed_point,
    scipy_optimizer,
)
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import get_NEHVI
from ax.models.torch.botorch_moo_defaults import pareto_frontier_evaluator
from ax.models.torch.frontier_utils import TFrontierEvaluator
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.acquisition import AcquisitionFunction
from botorch.fit import _set_transformed_inputs
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor

logger = get_logger(__name__)

MIN_OBSERVED_NOISE_LEVEL_MCMC = 1e-6


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
    num_samples: int = 512,
    warmup_steps: int = 1024,
    thinning: int = 16,
    max_tree_depth: int = 6,
    use_saas: bool = False,
    disable_progbar: bool = False,
    **kwargs: Any,
) -> GPyTorchModel:
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    model = None
    # TODO: Better logic for deciding when to use a ModelListGP. Currently the
    # logic is unclear. The two cases in which ModelListGP is used are
    # (i) the training inputs (Xs) are not the same for the different outcomes, and
    # (ii) a multi-task model is used

    num_mcmc_samples = num_samples // thinning
    if len(Xs) == 1:
        # Use single output, single task GP
        model = _get_model(
            X=Xs[0].unsqueeze(0).expand(num_mcmc_samples, Xs[0].shape[0], -1),
            Y=Ys[0].unsqueeze(0).expand(num_mcmc_samples, Xs[0].shape[0], -1),
            Yvar=Yvars[0].unsqueeze(0).expand(num_mcmc_samples, Xs[0].shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            **kwargs,
        )
    else:
        models = [
            _get_model(
                X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1).clone(),
                Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1).clone(),
                Yvar=Yvar.unsqueeze(0)
                .expand(num_mcmc_samples, Yvar.shape[0], -1)
                .clone(),
                use_input_warping=use_input_warping,
                **kwargs,
            )
            for X, Y, Yvar in zip(Xs, Ys, Yvars)
        ]
        model = ModelListGP(*models)
    model.to(Xs[0])
    if isinstance(model, ModelListGP):
        models = model.models
    else:
        models = [model]
    if state_dict is not None:
        # pyre-fixme[6]: Expected `OrderedDict[typing.Any, typing.Any]` for 1st
        #  param but got `Dict[str, Tensor]`.
        model.load_state_dict(state_dict)
    if state_dict is None or refit_model:
        for X, Y, Yvar, m in zip(Xs, Ys, Yvars, models):
            samples = run_inference(
                pyro_model=pyro_model,  # pyre-ignore [6]
                X=X,
                Y=Y,
                Yvar=Yvar,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                thinning=thinning,
                use_input_warping=use_input_warping,
                use_saas=use_saas,
                max_tree_depth=max_tree_depth,
                disable_progbar=disable_progbar,
            )
            if "noise" in samples:
                m.likelihood.noise_covar.noise = (
                    samples["noise"]
                    .detach()
                    .clone()
                    .view(m.likelihood.noise_covar.noise.shape)
                    .clamp_min(MIN_INFERRED_NOISE_LEVEL)
                )
            m.covar_module.base_kernel.lengthscale = (
                samples["lengthscale"]
                .detach()
                .clone()
                .view(m.covar_module.base_kernel.lengthscale.shape)
            )
            m.covar_module.outputscale = (
                samples["outputscale"]
                .detach()
                .clone()
                .view(m.covar_module.outputscale.shape)
            )
            m.mean_module.constant.data = (
                samples["mean"].detach().clone().view(m.mean_module.constant.shape)
            )
            if "c0" in samples:
                m.input_transform._set_concentration(
                    i=0,
                    value=samples["c0"]
                    .detach()
                    .clone()
                    .view(m.input_transform.concentration0.shape),
                )
                m.input_transform._set_concentration(
                    i=1,
                    value=samples["c1"]
                    .detach()
                    .clone()
                    .view(m.input_transform.concentration1.shape),
                )
    _set_transformed_inputs(model=model)
    return model


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
        posterior = model.posterior(X.unsqueeze(-3))
    # the mean and variance both have shape: n x num_samples x m (after squeezing)
    mean = posterior.mean.cpu().detach()
    # TODO: Allow Posterior to (optionally) return the full covariance matrix
    # pyre-ignore
    variance = posterior.variance.cpu().detach().clamp_min(0)
    # marginalize over samples
    t1 = variance.sum(dim=0) / variance.shape[0]
    t2 = mean.pow(2).sum(dim=0) / variance.shape[0]
    t3 = -(mean.sum(dim=0) / variance.shape[0]).pow(2)
    variance = t1 + t2 + t3
    mean = mean.mean(dim=0)
    cov = torch.diag_embed(variance)
    return mean, cov


def matern_kernel(X: Tensor, Z: Tensor, lengthscale: Tensor, nu: float = 2.5) -> Tensor:
    """Scaled Matern kernel.

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
        res.diagonal(dim1=-2, dim2=-1).fill_(0)  # pyre-ignore [16]

    # Zero out negative values
    dist = res.clamp_min_(1e-30).sqrt_()
    exp_component = torch.exp(-math.sqrt(nu * 2) * dist)

    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = (math.sqrt(3) * dist).add(1)  # pyre-ignore [16]
    elif nu == 2.5:
        constant_component = (math.sqrt(5) * dist).add(1).add(5.0 / 3.0 * dist ** 2)
    else:
        raise AxError(f"Unsupported value of nu: {nu}")
    return constant_component * exp_component


def pyro_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    use_input_warping: bool = False,
    use_saas: bool = False,
    eps: float = 1e-7,
) -> None:
    try:
        import pyro
    except ImportError:  # pragma: no cover
        raise RuntimeError("Cannot call pyro_model without pyro installed!")
    Y = Y.view(-1)
    Yvar = Yvar.view(-1)
    tkwargs = {"dtype": X.dtype, "device": X.device}
    dim = X.shape[-1]
    # TODO: test alternative outputscale priors
    outputscale = pyro.sample(
        "outputscale",
        pyro.distributions.Gamma(  # pyre-ignore [16]
            # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
            #  `Union[torch.device, torch.dtype]`.
            torch.tensor(2.0, **tkwargs),
            # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
            #  `Union[torch.device, torch.dtype]`.
            torch.tensor(0.15, **tkwargs),
        ),
    )
    mean = pyro.sample(
        "mean",
        pyro.distributions.Uniform(  # pyre-ignore [16]
            # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
            #  `Union[torch.device, torch.dtype]`.
            torch.tensor(-1.0, **tkwargs),
            # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but got
            #  `Union[torch.device, torch.dtype]`.
            torch.tensor(1.0, **tkwargs),
        ),
    )
    if torch.isnan(Yvar).all():
        # infer noise level
        noise = pyro.sample(
            "noise",
            pyro.distributions.Gamma(  # pyre-ignore [16]
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor(0.9, **tkwargs),
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor(10.0, **tkwargs),
            ),
        )
    else:
        # pyre-ignore [16]
        noise = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL_MCMC)

    if use_saas:
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(  # pyre-ignore [16]
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor(0.1, **tkwargs)
            ),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(  # pyre-ignore [16]
                torch.ones(dim, **tkwargs)
            ),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            (1.0 / inv_length_sq).sqrt(),  # pyre-ignore [16]
        )
    else:
        lengthscale = pyro.sample(
            "lengthscale",
            # pyro.distributions.Uniform does not jit-compile with
            # vector parameters, so use expand() instead on a
            # distribution with scalar parameters.
            # https://github.com/pyro-ppl/pyro/issues/2810
            pyro.distributions.Uniform(  # pyre-ignore [16]
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor(0.0, **tkwargs),
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor(10.0, **tkwargs),
            ).expand(torch.Size([dim])),
        )

    # transform inputs through kumaraswamy cdf
    if use_input_warping:
        c0 = pyro.sample(
            "c0",
            pyro.distributions.LogNormal(  # pyre-ignore [16]
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor([0.0] * dim, **tkwargs),
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor([0.75 ** 0.5] * dim, **tkwargs),
            ),
        )
        c1 = pyro.sample(
            "c1",
            pyro.distributions.LogNormal(  # pyre-ignore [16]
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor([0.0] * dim, **tkwargs),
                # pyre-fixme[6]: Expected `Optional[torch.dtype]` for 2nd param but
                #  got `Union[torch.device, torch.dtype]`.
                torch.tensor([0.75 ** 0.5] * dim, **tkwargs),
            ),
        )
        # unnormalize X from [0, 1] to [eps, 1-eps]
        X = (X * (1 - 2 * eps) + eps).clamp(eps, 1 - eps)
        X_tf = 1 - torch.pow((1 - torch.pow(X, c1)), c0)
    else:
        X_tf = X
    # compute kernel
    k = matern_kernel(X=X_tf, Z=X_tf, lengthscale=lengthscale)
    # add noise
    k = outputscale * k + noise * torch.eye(X.shape[0], dtype=X.dtype, device=X.device)

    pyro.sample(
        "Y",
        pyro.distributions.MultivariateNormal(  # pyre-ignore [16]
            loc=mean.view(-1).expand(X.shape[0]), covariance_matrix=k
        ),
        obs=Y,
    )


def run_inference(
    pyro_model: Callable[[Tensor, Tensor, Tensor, bool, str, float], None],
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    num_samples: int = 512,
    warmup_steps: int = 1024,
    thinning: int = 16,
    use_input_warping: bool = False,
    max_tree_depth: int = 6,
    use_saas: bool = False,
    disable_progbar: bool = False,
) -> Tensor:
    start = time.time()
    try:
        from pyro.infer.mcmc import NUTS, MCMC
    except ImportError:  # pragma: no cover
        raise RuntimeError("Cannot call run_inference without pyro installed!")
    kernel = NUTS(
        pyro_model,
        jit_compile=True,
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
        # there is an issue with jit-compilation and cuda
        # for now, we run MCMC on the CPU.
        X.cpu(),
        Y.cpu(),
        Yvar.cpu(),
        use_input_warping=use_input_warping,
        use_saas=use_saas,
    )
    # this prints the summary
    orig_std_out = sys.stdout.write
    sys.stdout.write = logger.info
    mcmc.summary()
    sys.stdout.write = orig_std_out
    logger.info(f"MCMC elapsed time: {time.time() - start}")
    samples = mcmc.get_samples()
    if use_saas:  # compute the lengthscale for saas and throw away everything else
        inv_length_sq = (
            samples["kernel_tausq"].unsqueeze(-1) * samples["_kernel_inv_length_sq"]
        )
        samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()  # pyre-ignore [16]
        del samples["kernel_tausq"], samples["_kernel_inv_length_sq"]
    # thin
    for k, v in samples.items():
        # apply thinning and move back to X's device
        samples[k] = v[::thinning].to(device=X.device)
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
            models = self.model.models  # pyre-ignore: [16]
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
        use_saas: bool = False,
        num_samples: int = 512,
        warmup_steps: int = 1024,
        thinning: int = 16,
        max_tree_depth: int = 6,
        disable_progbar: bool = False,
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
            use_saas: A boolean indicating whether to use the SAAS model
            num_samples: The number of MCMC samples. Note that with thinning,
                num_samples/thinning samples are retained.
            warmup_steps: The number of burn-in steps for NUTS.
            thinning: The amount of thinning. Every nth sample is retained.
            max_tree_depth: The max_tree_depth for NUTS.
            disable_progbar: A boolean indicating whether to print the progress
                bar and diagnostics during MCMC.
        """
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
            use_saas=use_saas,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            thinning=thinning,
            max_tree_depth=max_tree_depth,
            disable_progbar=disable_progbar,
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
        num_samples: int = 512,
        warmup_steps: int = 1024,
        thinning: int = 16,
        max_tree_depth: int = 6,
        use_saas: bool = False,
        disable_progbar: bool = False,
        **kwargs: Any,
    ) -> None:
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
            use_saas=use_saas,
            disable_progbar=disable_progbar,
        )
