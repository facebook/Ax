#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.models.random.alebo_initializer import ALEBOInitializer
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.fit import fit_gpytorch_scipy
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.optim.numpy_converter import module_to_array
from botorch.optim.optimize import optimize_acqf
from botorch.optim.utils import _scipy_objective_and_grad
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.rbf_kernel import postprocess_rbf
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from scipy.optimize import approx_fprime
from torch import Tensor


logger = get_logger(name="ALEBO")


class ALEBOKernel(Kernel):
    """The kernel for ALEBO.

    Suppose there exists an ARD RBF GP on an (unknown) linear embedding with
    projection matrix A. We make function evaluations in a different linear
    embedding with projection matrix B (known). This is the appropriate kernel
    for fitting those data.

    This kernel computes a Mahalanobis distance, and the (d x d) PD distance
    matrix Gamma is a parameter that must be fit. This is done by fitting its
    upper Cholesky decomposition, U.

    Args:
        B: (d x D) Projection matrix.
        batch_shape: Batch shape as usual for gpytorch kernels.
    """

    def __init__(self, B: Tensor, batch_shape: torch.Size) -> None:
        super().__init__(
            has_lengthscale=False, ard_num_dims=None, eps=0.0, batch_shape=batch_shape
        )
        self.d, D = B.shape
        assert self.d < D
        self.B = B
        # Initialize U
        Arnd = torch.randn(D, D, dtype=B.dtype, device=B.device)
        Arnd = torch.qr(Arnd)[0]
        ABinv = Arnd[: self.d, :] @ torch.pinverse(B)
        # U is the upper Cholesky decomposition of Gamma, the Mahalanobis
        # matrix. Uvec is the upper triangular portion of U squeezed out into
        # a vector.
        U = torch.cholesky(torch.mm(ABinv.t(), ABinv), upper=True)
        self.triu_indx = torch.triu_indices(self.d, self.d, device=B.device)
        Uvec = U[self.triu_indx.tolist()].repeat(*batch_shape, 1)
        self.register_parameter(name="Uvec", parameter=torch.nn.Parameter(Uvec))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        # Unpack Uvec into an upper triangular matrix U
        shapeU = self.Uvec.shape[:-1] + torch.Size([self.d, self.d])
        U_t = torch.zeros(shapeU, dtype=self.B.dtype, device=self.B.device)
        U_t[..., self.triu_indx[1], self.triu_indx[0]] = self.Uvec
        # Compute kernel distance
        z1 = torch.matmul(x1, U_t)
        z2 = torch.matmul(x2, U_t)
        return self.covar_dist(
            z1,
            z2,
            square_dist=True,
            diag=diag,
            dist_postprocess_func=postprocess_rbf,
            postprocess=True,
            **params,
        )


class ALEBOGP(FixedNoiseGP):
    """The GP for ALEBO.

    Uses the Mahalanobis kernel defined in ALEBOKernel, along with a
    ScaleKernel to add a kernel variance and a fitted constant mean.

    In non-batch mode, there is a single kernel that produces MVN predictions
    as usual for a GP.
    With b batches, each batch has its own set of kernel hyperparameters and
    each batch represents a sample from the hyperparameter posterior
    distribution. When making a prediction (with `__call__`), these samples are
    integrated over using moment matching. So, the predictions are an MVN as
    usual with the same shape as in non-batch mode.

    Args:
        B: (d x D) Projection matrix.
        train_X: (n x d) X training data.
        train_Y: (n x 1) Y training data.
        train_Yvar: (n x 1) Noise variances of each training Y.
    """

    def __init__(
        self, B: Tensor, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor
    ) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        self.covar_module = ScaleKernel(
            base_kernel=ALEBOKernel(B=B, batch_shape=self._aug_batch_shape),
            batch_shape=self._aug_batch_shape,
        )
        self.to(train_X)

    def __call__(self, x: Tensor) -> MultivariateNormal:
        """
        If model is non-batch, then just make a prediction. If model has
        multiple batches, then these are samples from the kernel hyperparameter
        posterior and we integrate over them with moment matching.

        The shape of the MVN that this outputs will be the same regardless of
        whether the model is batched or not.

        Args:
            x: Point to be predicted.

        Returns: MultivariateNormal distribution of prediction.
        """
        if len(self._aug_batch_shape) == 0:
            return super().__call__(x)
        # Else, approximately integrate over batches with moment matching.
        # Take X as (b) x q x d, and expand to (b) x ns x q x d
        if x.ndim > 3:  # pyre-ignore
            raise ValueError("Don't know how to predict this shape")  # pragma: no cover
        x = x.unsqueeze(-3).expand(
            x.shape[:-2]
            + torch.Size([self._aug_batch_shape[0]])  # pyre-ignore
            + x.shape[-2:]
        )
        mvn_b = super().__call__(x)
        mu = mvn_b.mean.mean(dim=-2)
        C = (
            mvn_b.covariance_matrix.mean(dim=-3)
            + torch.matmul(mvn_b.mean.transpose(-2, -1), mvn_b.mean)
            / mvn_b.mean.shape[-2]
            - torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
        )  # Law of Total Covariance
        mvn = MultivariateNormal(mu, C)
        return mvn

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        assert output_indices is None
        assert not observation_noise
        mvn = self(X)
        return GPyTorchPosterior(mvn=mvn)


def get_fitted_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    restarts: int,
    nsamp: int,
    init_state_dict: Optional[Dict[str, Tensor]],
) -> ALEBOGP:
    """Get a fitted ALEBO GP.

    We do random restart optimization to get a MAP model, then use the Laplace
    approximation to draw posterior samples of kernel hyperparameters, and
    finally construct a batch-mode model where each batch is one of those
    sampled sets of kernel hyperparameters.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        restarts: Number of restarts for MAP estimation.
        nsamp: Number of samples to draw from kernel hyperparameter posterior.
        init_state_dict: Optionally begin MAP estimation with this state dict.

    Returns: Batch-mode (nsamp batches) fitted ALEBO GP.
    """
    # Get MAP estimate.
    mll = get_map_model(
        B=B,
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        restarts=restarts,
        init_state_dict=init_state_dict,
    )
    # Compute Laplace approximation of posterior
    Uvec_batch, mean_constant_batch, output_scale_batch = laplace_sample_U(
        mll=mll, nsamp=nsamp
    )
    # Construct batch model with samples
    m_b = get_batch_model(
        B=B,
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        Uvec_batch=Uvec_batch,
        mean_constant_batch=mean_constant_batch,
        output_scale_batch=output_scale_batch,
    )
    return m_b


def get_map_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    restarts: int,
    init_state_dict: Optional[Dict[str, Tensor]],
) -> ExactMarginalLogLikelihood:
    """Do random-restart optimization for MAP fitting of an ALEBO GP model.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        restarts: Number of restarts for MAP estimation.
        init_state_dict: Optionally begin MAP estimation with this state dict.

    Returns: non-batch ALEBO GP with MAP kernel hyperparameters.
    """
    f_best = 1e8
    sd_best = {}
    # Fit with random restarts
    for _ in range(restarts):
        m = ALEBOGP(B=B, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
        if init_state_dict is not None:
            m.load_state_dict(init_state_dict)
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        mll.train()
        mll, info_dict = fit_gpytorch_scipy(mll, track_iterations=False, method="tnc")
        logger.debug(info_dict)
        # pyre-fixme[6]: Expected `List[botorch.optim.fit.OptimizationIteration]`
        #  for 1st param but got `float`.
        # pyre-fixme[6]: Expected `List[botorch.optim.fit.OptimizationIteration]`
        #  for 1st param but got `float`.
        if info_dict["fopt"] < f_best:
            f_best = float(info_dict["fopt"])  # pyre-ignore
            sd_best = m.state_dict()
    # Set the final value
    m = ALEBOGP(B=B, train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
    m.load_state_dict(sd_best)
    mll = ExactMarginalLogLikelihood(m.likelihood, m)
    return mll


def laplace_sample_U(
    mll: ExactMarginalLogLikelihood, nsamp: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Draw posterior samples of kernel hyperparameters using Laplace
    approximation.

    Only the Mahalanobis distance matrix is sampled.

    The diagonal of the Hessian is estimated using finite differences of the
    autograd gradients. The Laplace approximation is then N(p_map, inv(-H)).
    We construct a set of nsamp kernel hyperparameters by drawing nsamp-1
    values from this distribution, and prepending as the first sample the MAP
    parameters.

    Args:
        mll: MLL object of MAP ALEBO GP.
        nsamp: Number of samples to return.

    Returns: Batch tensors of the kernel hyperparameters Uvec, mean constant,
        and output scale.
    """
    # Estimate diagonal of the Hessian
    mll.train()
    x0, property_dict, bounds = module_to_array(module=mll)
    x0 = x0.astype(np.float64)  # This is the MAP parameters
    H = np.zeros((len(x0), len(x0)))
    epsilon = 1e-4 + 1e-3 * np.abs(x0)
    for i, _ in enumerate(x0):
        # Compute gradient of df/dx_i wrt x_i
        def f(x):
            x_all = x0.copy()
            x_all[i] = x[0]
            return -_scipy_objective_and_grad(x_all, mll, property_dict)[1][i]

        H[i, i] = approx_fprime(np.array([x0[i]]), f, epsilon=epsilon)

    # Sample only Uvec; leave mean and output scale fixed.
    assert list(property_dict.keys()) == [
        "model.mean_module.constant",
        "model.covar_module.raw_outputscale",
        "model.covar_module.base_kernel.Uvec",
    ]
    H = H[2:, 2:]
    H += np.diag(-1e-3 * np.ones(H.shape[0]))  # Add a nugget for inverse stability
    Sigma = np.linalg.inv(-H)
    samples = np.random.multivariate_normal(mean=x0[2:], cov=Sigma, size=(nsamp - 1))
    # Include the MAP estimate
    samples = np.vstack((x0[2:], samples))
    # Reshape
    attrs = property_dict["model.covar_module.base_kernel.Uvec"]
    Uvec_batch = torch.tensor(samples, dtype=attrs.dtype, device=attrs.device).reshape(
        nsamp, *attrs.shape
    )
    # Get the other properties into batch mode
    mean_constant_batch = mll.model.mean_module.constant.repeat(nsamp, 1)
    output_scale_batch = mll.model.covar_module.raw_outputscale.repeat(nsamp)
    return Uvec_batch, mean_constant_batch, output_scale_batch


def get_batch_model(
    B: Tensor,
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    Uvec_batch: Tensor,
    mean_constant_batch: Tensor,
    output_scale_batch: Tensor,
) -> ALEBOGP:
    """Construct a batch-mode ALEBO GP using batch tensors of hyperparameters.

    Args:
        B: Projection matrix.
        train_X: X training data.
        train_Y: Y training data.
        train_Yvar: Noise variances of each training Y.
        Uvec_batch: Batch tensor of Uvec hyperparameters.
        mean_constant_batch: Batch tensor of mean constant hyperparameter.
        output_scale_batch: Batch tensor of output scale hyperparameter.

    Returns: Batch-mode ALEBO GP.
    """
    b = Uvec_batch.size(0)
    m_b = ALEBOGP(
        B=B,
        train_X=train_X.repeat(b, 1, 1),
        train_Y=train_Y.repeat(b, 1, 1),
        train_Yvar=train_Yvar.repeat(b, 1, 1),
    )
    m_b.train()
    # Set mean constant
    m_b.mean_module.constant.requires_grad_(False)
    m_b.mean_module.constant.copy_(mean_constant_batch)
    m_b.mean_module.constant.requires_grad_(True)
    # Set output scale
    m_b.covar_module.raw_outputscale.requires_grad_(False)
    m_b.covar_module.raw_outputscale.copy_(output_scale_batch)
    m_b.covar_module.raw_outputscale.requires_grad_(True)
    # Set Uvec
    m_b.covar_module.base_kernel.Uvec.requires_grad_(False)
    m_b.covar_module.base_kernel.Uvec.copy_(Uvec_batch)
    m_b.covar_module.base_kernel.Uvec.requires_grad_(True)
    m_b.eval()
    return m_b


def extract_map_statedict(
    m_b: Union[ALEBOGP, ModelListGP], num_outputs: int
) -> List[MutableMapping[str, Tensor]]:
    """Extract MAP statedict from the batch-mode ALEBO GP.

    The batch GP can be either a single ALEBO GP or a ModelListGP of ALEBO GPs.

    Args:
        m_b: Batch-mode GP.
        num_outputs: Number of outputs being modeled.
    """
    is_modellist = num_outputs > 1
    map_sds: List[MutableMapping[str, Tensor]] = [
        OrderedDict() for i in range(num_outputs)
    ]
    sd = m_b.state_dict()
    for k, v in sd.items():
        # Extract model index and parameter name
        if is_modellist:
            g = re.match(r"^models\.([0-9]+)\.(.*)$", k)
            if g is None:
                raise Exception(
                    "Unable to parse ModelList structure"
                )  # pragma: no cover
            model_idx = int(g.group(1))
            param_name = g.group(2)
        else:
            model_idx = 0
            param_name = k
        map_sds[model_idx][param_name] = torch.select(v, 0, 0)
    return map_sds


def ei_or_nei(
    model: Union[ALEBOGP, ModelListGP],
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]],
    X_observed: Tensor,
    X_pending: Optional[Tensor],
    q: int,
    noiseless: bool,
) -> AcquisitionFunction:
    """Use analytic EI if appropriate, otherwise Monte Carlo NEI.

    Analytic EI can be used if: Single outcome, no constraints, no pending
    points, not batch, and no noise.

    Args:
        model: GP.
        objective_weights: Weights on each outcome for the objective.
        outcome_constraints: Outcome constraints.
        X_observed: Observed points for NEI.
        X_pending: Pending points.
        q: Batch size.
        noiseless: True if evaluations are noiseless.

    Returns: An AcquisitionFunction, either analytic EI or MC NEI.
    """
    if (
        len(objective_weights) == 1
        and outcome_constraints is None
        and X_pending is None
        and q == 1
        and noiseless
    ):
        maximize = objective_weights[0] > 0
        if maximize:
            best_f = model.train_targets.max()
        else:
            best_f = model.train_targets.min()
        return ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
    else:
        with gpytorch.settings.max_cholesky_size(2000):
            acq = get_NEI(
                model=model,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                X_observed=X_observed,
                X_pending=X_pending,
            )
        return acq


def alebo_acqf_optimizer(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    n: int,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]],
    fixed_features: Optional[Dict[int, float]],
    rounding_func: Optional[Callable[[Tensor], Tensor]],
    raw_samples: int,
    num_restarts: int,
    B: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Optimize the acquisition function for ALEBO.

    We are optimizing over a polytope within the subspace, and so begin each
    random restart of the acquisition function optimization with points that
    lie within that polytope.
    """
    assert n == 1  # Handle batch later
    # Generate initial points for optimization inside embedding
    m_init = ALEBOInitializer(B.cpu().numpy(), nsamp=10 * raw_samples)
    Xrnd_npy, _ = m_init.gen(n=raw_samples, bounds=[(-1.0, 1.0)] * B.shape[1])

    Xrnd = torch.tensor(Xrnd_npy, dtype=B.dtype, device=B.device).unsqueeze(1)
    Yrnd = torch.matmul(Xrnd, B.t())  # Project down to the embedding
    with gpytorch.settings.max_cholesky_size(2000):
        with torch.no_grad():
            alpha = acq_function(Yrnd)

        Yinit = initialize_q_batch_nonneg(X=Yrnd, Y=alpha, n=num_restarts)

        # Optimize the acquisition function, separately for each random restart.
        Xopt = optimize_acqf(
            acq_function=acq_function,
            bounds=[None, None],  # pyre-ignore
            q=n,
            num_restarts=num_restarts,
            raw_samples=0,
            options={"method": "SLSQP", "batch_limit": 1},
            inequality_constraints=inequality_constraints,
            batch_initial_conditions=Yinit,
            sequential=False,
        )
    return Xopt


class ALEBO(BotorchModel):
    """Does Bayesian optimization in a linear subspace with ALEBO.

    The (d x D) projection down matrix B must be provided, and must be that
    used for the initialization.

    Function evaluations happen in the high-D space. We only evaluate points
    such that x = pinverse(B) @ B @ x (that is, points inside the subspace).
    Under that constraint, the projection is invertible.

    Args:
        B: (d x D) projection matrix (projects down).
        laplace_nsamp: Number of samples for posterior sampling of kernel
            hyperparameters.
        fit_restarts: Number of random restarts for MAP estimation.
    """

    def __init__(
        self, B: Tensor, laplace_nsamp: int = 25, fit_restarts: int = 10
    ) -> None:
        self.B = B
        self.Binv = torch.pinverse(B)
        self.laplace_nsamp = laplace_nsamp
        self.fit_restarts = fit_restarts
        super().__init__(
            refit_on_update=True,  # Important to not get stuck in local opt.
            refit_on_cv=False,
            warm_start_refitting=False,
            acqf_constructor=ei_or_nei,  # pyre-ignore
            acqf_optimizer=alebo_acqf_optimizer,
        )

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
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        assert len(task_features) == 0
        assert len(fidelity_features) == 0
        for b in bounds:
            assert b == (-1, 1)
        # GP is fit in the low-d space, so project Xs down.
        self.Xs = [(self.B @ X.t()).t() for X in Xs]
        self.Ys = Ys
        self.Yvars = Yvars
        self.device = self.B.device
        self.dtype = self.B.dtype
        self.model = self.get_and_fit_model(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        Xd = (self.B @ X.t()).t()  # Project down
        with gpytorch.settings.max_cholesky_size(2000):
            return super().predict(X=Xd)

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
        raise NotImplementedError

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
    ) -> Tuple[Tensor, Tensor, TGenMetadata, List[TCandidateMetadata]]:
        """Generate candidates.

        Candidates are generated in the linear embedding with the polytope
        constraints described in the paper.

        model_gen_options can contain 'raw_samples' (number of samples used for
        initializing the acquisition function optimization) and 'num_restarts'
        (number of restarts for acquisition function optimization).
        """
        for b in bounds:
            assert b == (-1, 1)
        # The following can be easily handled in the future when needed
        assert linear_constraints is None
        assert fixed_features is None
        assert pending_observations is None
        # Setup constraints
        A = torch.cat((self.Binv, -self.Binv))
        b = torch.ones(2 * self.Binv.shape[0], 1, dtype=self.dtype, device=self.device)
        linear_constraints = (A, b)
        # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
        noiseless = max(Yvar.min().item() for Yvar in self.Yvars) < 1e-5
        if model_gen_options is None:
            model_gen_options = {}
        model_gen_options = {
            "acquisition_function_kwargs": {"q": n, "noiseless": noiseless},
            "optimizer_kwargs": {
                "raw_samples": model_gen_options.get("raw_samples", 1000),
                "num_restarts": model_gen_options.get("num_restarts", 10),
                "B": self.B,
            },
        }
        Xd_opt, w, _gen_metadata, _candidate_metadata = super().gen(
            n=n,
            bounds=[(-1e8, 1e8)] * self.B.shape[0],
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            model_gen_options=model_gen_options,
        )
        # Project up
        Xopt = (self.Binv @ Xd_opt.t()).t()
        # Sometimes numerical tolerance can have Xopt epsilon outside [-1, 1],
        # so clip it back.
        if Xopt.min() < -1 or Xopt.max() > 1:
            logger.debug(f"Clipping from [{Xopt.min()}, {Xopt.max()}]")
            Xopt = torch.clamp(Xopt, min=-1.0, max=1.0)
        # pyre-fixme[7]: Expected `Tuple[Tensor, Tensor, Dict[str, typing.Any],
        #  List[Optional[Dict[str, typing.Any]]]]` but got `Tuple[typing.Any, Tensor,
        #  Dict[str, typing.Any], None]`.
        return Xopt, w, {}, None

    @copy_doc(TorchModel.update)
    def update(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        if self.model is None:
            raise RuntimeError(
                "Cannot update model that has not been fit"
            )  # pragma: no cover
        self.Xs = [(self.B @ X.t()).t() for X in Xs]  # Project down.
        self.Ys = Ys
        self.Yvars = Yvars
        if self.refit_on_update:
            state_dicts = None
        else:
            state_dicts = extract_map_statedict(
                m_b=self.model, num_outputs=len(Xs)  # pyre-ignore
            )
        self.model = self.get_and_fit_model(
            Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars, state_dicts=state_dicts
        )

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[Tensor],
        Ys_train: List[Tensor],
        Yvars_train: List[Tensor],
        X_test: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.model is None:
            raise RuntimeError(
                "Cannot cross-validate model that has not been fit"
            )  # pragma: no cover
        if self.refit_on_cv:
            state_dicts = None
        else:
            state_dicts = extract_map_statedict(
                m_b=self.model, num_outputs=len(self.Xs)  # pyre-ignore
            )
        Xs_train = [X @ self.B.t() for X in Xs_train]  # Project down.
        X_test = X_test @ self.B.t()
        model = self.get_and_fit_model(
            Xs=Xs_train, Ys=Ys_train, Yvars=Yvars_train, state_dicts=state_dicts
        )
        return self.model_predictor(model=model, X=X_test)  # pyre-ignore: [28]

    def get_and_fit_model(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        state_dicts: Optional[List[MutableMapping[str, Tensor]]] = None,
    ) -> GPyTorchModel:
        """Get a fitted ALEBO model for each outcome.

        Args:
            Xs: X for each outcome, already projected down.
            Ys: Y for each outcome.
            Yvars: Noise variance of Y for each outcome.
            state_dicts: State dicts to initialize model fitting.

        Returns: Fitted ALEBO model.
        """
        if state_dicts is None:
            state_dicts = [None] * len(Xs)
            fit_restarts = self.fit_restarts
        else:
            fit_restarts = 1  # Warm-started
        Yvars = [Yvar.clamp_min_(1e-7) for Yvar in Yvars]
        models = [
            get_fitted_model(
                B=self.B,
                train_X=X,
                train_Y=Ys[i],
                train_Yvar=Yvars[i],
                restarts=fit_restarts,
                nsamp=self.laplace_nsamp,
                # pyre-fixme[6]: Expected `Optional[Dict[str, Tensor]]` for 7th
                #  param but got `Optional[MutableMapping[str, Tensor]]`.
                init_state_dict=state_dicts[i],
            )
            for i, X in enumerate(Xs)
        ]
        if len(models) == 1:
            model = models[0]
        else:
            model = ModelListGP(*models)
        model.to(Xs[0])
        return model
