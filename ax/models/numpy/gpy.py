#!/usr/bin/env python3

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import GPy
import numpy as np
from ax.core.types import TConfig
from ax.models.model_utils import best_observed_point
from ax.models.numpy.gpy_nei import (
    compute_best_feasible_value,
    get_infeasible_cost,
    get_initial_points,
    omp_max_threads,
    optimize,
)
from ax.models.numpy_base import NumpyModel
from ax.utils.common.docutils import copy_doc
from ax.utils.stats.sobol import SobolEngine  # pyre-ignore
from scipy.stats import norm


def suppress_gpy_logs(func: Callable[..., Any]) -> Callable[..., Any]:
    """Suppress gpy log info to avoid notebook clutter."""

    @wraps(func)
    def _decorated(*args, **kwargs):
        gpy_logger = logging.getLogger("GP")
        prev_level = gpy_logger.level
        paramz_logger = logging.getLogger("paramz")
        prev_level_p = paramz_logger.level

        # Only log anything more critical than warning or error
        gpy_logger.setLevel(logging.WARN)
        paramz_logger.setLevel(logging.ERROR)
        output = func(*args, **kwargs)

        # revert to previous setting to avoid side effects
        gpy_logger.setLevel(prev_level)
        paramz_logger.setLevel(prev_level_p)
        return output

    return _decorated


class GPyGP(NumpyModel):
    """A single task GP via GPy.

    Fits a separate GP for each input. The kernel for that GP will depend on
    the number of task features:

    - 0 task features uses a Matern52 kernel
    - 1 task feature combines that with an ICM coregionalization kernel
    - 2 task features uses 2 coregionalization kernels, typically one of lower
      rank.

    See docstrings for _get_SGTP, _get_MTGP1, and _get_MTGP2 for more details.

    Args:
        map_fit_restarts: Number of random restarts for initial MAP fit.
        refit_on_cv: Re-fit hyperparameters during cross validation.
        primary_task_name: If 2 task features, specify which is primary. Not
            necessary for 0 or 1 tasks.
        primary_task_rank: Rank of primary task ICM kernel. Defaults to full
            rank.
        secondary_task_name: Name of secondary task feature.
        seconary_task_rank: Rank of secondary task ICM kernel. Defaults to 1.
    """

    def __init__(
        self,
        map_fit_restarts: int = 10,
        refit_on_cv: bool = False,
        refit_on_update: bool = True,
        primary_task_name: Optional[str] = None,
        primary_task_rank: Optional[int] = None,
        secondary_task_name: Optional[str] = None,
        secondary_task_rank: Optional[int] = None,
        use_multiprocessing: bool = True,
    ) -> None:
        self.map_fit_restarts = map_fit_restarts
        self.refit_on_cv = refit_on_cv
        self.refit_on_update = refit_on_update
        self.models: List[GPy.core.gp.GP] = []
        self.parameters: List[np.ndarray] = []
        self.task_features: List[int] = []
        self.Xs: List[np.ndarray] = []
        self.Ys: List[np.ndarray] = []
        self.Yvars: List[np.ndarray] = []
        self.primary_task_name = primary_task_name
        self.primary_task_rank = primary_task_rank
        self.secondary_task_name = secondary_task_name
        self.secondary_task_rank = secondary_task_rank
        self.use_multiprocessing = use_multiprocessing

    @copy_doc(NumpyModel.fit)
    def fit(
        self,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
    ) -> None:
        task_features = _validate_tasks(
            task_features=task_features,
            feature_names=feature_names,
            primary_task_name=self.primary_task_name,
            secondary_task_name=self.secondary_task_name,
        )

        # Construct the GPs
        fit_kwargs = [
            {
                "X": X,
                "Y": Ys[i],
                "Yvar": Yvars[i],
                "task_features": task_features,
                "map_fit_restarts": self.map_fit_restarts,
                "primary_task_rank": self.primary_task_rank,
                "secondary_task_rank": self.secondary_task_rank,
            }
            for i, X in enumerate(Xs)
        ]
        self.task_features = task_features
        if self.use_multiprocessing:
            with omp_max_threads(1):
                with ProcessPoolExecutor() as executor:
                    futures = [
                        # pyre-fixme[6]: Expected `Callable[..., _T]` for 1st param
                        #  but got `Callable[[ndarray, ndarray, ndarray, List[int],
                        #  int, Optional[ndarray], Optional[int], Optional[int]], GP]`.
                        executor.submit(_get_GP, **kwargs)
                        for kwargs in fit_kwargs
                    ]
                self.models = [future.result() for future in futures]
        else:
            # pyre-fixme[6]: Expected `ndarray` for 1st param but got `Optional[Union...
            self.models = [_get_GP(**kwargs) for kwargs in fit_kwargs]
        # Store MAP parameters
        self.parameters = [m.optimizer_array.copy() for m in self.models]
        # Store data
        self.Xs = Xs
        self.Ys = Ys
        self.Yvars = Yvars

    @copy_doc(NumpyModel.predict)
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _gp_predict(self.models, X)

    @suppress_gpy_logs
    @copy_doc(NumpyModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[np.ndarray],
        Ys_train: List[np.ndarray],
        Yvars_train: List[np.ndarray],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        parameters = [None] * len(self.models) if self.refit_on_cv else self.parameters
        cv_models = [
            _get_GP(
                X=Xs_train[i],
                Y=Ys_train[i],
                Yvar=Yvars_train[i],
                task_features=self.task_features,
                map_fit_restarts=self.map_fit_restarts,
                parameters=parameters[i],
                primary_task_rank=self.primary_task_rank,
                secondary_task_rank=self.secondary_task_rank,
            )
            for i, _ in enumerate(self.models)
        ]
        return _gp_predict(cv_models, X_test)

    @suppress_gpy_logs
    @copy_doc(NumpyModel.update)
    def update(
        self, Xs: List[np.ndarray], Ys: List[np.ndarray], Yvars: List[np.ndarray]
    ) -> None:
        for i, _ in enumerate(Xs):
            self.Xs[i] = np.vstack((self.Xs[i], Xs[i]))
            self.Ys[i] = np.vstack((self.Ys[i], Ys[i]))
            self.Yvars[i] = np.vstack((self.Yvars[i], Yvars[i]))
        parameters = (
            [None] * len(self.models) if self.refit_on_update else self.parameters
        )
        self.models = [
            _get_GP(
                X=self.Xs[i],
                Y=self.Ys[i],
                Yvar=self.Yvars[i],
                task_features=self.task_features,
                map_fit_restarts=self.map_fit_restarts,
                parameters=parameters[i],
                primary_task_rank=self.primary_task_rank,
                secondary_task_rank=self.secondary_task_rank,
            )
            for i, _ in enumerate(self.Xs)
        ]

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[np.ndarray]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate candidate points using noisy EI

        The following options can be set using model_gen_options:

        - nsamp: Number of fantasy samples (default 20)
        - qmc: Use QMC for fantasy samples (default True)
        - nopt: Number of random restarts for optimizing NEI (default 50)
        - init_samples: Number of samples in vectorized NEI evaluation
          used to select restart initializations (default 1e4)

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). Only one weight should be non-zero.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b. Each outcome constraint should operate on
                exactly one outcome.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature arrays X
                for m outcomes and k_i pending observations for outcome i.
            model_gen_options: A config dictionary with options described
                above.

        Returns:
            2-element tuple containing

            - (n x d) array of generated points.
            - n-array of weights for each point.
        """
        if model_gen_options is None:
            model_gen_options = {}
        nsamp: int = model_gen_options.get("nsamp", 20)
        qmc: bool = model_gen_options.get("qmc", True)
        nopt: int = model_gen_options.get("nopt", 50)
        init_samples: int = model_gen_options.get("init_samples", 10000)
        obj_idx, obj_sign, con_list = _parse_gen_inputs(
            objective_weights, outcome_constraints
        )
        if pending_observations is None:
            pending_observations = [np.array([[]]) for X in self.Xs]
        if fixed_features is None:
            fixed_features = {}
        # Get a value for the infeasible cost -M.
        M = get_infeasible_cost(
            obj_model=self.models[obj_idx], obj_sign=obj_sign, X=self.Xs[obj_idx]
        )
        # Generate!
        Xopt: List[np.ndarray] = []
        for _ in range(n):
            # Fantasize.
            fantasy_models, cand_X_array = self._get_fantasy_models(
                obj_idx=obj_idx,
                con_list=con_list,
                pending_observations=pending_observations,
                fixed_features=fixed_features,
                nsamp=nsamp,
                qmc=qmc,
            )
            # Compute best-feasible values
            f_best = compute_best_feasible_value(
                cand_X_array=cand_X_array,
                fantasy_models=fantasy_models,
                obj_idx=obj_idx,
                obj_sign=obj_sign,
                con_list=con_list,
            )
            # Get initial points
            x0s = get_initial_points(
                nopt=nopt,
                init_samples=init_samples,
                bounds=bounds,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                fantasy_models=fantasy_models,
                obj_idx=obj_idx,
                obj_sign=obj_sign,
                con_list=con_list,
                f_best=f_best,
                M=M,
            )
            # Optimize.
            xbest, _ = optimize(
                x0s=x0s,
                bounds=bounds,
                fixed_features=fixed_features,
                fantasy_models=fantasy_models,
                obj_idx=obj_idx,
                obj_sign=obj_sign,
                con_list=con_list,
                linear_constraints=linear_constraints,
                f_best=f_best,
                M=M,
                use_multiprocessing=self.use_multiprocessing,
            )

            if rounding_func is not None:
                xbest = rounding_func(xbest)

            Xopt.append(xbest.copy())
            pending_observations = _update_pending_observations(
                pending_observations=pending_observations, x=xbest
            )
            # Do it again!
        return np.array(Xopt), np.ones(n)

    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[np.ndarray]:
        """
        Identify the current best point, satisfying the constraints in the same
        format as to gen.

        Best point is chosen from among those that have already been observed.
        Strategy is described in the docstring from best_observed_point.

        Returns None if no best point can be identified.

        Args:
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: As in gen.
            outcome_constraints: As in gen.
            linear_constraints: As in gen.
            fixed_features: As in gen.
            model_gen_options: Config dictionary.

        Returns:
            A d-array of the best point.
        """
        return best_observed_point(
            model=self,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            options=model_gen_options,
        )

    def _get_fantasy_models(
        self,
        obj_idx: int,
        con_list: List[Tuple[int, float, float]],
        pending_observations: List[np.ndarray],
        fixed_features: Dict[int, float],
        nsamp: int,
        qmc: bool,
    ) -> Tuple[Dict[int, List[GPy.core.gp.GP]], np.ndarray]:
        """Generate fantasy models

        Also determines candidates for best-feasible point and returns those.

        Args:
            obj_idx: Index in models of the objective.
            con_list: List of (outcome_index, weight, ub) for each constraint.
            pending_observations: Pending observations as given to gen.
            fixed_features: Fixed features as given to gen.
            nsamp: Number of fantasy samples.
            qmc: Use QMC.

        Returns:
            2-element tuple containing

            - fantasy_models: Dictionary from outcome index to a list of fantasy
              models for that outcome.
            - cand_X_array: np.ndarray of potential best-feasible points.
        """
        # Data for fantasy models begins with the observations
        fantasy_Xs = [X.copy() for X in self.Xs]
        fantasy_Ys = [Y.copy() for Y in self.Ys]
        fantasy_Yvars = [Yvar.copy() for Yvar in self.Yvars]
        # We must fantasize pending observations, and candidates for incumbent best.
        # First identify pending observations. They are all sampled.
        X_to_sample: Set[Tuple[float]] = set()
        for i, X in enumerate(pending_observations):
            if X.size > 0:
                X_to_sample = X_to_sample.union({tuple(x) for x in X})
                fantasy_Xs[i] = np.vstack((fantasy_Xs[i], X))
                # These 0s will be filled in with fantasy samples
                fantasy_Ys[i] = np.vstack((fantasy_Ys[i], np.zeros((X.shape[0], 1))))
                fantasy_Yvars[i] = np.vstack(
                    (fantasy_Yvars[i], np.zeros((X.shape[0], 1)))
                )
        # Candidates for incumbent best are points that are observed or pending for
        # objective and all constraints, and also satisfy fixed_features.
        cand_X = {tuple(x) for x in fantasy_Xs[obj_idx]}
        for idx, _, _ in con_list:
            cand_X = cand_X.intersection({tuple(x) for x in fantasy_Xs[idx]})
        for k, v in fixed_features.items():
            cand_X = {x for x in cand_X if x[k] == v}
        # Include candidates for incumbent best as points to sample
        # pyre-fixme[6]: Expected `Iterable[Tuple[float]]` for 1st param but got
        #  `Set[Tuple[Any, ...]]`.
        X_to_sample = X_to_sample.union(cand_X)
        cand_X_array = np.array([list(x) for x in cand_X])
        if len(X_to_sample) == 0:
            # Nothing needs fantasizing
            fantasy_models = {idx: [self.models[idx]] for idx, _, _ in con_list}
            fantasy_models[obj_idx] = [self.models[obj_idx]]
            return fantasy_models, cand_X_array
        # Else, fantasize
        # Outcomes used in optimization must be fantasized
        fantasy_models: Dict[int, List[GPy.core.gp.GP]] = {obj_idx: []}
        for idx, _, _ in con_list:
            # pyre-fixme[35]: Target cannot be annotated.
            fantasy_models[idx]: List[GPy.core.gp.GP] = []
        # Args to _get_GP
        fit_kwargs = {
            "map_fit_restarts": self.map_fit_restarts,
            "task_features": self.task_features,
            "primary_task_rank": self.primary_task_rank,
            "secondary_task_rank": self.secondary_task_rank,
        }
        for idx in fantasy_models:
            # Identify points that should be fantasized for this outcome
            # pyre-fixme[6]: Expected `object` for 1st param but got `Tuple[_T_co,
            #  ...]`.
            to_sample = [tuple(x) in X_to_sample for x in fantasy_Xs[idx]]
            if sum(to_sample) == 0:
                # Nothing to fantasize for this outcome
                fantasy_models[idx] = [self.models[idx]] * nsamp
                continue
            Y_with_fantasy = np.tile(
                fantasy_Ys[idx], (1, nsamp)
            ).transpose()  # nsamp x n
            # Fantasize those points and fill in their values
            mu, cov = self.models[idx]._raw_predict(
                fantasy_Xs[idx][to_sample, :], full_cov=True
            )
            Y_with_fantasy[:, to_sample] = _mvn_sample(mu, cov, nsamp, qmc)
            fantasy_Yvars[idx][to_sample, 0] = 0.0  # Fantasies are noiseless.
            # Fit the models
            for j in range(nsamp):
                fantasy_models[idx].append(
                    _get_GP(
                        X=fantasy_Xs[idx],
                        Y=Y_with_fantasy[j, :, None],
                        Yvar=fantasy_Yvars[idx],
                        parameters=self.parameters[idx],
                        **fit_kwargs,
                    )
                )
        return fantasy_models, cand_X_array


def _gp_predict(
    models: List[GPy.core.gp.GP], X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    f = np.zeros((X.shape[0], len(models)))
    cov = np.zeros((X.shape[0], len(models), len(models)))
    for i, m in enumerate(models):
        f_i, var_i = m._raw_predict(X)
        f[:, i] = f_i[:, 0]
        cov[:, i, i] = var_i[:, 0]
    return f, cov


@suppress_gpy_logs
def _get_GP(
    X: np.ndarray,
    Y: np.ndarray,
    Yvar: np.ndarray,
    task_features: List[int],
    map_fit_restarts: int,
    parameters: Optional[np.ndarray] = None,
    primary_task_rank: Optional[int] = None,
    secondary_task_rank: Optional[int] = None,
) -> GPy.core.gp.GP:
    if len(task_features) == 0:
        m = _get_STGP(X, Y)
    elif len(task_features) == 1:
        m = _get_MTGP1(X, Y, task_features, primary_task_rank)
    elif len(task_features) == 2:
        m = _get_MTGP2(X, Y, task_features, primary_task_rank, secondary_task_rank)
    else:
        raise ValueError("More than 2 task features not supported.")
    # Set observation noise
    m.likelihood["het_Gauss.variance"].unconstrain()
    m.likelihood["het_Gauss.variance"].fix()
    m.het_Gauss.variance = Yvar
    if parameters is None:
        # Do MAP fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            m.optimize_restarts(map_fit_restarts, verbose=False)
    else:
        m.optimizer_array = parameters.copy()
    return m


def _get_STGP(X: np.ndarray, Y: np.ndarray) -> GPy.core.gp.GP:
    """Construct a single task GP with 0 mean and Matern52 kernel."""
    # Single-task
    kernel = GPy.kern.Matern52(X.shape[1], ARD=True)  # pyre-ignore
    m = GPy.models.GPHeteroscedasticRegression(X, Y, kernel)  # pyre-ignore
    # Set priors
    m.kern.lengthscale.set_prior(GPy.priors.Gamma(2.0, 1.0 / 0.2), warning=False)
    m.kern.variance.set_prior(GPy.priors.Gamma(1.1, 1.0 / 20.0), warning=False)
    return m


def _get_MTGP1(
    X: np.ndarray,
    Y: np.ndarray,
    task_features: List[int],
    primary_task_rank: Optional[int],
) -> GPy.core.gp.GP:
    """Construct a multi-task GP with ICM kernel.

    This GP applies an ICM coregionalization kernel to a discrete parameter z.
    The overall kernel is

    K(x, z, x', z') = B[z, z'] k(x, x')

    where B is a covariance matrix over levels of z. In particular,

    B = W*W^T + diag(kappa).

    We allow W to be full rank, unless primary_task_rank specifies otherwise.
    There are two sources of unidentifiability. The  first is the symmetry
    p(W | data) = p(-W | data), which makes the posterior  mean not a useful
    estimate. We resolve this by constraining W > 0, which is  the same thing
    done in "Multi-task Bayesian optimization" (NIPS 2013). This  limits the
    model to learning positive correlations. The second source of
    unidentifiability is between W and the variance term of k, which for a
    Matern or RBF kernel is a multiplier like k(x, x') = sigma_k k2(x, x').
    This means p(aW, sigma_k / a | data) = p(W, sigma_k | data) for any a. This
    is resolved by fixing sigma_k = 1, and leaving all of the kernel variance
    inference in W.
    """
    cont_idx = [i for i in range(X.shape[1]) if i not in task_features]
    # Construct the continuous kernel
    kernel_cont = GPy.kern.Matern52(  # pyre-ignore
        input_dim=len(cont_idx), active_dims=cont_idx, ARD=True, name="Mat52"
    )
    # Construct the coregionaliation kernel
    num_tasks = int(X[:, task_features[0]].max()) + 1
    if primary_task_rank is None:
        primary_task_rank = num_tasks
    kernel_discrete = GPy.kern.Coregionalize(  # pyre-ignore
        input_dim=1,
        output_dim=num_tasks,
        rank=primary_task_rank,
        active_dims=task_features,
        name="B",
    )
    kernel = kernel_cont.prod(kernel_discrete, name="ICM")
    m = GPy.models.GPHeteroscedasticRegression(X, Y, kernel)  # pyre-ignore
    # Set priors
    m.ICM.Mat52.lengthscale.set_prior(GPy.priors.Gamma(2.0, 1.0 / 0.2), warning=False)
    m.ICM.Mat52.variance.fix(1.0)
    m.ICM.B.W.constrain_positive(warning=False)
    m.ICM.B.W.set_prior(GPy.priors.Gamma(1.0, 1.0 / 10.0), warning=False)
    m.ICM.B.kappa.set_prior(GPy.priors.Gamma(1.1, 1.0 / 20.0), warning=False)
    return m


def _get_MTGP2(
    X: np.ndarray,
    Y: np.ndarray,
    task_features: List[int],
    primary_task_rank: Optional[int],
    secondary_task_rank: Optional[int],
) -> GPy.core.gp.GP:
    """
    This GP applies an ICM coregionalization kernel to a discrete parameter z,
    and a rank-1 ICM kernel to a second discrete parameter a. The kernel is

    K((x, z, a), (x', z', a')) = B[z, z'] Q[a, a'] k(x, x')

    where B is a covariance matrix over levels of z - see the docstring for
    _get_MTGP1 for details about it. Q is rank 1:

    Q = a*a^T + diag(kappa)

    kappa is just for numerical stability, so we fix kappa = 1e-6.

    We then add in a Bias kernel on a, which corresponds to fitting a constant
    mean function for each a.

    Full-rank coregionalization (B) is applied to the first item in
    task_features, and rank-1 coregionalization (Q) is applied to the second.

    Ranks can be specified to something other than full or 1 with
    primary_task_rank and secondary_task_rank.
    """
    cont_idx = [i for i in range(X.shape[1]) if i not in task_features]
    # Construct the continuous kernel
    kernel_cont = GPy.kern.Matern52(  # pyre-ignore
        input_dim=len(cont_idx), active_dims=cont_idx, ARD=True, name="Mat52"
    )
    # Construct the primary coregionaliation kernel
    num_tasks = int(X[:, task_features[0]].max()) + 1
    if primary_task_rank is None:
        primary_task_rank = num_tasks
    kernel_discrete = GPy.kern.Coregionalize(  # pyre-ignore
        input_dim=1,
        output_dim=num_tasks,
        rank=primary_task_rank,
        active_dims=[task_features[0]],
        name="B",
    )
    kernel = kernel_cont.prod(kernel_discrete, name="ICM")
    # Construct the secondary coregionaliation kernel
    num_tasks = int(X[:, task_features[1]].max()) + 1
    if secondary_task_rank is None:
        secondary_task_rank = 1
    kernel_discrete2 = GPy.kern.Coregionalize(  # pyre-ignore
        input_dim=1,
        output_dim=num_tasks,
        rank=secondary_task_rank,
        active_dims=[task_features[1]],
        name="Q",
    )
    kernel = kernel.prod(kernel_discrete2, name="ICM")
    # Add in a bias kernel, for the offset
    kernel_bias = GPy.kern.Coregionalize(  # pyre-ignore
        input_dim=1,
        output_dim=num_tasks,
        rank=1,
        active_dims=[task_features[1]],
        name="QB",
    )
    kernel_bias = kernel_bias.prod(GPy.kern.Bias(1), name="bias")  # pyre-ignore
    kernel = kernel.add(kernel_bias, name="add")
    m = GPy.models.GPHeteroscedasticRegression(X, Y, kernel)  # pyre-ignore
    # Set priors
    # Mat52 kernel
    m.add.ICM.Mat52.lengthscale.set_prior(
        GPy.priors.Gamma(2.0, 1.0 / 0.2), warning=False
    )
    m.add.ICM.Mat52.variance.fix(1.0)
    # Covariance over task
    m.add.ICM.B.W.constrain_positive(warning=False)
    m.add.ICM.B.W.set_prior(GPy.priors.Gamma(1.0, 1.0 / 10.0), warning=False)
    m.add.ICM.B.kappa.set_prior(GPy.priors.Gamma(1.1, 1.0 / 20.0), warning=False)
    # Covariance over affine parameter
    m.add.ICM.Q.W.constrain_positive(warning=False)
    m.add.ICM.Q.W.set_prior(
        GPy.priors.Gamma(6.0, 1.0 / 0.2), warning=False  # Prior around 1.
    )
    m.add.ICM.Q.kappa.fix(1e-6)
    # Bias kernel
    # We get the magnitude of the offset from kappa
    m.add.bias.bias.variance.fix(1)
    m.add.bias.QB.W.fix(0)  # No covariance in offsets
    m.add.bias.QB.kappa.set_prior(GPy.priors.Gamma(1.1, 1.0 / 20.0), warning=False)
    return m


def _validate_tasks(
    task_features: List[int],
    feature_names: List[str],
    primary_task_name: Optional[str],
    secondary_task_name: Optional[str],
) -> List[int]:
    if len(task_features) == 0 and (
        primary_task_name is not None or secondary_task_name is not None
    ):
        raise ValueError("Task names specified, but no task features.")
    elif len(task_features) == 1:
        if secondary_task_name is not None:
            raise ValueError("Secondary task name specified, but only 1 task feature")
        if (
            primary_task_name is not None
            and primary_task_name != feature_names[task_features[0]]
        ):
            raise ValueError(
                f"Primary task name specified as {primary_task_name}, "
                "but task feature is {feature_names[task_features[0]]}"
            )
    elif len(task_features) == 2:
        if primary_task_name is None or secondary_task_name is None:
            raise ValueError("Names of primary and secondary tasks must be specified.")
        primary_idx = feature_names.index(primary_task_name)
        secondary_idx = feature_names.index(secondary_task_name)
        if {primary_idx, secondary_idx} != set(task_features):
            feat_names = [feature_names[t] for t in task_features]
            spec_names = [primary_task_name, secondary_task_name]
            raise ValueError(f"Task features are {feat_names}, not f{spec_names}")
        # Sort features
        task_features = [primary_idx, secondary_idx]
    elif len(task_features) > 2:
        raise ValueError("Model supports at most 2 tasks")
    return task_features


def _parse_gen_inputs(
    objective_weights: np.ndarray,
    outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, float, List[Tuple[int, float, float]]]:
    """Convert gen inputs to the formats we'll want for optimizing NEI.

    Args:
        objective_weights: Weights on each outcome for the objective.
        outcome_constraints: Tuple of linear outcome constraints.

    Returns:
        3-element tuple containing

        - The index of the objective outcome.
        - Sign of the weight on the objective.
        - List of (outcome_index, weight, ub) for each constraint.
    """
    Z = np.nonzero(objective_weights)[0]
    if len(Z) != 1:
        raise ValueError(
            f"Require a single objective outcome. Got {objective_weights}."
        )
    obj_idx = Z[0]
    obj_sign = float(np.sign(objective_weights[obj_idx]))

    # Parse outcome constraints
    if outcome_constraints is not None:
        A, b = outcome_constraints
        Z = np.nonzero(A)
        if not np.array_equal(Z[0], np.arange(len(A))):
            raise ValueError(
                f"Each constraint must operate on exactly one outcome. Got {A}."
            )
        con_idx = Z[1].tolist()
        ubs = b[:, 0].tolist()
        ws = A[Z].tolist()
        con_list = list(zip(con_idx, ws, ubs))
    else:
        con_list = []
    return obj_idx, obj_sign, con_list


def _update_pending_observations(
    pending_observations: List[np.ndarray], x: np.ndarray
) -> List[np.ndarray]:
    """Update pending observations to include x.

    Assumes all outcomes will be observed.

    Args:
        pending_observations: Pending observations as given to gen.
        x: array of point whose observation is pending.

    Returns:
        Pending observations including x.
    """
    for i, X in enumerate(pending_observations):
        if X.size > 0:
            pending_observations[i] = np.vstack((X, x))
        else:
            pending_observations[i] = np.array([x])
    return pending_observations


def _mvn_sample(mu: np.ndarray, cov: np.ndarray, nsamp: int, qmc: bool) -> np.ndarray:
    """
    Sample from an MVN, with or without QMC.

    QMC should only be used when the samples are being used to compute an
    expectation.

    Args:
        mu: d x 1 array of means.
        cov: d x d covariance matrix.
        nsamp: Number of samples to draw.
        qmc: Use QMC or not.

    Returns:
        (nsamp x d) array of MVN samples.
    """
    if qmc:
        try:
            g = SobolEngine(mu.shape[0], scramble=True)  # pyre-ignore
            u = g.draw(n=nsamp)
            r = norm.ppf(u)  # pyre-ignore
            A = np.linalg.cholesky(cov)
            Ys = (np.dot(A, r.transpose()) + mu).transpose()
            return Ys
        except np.linalg.linalg.LinAlgError:
            pass
    Ys = np.random.multivariate_normal(mu[:, 0], cov, nsamp)
    return Ys
