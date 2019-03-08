#!/usr/bin/env python3

import contextlib
import ctypes
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Generator as GeneratorType, List, Optional, Tuple

import GPy
import numpy as np
from ae.lazarus.ae.models.model_utils import add_fixed_features
from ae.lazarus.ae.utils.stats.sobol import SobolEngine  # pyre-ignore
from scipy.optimize import minimize
from scipy.stats import norm


def get_infeasible_cost(
    obj_model: GPy.core.gp.GP, obj_sign: float, X: np.ndarray
) -> float:
    """Get a suitable value for the penalty for having no feasible point.

    Args:
        obj_model: The model for the objective
        obj_sign: The sign of the objective weights
        X: The data for the objective

    Returns: The no-feasible penalty M
    """
    # Make predictions on the observations
    f, var = obj_model._raw_predict(X)
    f = f * obj_sign
    # We solve maximization problems, and so want -M < min(f(X))
    lb = (f - 6 * np.sqrt(var)).min()
    #  leave M at 0 if f(X) is positive.
    M = max(0.0, -lb)
    return M


def compute_best_feasible_value(
    cand_X_array: np.ndarray,
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
) -> np.ndarray:
    """For each fantasy, get the value of the best feasible point.

    Args:
        cand_X_array: X values that are options for best feasible point.
        fantasy_models: nsamp fantasy models for each outcome.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of weight on objective.
        con_list: List of (index, weight, upper bound) for outcome constraints.

    Returns: (nsamp,) array of best feasible objective value.
    """
    nsamp = len(fantasy_models[obj_idx])
    if cand_X_array.size == 0:
        # No feasible observations, best-feasible value is -Inf.
        return -np.Inf * np.ones(nsamp)
    # Get objective values
    Yobj = np.zeros((cand_X_array.shape[0], nsamp))
    for i in range(nsamp):
        Yobj[:, i] = (
            obj_sign * fantasy_models[obj_idx][i]._raw_predict(cand_X_array)[0][:, 0]
        )
    # Check constraints
    for idx, w, ub in con_list:
        Ycon = np.zeros((cand_X_array.shape[0], nsamp))
        for i in range(nsamp):
            Ycon[:, i] = fantasy_models[idx][i]._raw_predict(cand_X_array)[0][:, 0]
        infeas = w * Ycon > ub
        Yobj[infeas] = -np.Inf
    # Max over observations for each fantasy
    f_best = np.max(Yobj, axis=0)  # (nsamp, )
    return f_best


def get_initial_points(
    nopt: int,
    init_samples: int,
    bounds: List[Tuple[float, float]],
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    fixed_features: Dict[int, float],
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    f_best: np.ndarray,
    M: float,
) -> np.ndarray:
    """Get initial points for NEI maximization.

    Does a vectorized evaluation of NEI on a large quasirandom sequence to
    select good initial points for optimization. The (parameter-constraint-
    feasible) point with max NEI is included, along with a random sampling of
    other feasible points with positive NEI, to get a collection of nopt points.

    If less than nopt points are feasible, an exception is raised. If less than
    nopt points have positive NEI, then feasible points with 0 NEI will be
    included to reach nopt initial points.

    Args:
        nopt: Number of initial points to select.
        init_samples: Size of the vectorized NEI evaluation from which initial
            points are selected.
        bounds: Parameter box bounds.
        linear_constraints: (A, b) linear constraints.
        fixed_features: Features that should be fixed at a particular value.
        fantasy_models: Dictionary from outcome index to fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weight, and upper bound for each outcome constraint.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.

    Returns: (nopt x d) array of optimization initial points.
    """
    g = SobolEngine(len(bounds), scramble=True)  # pyre-ignore
    z = g.draw(n=init_samples)
    # Scale z to the bounds
    lb, ub = zip(*bounds)
    lb = np.array(lb)
    ub = np.array(ub)
    for k, v in fixed_features.items():
        lb[k] = v
        ub[k] = v
    z = z * (ub - lb) + lb
    # Filter to points that satisfy parameter constraints.
    if linear_constraints is not None:
        A, b = linear_constraints
        feas = (A @ z.transpose() <= b).all(axis=0)
        z = z[feas, :]
        if z.shape[0] < nopt:
            pcnt_feas = 100 * z.shape[0] / init_samples
            raise Exception(
                f"Only {pcnt_feas}% of space satisfies parameter constraints."
            )
    # Compute NEI
    vals = nei_vectorized(
        X=z,
        fantasy_models=fantasy_models,
        obj_idx=obj_idx,
        obj_sign=obj_sign,
        con_list=con_list,
        f_best=f_best,
        M=M,
    )
    best_idx = np.argmax(vals)
    to_use = {best_idx}
    # Threshold for NEI to be considered positive
    threshold = vals[best_idx] * 1e-6
    positive_idx = np.arange(len(vals))[vals > threshold]
    np.random.shuffle(positive_idx)
    # Build up to a size of nopt
    s = nopt if best_idx in positive_idx[:nopt] else nopt - 1
    to_use = to_use.union(positive_idx[:s])
    # if len(to_use) < nopt, it is because there were not enough positive points.
    n = nopt - len(to_use)
    if n > 0:
        others_idx = np.arange(len(vals))[vals <= threshold]
        np.random.shuffle(others_idx)
        to_use = to_use.union(others_idx[:n])
    return z[list(to_use), :]


def nei_vectorized(
    X: np.ndarray,
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    f_best: np.ndarray,
    M: float,
) -> np.ndarray:
    """Compute NEI at an array of points

    Args:
        X: n Points at which to evaluate NEI.
        fantasy_models: Dictionary of fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weights, and upper bounds for outcome constraints.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.

    Returns: (n,) array of NEI at X.
    """
    eis = np.zeros(X.shape[0])
    for i, m in enumerate(fantasy_models[obj_idx]):
        # For each fantasy,
        f, f_var = m._raw_predict(
            X, full_cov=False
        )  # type: Tuple[np.ndarray, np.ndarray]
        if f_best[i] == -np.Inf:
            # All observations infeasible in this fantasy.
            # Use formula for all points infeasible.
            eis_i: np.ndarray = obj_sign * f + M
        else:
            eis_i = ei_vectorized(obj_sign * f, f_var, f_best[i])
        for idx, w, ub in con_list:
            g, g_var = fantasy_models[idx][i]._raw_predict(X, full_cov=False)
            p_feas = norm.cdf((ub - g * w) / np.sqrt(g_var * w ** 2))  # pyre-ignore
            eis_i *= p_feas
        eis += eis_i[:, 0]
    return eis / len(fantasy_models[obj_idx])


def nei_and_grad(
    x: np.ndarray,
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    f_best: np.ndarray,
    M: float,
) -> Tuple[float, np.ndarray]:
    """Compute NEI and its gradient at a point.

    Args:
        x: Point at which to evaluate NEI.
        fantasy_models: Dictionary of fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weights, and upper bounds for outcome constraints.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.

    Returns: NEI at x and its gradient.
    """
    nsamp = len(fantasy_models[obj_idx])
    nei = 0.0
    dnei = np.zeros(len(x))
    for i, m in enumerate(fantasy_models[obj_idx]):
        # For each fantasy,
        # First EI
        f, f_var, df, df_var = _f_and_grad(m, x, obj_sign)
        if f_best[i] == -np.Inf:
            # All observations infeasible in this fantasy.
            # Use formula for all points infeasible.
            ei_i = f + M
            dei_i = df
        else:
            ei_i, dei_i = ei_and_grad(f, f_var, f_best[i], df, df_var)
        # And now feasibility
        if len(con_list) > 0:
            dp_feas = np.zeros((len(con_list), len(x)))
            p_feas = np.ones((len(con_list)))
            for j, (idx, w, ub) in enumerate(con_list):
                g, g_var, dg, dg_var = _f_and_grad(fantasy_models[idx][i], x, w)
                g_sd = np.sqrt(g_var)
                p_feas[j] = norm.cdf((ub - g) / g_sd)  # pyre-ignore
                dg_sd = dg_var / (2 * g_sd)
                dp_feas[j, :] = -norm.pdf((ub - g) / g_sd) * (  # pyre-ignore
                    (dg * g_sd + dg_sd * (ub - g)) / g_var
                )
            p_all_feas = p_feas.prod()
            dp_all_feas = np.zeros(len(x))
            # Apply product rule over constraints
            for j in range(len(con_list)):
                p_feas_excl_j = p_feas[np.arange(len(p_feas)) != j].prod()
                dp_all_feas += dp_feas[j, :] * p_feas_excl_j
            # And product rule over EI and feasibility
            nei_i = ei_i * p_all_feas
            dnei_i = dei_i * p_all_feas + ei_i * dp_all_feas
        else:
            nei_i = ei_i
            dnei_i = dei_i
        nei += nei_i
        dnei += dnei_i
    return nei / nsamp, dnei / nsamp


def ei_and_grad(
    f: float, f_var: float, f_best: float, df: np.ndarray, df_var: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    EI and its gradient.

    Args:
        f: Model predicted mean.
        f_var: Model predicted variances.
        f_best: Current feasible best.
        df: Gradient of mean, in n dimensional space.
        dvar: Gradient of variances.

    Returns: ei and its gradient.
    """
    sd = np.sqrt(f_var)
    sd = max(sd, 1e-9)  # Avoid dividing by zero
    u = (f - f_best) / sd
    dsd = df_var / (2 * sd)
    ei = sd * (u * norm.cdf(u) + norm.pdf(u))  # pyre-ignore
    dei = dsd * norm.pdf(u) + df * norm.cdf(u)  # pyre-ignore
    return ei, dei


def ei_vectorized(f: np.ndarray, f_var: np.ndarray, f_best: float) -> np.ndarray:
    """
    Compute EI.

    Args:
        f: Model predicted means at x, nsamp fantasies.
        f_var: Model predicted variances.
        f_best: Current feasible best.

    Returns:
        ei: The expected improvement at each point.
    """
    sd = np.sqrt(f_var)
    indx = sd < 1e-9
    sd[indx] = 1e-9  # Avoid dividing by zero
    u = (f - f_best) / sd
    ei = sd * (u * norm.cdf(u) + norm.pdf(u))  # pyre-ignore
    return ei


def _f_and_grad(
    m: GPy.core.gp.GP, x: np.ndarray, w: float
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the GP and its gradients at x.

    Args:
        m: A GP.
        x: Point at which to evaluate.
        w: Evaluation is weighted by this factor.

    Returns:
       f: Mean prediction.
       f_var: Variance prediction.
       df: Gradient of f.
       df_var: Gradient of f_var.
    """
    f_arr, f_var_arr = m._raw_predict(np.array([x]), full_cov=False)
    f = w * f_arr[0, 0]
    f_var = w ** 2 * f_var_arr[0, 0]
    df_arr, df_var_arr = m.predictive_gradients(
        np.array([x])
    )  # type: Tuple[np.ndarray, np.ndarray]
    df: np.ndarray = w * df_arr[0, :, 0]  # pyre-ignore
    df_var: np.ndarray = w ** 2 * df_var_arr[0, :]  # pyre-ignore
    return f, f_var, df, df_var


def optimize(
    x0s: np.ndarray,
    bounds: List[Tuple[float, float]],
    fixed_features: Dict[int, float],
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    f_best: np.ndarray,
    M: float,
    use_multiprocessing: bool,
) -> Tuple[np.ndarray, float]:
    """Optimize NEI with restarts at x0s.

    Args:
        x0s: Initial arms for optimization restarts.
        bounds: Bounds on parameters.
        fixed_features: Features that must remain fixed at a particular value.
        fantasy_models: Dictionary of fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weights, and upper bounds for outcome constraints.
        linear_constraints: Parameter constraints.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.
        use_multiprocessing: Use multiprocessing.

    Returns: x that optimizes NEI, and NEI at that x.
    """
    opt_kwargs = [
        {
            "x0": x0,
            "bounds": bounds,
            "fixed_features": fixed_features,
            "fantasy_models": fantasy_models,
            "obj_idx": obj_idx,
            "obj_sign": obj_sign,
            "con_list": con_list,
            "linear_constraints": linear_constraints,
            "f_best": f_best,
            "M": M,
        }
        for x0 in x0s
    ]
    if use_multiprocessing:
        with omp_max_threads(1):
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(optimize_from_x0, **kwargs) for kwargs in opt_kwargs
                ]
            res = [future.result() for future in futures]
    else:
        res = [optimize_from_x0(**kwargs) for kwargs in opt_kwargs]
    # Find the best result
    xopt = None
    fmin = None
    for x, fun in res:
        if fmin is None or fun < fmin:
            fmin = fun
            xopt = x
    if fmin == np.Inf or fmin is None:
        raise Exception("All optimizations failed.")  # pragma: no cover
    # pyre: Expected `typing.Tuple[np.ndarray, float]` but got `typing.
    # pyre-fixme[7]: Tuple[Optional[np.ndarray], float]`.
    return xopt, -fmin


def optimize_from_x0(
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    fixed_features: Dict[int, float],
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    f_best: np.ndarray,
    M: float,
) -> Tuple[np.ndarray, float]:
    """Optimize NEI with restarts at x0s.

    Args:
        x0s: Initial arms for optimization restarts.
        bounds: Bounds on parameters.
        fixed_features: Features that are fixed at a value.
        fantasy_models: Dictionary of fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weights, and upper bounds for outcome constraints.
        linear_constraints: Parameter constraints.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.

    Returns: x that optimizes NEI, and NEI at that x.
    """
    # Restrict optimization problem to tunable features
    tunable_slice = [i for i, _ in enumerate(bounds) if i not in fixed_features]
    bounds_opt = [bounds[i] for i in tunable_slice]
    x0_opt = x0[tunable_slice]

    args = (
        tunable_slice,
        fixed_features,
        fantasy_models,
        obj_idx,
        obj_sign,
        con_list,
        f_best,
        M,
    )
    if len(x0_opt) == 0:
        # Nothing to optimize.
        xopt = np.array([fixed_features[i] for i, _ in enumerate(bounds)])
        # pyre-fixme[7]: Expected `Tuple[ndarray, float]` but got `Tuple[ndarray, Uni...
        return xopt, objective_and_grad(x0_opt, *args)[0]

    # Prepare linear constraints
    if linear_constraints is not None:
        A, b = linear_constraints
        # Constraints will be evaluated only on the tunable features
        A_opt = A[:, tunable_slice]
        Ax_fix = np.zeros((A.shape[0], 1))
        for j in range(A.shape[0]):
            Ax_fix[j, 0] = np.sum([A[j, k] * v for k, v in fixed_features.items()])

        def eval_linear_constraints(x: np.ndarray) -> np.ndarray:
            return (b - Ax_fix - A_opt @ x[:, None]).flatten()

        def deval_linear_constraints(x: np.ndarray) -> np.ndarray:
            return -A_opt

        constraints = {
            "type": "ineq",
            "fun": eval_linear_constraints,
            "jac": deval_linear_constraints,
        }
    else:
        constraints = ()

    try:
        res = minimize(
            fun=objective_and_grad,
            x0=x0_opt,
            args=args,
            method="SLSQP",
            jac=True,
            bounds=bounds_opt,
            callback=nan_cb,
            constraints=constraints,
        )
        # Add fixed features back in
        x = add_fixed_features(
            tunable_points=np.array([res.x]),
            d=len(x0),
            fixed_features=fixed_features,
            tunable_feature_indices=np.array(tunable_slice),
        )[0]
        return x, res.fun
    except StopIteration:
        return x0, np.Inf


def objective_and_grad(
    x: np.ndarray,
    tunable_slice: List[int],
    fixed_features: Dict[int, float],
    fantasy_models: Dict[int, List[GPy.core.gp.GP]],
    obj_idx: int,
    obj_sign: float,
    con_list: List[Tuple[int, float, float]],
    f_best: np.ndarray,
    M: float,
) -> Tuple[float, np.ndarray]:
    """Objective function to maximize NEI.

    Returns negative NEI, so that minimizing this function maximizes NEI.

    x is restricted to only tunable features, and the fixed features from
    fixed_features are added to compute NEI.

    Args:
        x: Point at which to evaluate NEI, tunable features only.
        tunable_slice: Which features are tunable.
        fixed_features: Values for fixed features.
        fantasy_models: Dictionary of fantasy models.
        obj_idx: Outcome index of objective.
        obj_sign: Sign of the objective weight.
        con_list: Index, weights, and upper bounds for outcome constraints.
        f_best: Incumbent best value for each fantasy.
        M: Penalty for no feasible point.

    Returns: NEI at x and its gradient.
    """
    x_full = np.zeros(len(tunable_slice) + len(fixed_features))
    x_full[tunable_slice] = x
    for k, v in fixed_features.items():
        x_full[k] = v
    ei, dei = nei_and_grad(
        x=x_full,
        fantasy_models=fantasy_models,
        obj_idx=obj_idx,
        obj_sign=obj_sign,
        con_list=con_list,
        f_best=f_best,
        M=M,
    )
    return -ei, -dei[tunable_slice]


def nan_cb(x: np.ndarray) -> None:
    """Optimizer callback to check for nans.

    If the optimizer gets stuck in an infeasible region, the subproblem may
    return NaN. If this happens, terminate this run.

    Args:
        x: Current iteration.
    """
    if np.isnan(sum(x)):
        raise StopIteration


@contextlib.contextmanager
def omp_max_threads(n: int) -> GeneratorType:
    """Temporarily changes the number of OpenMP threads. """
    # This works if Numpy is built against OpenMP
    try:
        omplib = ctypes.cdll.LoadLibrary("libgomp.so.1")
        old_n = omplib.omp_get_max_threads()
        omplib.omp_set_num_threads(n)
        try:
            yield
        finally:
            omplib.omp_set_num_threads(old_n)
    except Exception:
        # Do nothing; this is not a breaking issue.
        yield
