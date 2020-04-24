#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import FunctionType
from typing import Dict, List, Optional, Tuple, Union

from ax.core.base import Base
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast
from ax.utils.measurement.synthetic_functions import SyntheticFunction


logger = get_logger(__name__)


NONE_DOMAIN_ERROR = """
When creating a `BenchmarkProblem` with custom function, one of the `domain` or
`search_space` argumetns is required to be non-null.
"""

NONE_SYNTHETIC_FUNCTION_DOMAIN_ERROR = """
When creating a `BenchmarkProblem` with a `SyntheticFunction`, a non-null `domain`
argument or a non-null `domain` property on the `SyntheticFunction` is required.
"""

ANONYMOUS_FUNCTION_ERROR = """
If using anonymous function, please pass non-null `name` argument.
"""

ADHOC_FUNCTION_NOISE_SET_ERROR = """
Cannot set `noise_sd` setting for problems that use anonymous functions, as their
inherent noise level is unknown. Add a known synthetic function to `ax.utils.
measurement.synthetic_functions` to be able to add noise to the benchmark problem.
"""


class BenchmarkProblem(Base):
    """Benchmark problem, represented in terms of Ax search space and optimization
    config. Useful to represent complex problems that involve constaints, non-
    range parameters, etc.

    Note: if this problem is computationally intensive, consider setting
    `evaluate_suggested` argument to False.

    Args:
        search_space: Problem domain.
        optimization_config: Problem objective and constraints. Note that by
            default, an `Objective` in the `OptimizationConfig` has `minimize`
            set to False, so by default an `OptimizationConfig` is that of
            maximization.
        name: Optional name of the problem, will default to the name of the
            objective metric (e.g., "Branin" or "Branin_constrainted" if
            constraints are present). The name of the problem is reflected in the
            names of the benchmarking experiments (e.g. "Sobol_on_Branin").
        optimal_value: Optional target objective value for the optimization.
        evaluate_suggested: Whether the model-predicted best value should be
            evaluated when benchmarking on this problem. Note that in practice,
            this means that for every model-generated trial, an extra point will
            be evaluated. This extra point is often different from the model-
            generated trials, since those trials aim to both explore and exploit,
            so the aim is not usually to suggest the current model-predicted
            optimum.
    """

    name: str
    search_space: SearchSpace
    optimization_config: OptimizationConfig
    optimal_value: Optional[float]
    # Whether to evaluate model-predicted best values at each iteration to
    # compare to the model predictions. Should only be `False` if the problem
    # is expensive to evaluate and therefore no extra evaluations beyond the one
    # optimization loop should be performed.
    evaluate_suggested: bool

    def __init__(
        self,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        name: Optional[str] = None,
        optimal_value: Optional[float] = None,
        evaluate_suggested: bool = True,
    ) -> None:
        self.search_space = search_space
        self.optimization_config = optimization_config
        suffix = (  # To avoid clashing of two problem names, mark constrained
            "_constrained" if len(optimization_config.outcome_constraints) > 0 else ""
        )
        self.name = name or optimization_config.objective.metric.name + suffix
        self.optimal_value = optimal_value
        self.evaluate_suggested = evaluate_suggested


class SimpleBenchmarkProblem(BenchmarkProblem):
    """Benchmark problem, represented in terms of simplified constructions: a
    callable function, a domain that consists or ranges, etc. This problem does
    not support parameter or outcome constraints.

    Note: if this problem is computationally intensive, consider setting
    `evaluate_suggested` argument to False.

    Args:
        f: Ax `SyntheticFunction` or an ad-hoc callable that evaluates points
            represented as nd-arrays. Input to the callable should be an (n x d)
            array, where n is the number of points to evaluate, and d is the
            dimensionality of the points. Returns a float or an (1 x n) array.
            Used as problem objective.
        name: Optional name of the problem, will default to the name of the
            objective metric (e.g., "Branin" or "Branin_constrainted" if
            constraints are present). The name of the problem is reflected in the
            names of the benchmarking experiments (e.g. "Sobol_on_Branin").
        domain: Problem domain as list of tuples. Parameter names will be derived
            from the length of this list, as {"x1", ..., "xN"}, where N is the
            length of this list.
        optimal_value: Optional target objective value for the optimization.
        minimize: Whether this is a minimization problem, defatuls to False.
        noise_sd: Measure of the noise that will be added to the observations
            during the optimization. During the evaluation phase, true values
            will be extracted to measure a method's performance. Only applicable
            when using a known `SyntetheticFunction` as the `f` argument.
        evaluate_suggested: Whether the model-predicted best value should be
            evaluated when benchmarking on this problem. Note that in practice,
            this means that for every model-generated trial, an extra point will
            be evaluated. This extra point is often different from the model-
            generated trials, since those trials aim to both explore and exploit,
            so the aim is not usually to suggest the current model-predicted
            optimum.
    """

    f: Union[SyntheticFunction, FunctionType]
    name: str
    domain: List[Tuple[float, float]]
    optimal_value: Optional[float]
    minimize: bool
    noise_sd: float
    # Whether to evaluate model-predicted best values at each iteration to
    # compare to the model predictions. Should only be `False` if the problem
    # is expensive to evaluate and therefore no extra evaluations beyond the one
    # optimization loop should be performed.
    evaluate_suggested: bool  # NOTE: not yet implemented

    def __init__(
        self,
        f: Union[SyntheticFunction, FunctionType],
        name: Optional[str] = None,
        domain: Optional[List[Tuple[float, float]]] = None,
        optimal_value: Optional[float] = None,
        minimize: bool = False,
        noise_sd: float = 0.0,
        evaluate_suggested: bool = True,
    ) -> None:
        # Whether we are using Ax `SyntheticFunction` custom ad-hoc function
        self.uses_synthetic_function = isinstance(f, SyntheticFunction)

        # Validate that domain is available, since it's used to make search space
        if domain is None:
            if not self.uses_synthetic_function:
                # Custom callable was passed as `f`, ensure presence of `domain`
                raise ValueError(NONE_DOMAIN_ERROR)
            elif checked_cast(SyntheticFunction, f).domain is None:
                # If no domain was passed, will use one from the `SyntheticFunction`
                raise ValueError(NONE_SYNTHETIC_FUNCTION_DOMAIN_ERROR)

        # Validate that if noise setting specified, known synthetic function is used
        if noise_sd != 0.0 and not self.uses_synthetic_function:
            raise ValueError(ADHOC_FUNCTION_NOISE_SET_ERROR)

        self.f = f
        self.name = name or (
            checked_cast(SyntheticFunction, f).name
            if self.uses_synthetic_function
            else checked_cast(FunctionType, f).__name__
        )
        if self.name == "<lambda>":
            raise ValueError(ANONYMOUS_FUNCTION_ERROR)
        # If domain is `None`, grab it from the `SyntheticFunction`
        # pyre-fixme[8]: Attribute has type `List[Tuple[float, float]]`; used as
        #  `Union[List[Tuple[float, float]], List[typing.Tuple[float, ...]]]`.
        self.domain = domain or checked_cast(SyntheticFunction, self.f).domain
        if optimal_value is None and self.uses_synthetic_function:
            # If no optimal_value is passed, try extracting it from synthetic function.
            try:
                synt_f = checked_cast(SyntheticFunction, self.f)
                self.optimal_value = synt_f.fmin if minimize else synt_f.fmax
            except NotImplementedError as err:  # optimal_value is unknown
                logger.warning(err)
        else:
            self.optimal_value = optimal_value

        self.minimize = minimize
        self.noise_sd = noise_sd
        self.evaluate_suggested = evaluate_suggested

    def domain_as_ax_client_parameters(
        self,
    ) -> List[Dict[str, Union[TParamValue, List[TParamValue]]]]:
        return [  # pyre-ignore[7]: Union subtype
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": list(self.domain[i]),
                "value_type": "float",
            }
            for i in range(len(self.domain))
        ]

    @equality_typechecker
    def __eq__(self, other: "BenchmarkProblem") -> bool:
        for field in self.__dict__.keys():
            self_val = getattr(self, field)
            other_val = getattr(other, field)
            if field == "f" and (
                (  # For synthetic functions, they are same if same class.
                    self.uses_synthetic_function
                    and self_val.__class__ is other_val.__class__
                )
                or (  # For custom callables, considered same if same name.
                    not self.uses_synthetic_function
                    and self_val.__name__ is other_val.__name__
                )
            ):
                continue
            if self_val != other_val:
                return False
        return True
