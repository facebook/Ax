# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Functions that create ``BenchmarkProblem``s based on BoTorch test functions.
"""

from collections.abc import Mapping
from typing import Any, Literal

from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    get_continuous_search_space,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.benchmark_step_runtime_function import TBenchmarkStepRuntimeFunction
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from botorch.test_functions.base import (
    BaseTestProblem,
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.test_functions.multi_fidelity import AugmentedBranin
from pyre_extensions import assert_is_instance

# A mapping from (BoTorch problem class name, dim | None) to baseline value
# Obtained using `get_baseline_value_from_sobol`
BOTORCH_BASELINE_VALUES: Mapping[tuple[str, int | None], float] = {
    ("Ackley", 4): 19.837273921447853,
    ("Branin", None): 10.930455126654936,
    ("BraninCurrin", None): 0.9820209831769217,
    ("BraninCurrin", 30): 3.0187520516793587,
    ("ConstrainedGramacy", None): 1.0643958597443999,
    ("ConstrainedBraninCurrin", None): 0.9820209831769217,
    ("Griewank", 4): 60.037068040081095,
    ("Hartmann", 3): -2.3423173903286716,
    ("Hartmann", 6): -0.796988050854654,
    ("Hartmann", 30): -0.8359462084890045,
    ("KeaneBumpFunction", 2): -0.0799243632005311,
    ("KeaneBumpFunction", 10): -0.13609325522288143,
    ("Levy", 4): 14.198811442165178,
    ("Powell", 4): 932.3102865964689,
    ("PressureVessel", None): float("inf"),
    ("Rosenbrock", 4): 30143.767857949348,
    ("SixHumpCamel", None): 0.45755007063109004,
    ("SpeedReducer", None): float("inf"),
    ("TensionCompressionString", None): float("inf"),
    ("ThreeHumpCamel", None): 3.7321680621434155,
    ("WeldedBeamSO", None): float("inf"),
}


def _get_name(
    test_problem: BaseTestProblem,
    observe_noise_sd: bool,
    dim: int | None = None,
    n_dummy_dimensions: int = 0,
) -> str:
    """
    Get a string name describing the problem, in a format such as
    "hartmann_fixed_noise_6d" or "jenatton" (where the latter would
    not have fixed noise and have the default dimensionality).
    """
    base_name = f"{test_problem.__class__.__name__}"
    observed_noise = "_observed_noise" if observe_noise_sd else ""
    if dim is None and n_dummy_dimensions == 0:
        dim_str = ""
    else:
        total_dim = (test_problem.dim if dim is None else dim) + n_dummy_dimensions
        dim_str = f"_{total_dim}d"
    return f"{base_name}{observed_noise}{dim_str}"


def create_problem_from_botorch(
    *,
    test_problem_class: type[BaseTestProblem],
    test_problem_kwargs: dict[str, Any],
    noise_std: float | list[float] = 0.0,
    num_trials: int,
    baseline_value: float | None = None,
    name: str | None = None,
    lower_is_better: bool = True,
    observe_noise_sd: bool = False,
    use_shifted_function: bool = False,
    search_space: SearchSpace | None = None,
    report_inference_value_as_trace: bool = False,
    step_runtime_function: TBenchmarkStepRuntimeFunction | None = None,
    status_quo_params: TParameterization | None = None,
    auxiliary_experiments_by_purpose: (
        dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]] | None
    ) = None,
    n_dummy_dimensions: int = 0,
    use_map_metric: bool = False,
    n_steps: int = 1,
) -> BenchmarkProblem:
    """
    Create a ``BenchmarkProblem`` from a BoTorch ``BaseTestProblem``.

    The resulting ``BenchmarkProblem``'s ``test_function`` is constructed from
    the ``BaseTestProblem`` class (``test_problem_class``) and its arguments
    (``test_problem_kwargs``). All other fields are passed to
    ``BenchmarkProblem`` if they are specified and populated with reasonable
    defaults otherwise. ``num_trials``, however, must be specified.

    Args:
        test_problem_class: The BoTorch test problem class which will be used
            to define the `search_space`, `optimization_config`, and `runner`.
        test_problem_kwargs: Keyword arguments used to instantiate the
            `test_problem_class`. This should *not* include `noise_std` or
            `negate`, since these are handled through Ax benchmarking (as the
            `noise_std` and `lower_is_better` arguments to `BenchmarkProblem`).
        noise_std: Standard deviation of synthetic noise added to outcomes. If a
            float, the same noise level is used for all objectives.
        lower_is_better: Whether this is a minimization problem. For MOO, this
            applies to all objectives.
        num_trials: Simply the `num_trials` of the `BenchmarkProblem` created.
        baseline_value: If not provided, will be looked up from
            `BOTORCH_BASELINE_VALUES`.
        name: Will be passed to ``BenchmarkProblem`` if specified and populated
            with reasonable defaults otherwise.
        observe_noise_sd: Whether the standard deviation of the observation noise is
            observed or not (in which case it must be inferred by the model).
            This is separate from whether synthetic noise is added to the
            problem, which is controlled by the `noise_std` of the test problem.
        use_shifted_function: Whether to use the shifted version of the test function.
            If True, an offset tensor is randomly drawn from the test problem bounds,
            and the we evaluate `f(X-offset)` rather than `f(X)`. This is useful for
            changing the location of the optima for test functions that favor the
            center of the search space.
            If True, the default search space is enlarged to include the optimizer
            within the bounds even after the offset is applied. If the original
            bounds were `[low, high]`, the new bounds are `[2*low, 2*high]`.
        search_space: If provided, the `search_space` of the `BenchmarkProblem`.
            Otherwise, a `SearchSpace` with all `RangeParameter`s is created
            from the bounds of the test problem.
        report_inference_value_as_trace: If True, indicates that the
            ``optimization_trace`` on a ``BenchmarkResult`` ought to be the
            ``inference_trace``; otherwise, it will be the ``oracle_trace``.
            See ``BenchmarkResult`` for more information.
        status_quo_params: The status quo parameters for the problem.
        step_runtime_function: Optionally, a function that takes in ``params``
            (typically dictionaries mapping strings to ``TParamValue``s) and
            returns the runtime of an step. If ``step_runtime_function`` is
            left as ``None``, each step will take one simulated second.  (When
            data is not time-series, the whole trial consists of one step.)
        auxiliary_experiments_by_purpose: A mapping from experiment purpose to
            a list of auxiliary experiments.
        n_dummy_dimensions: If >0, the search space will be augmented
            with extra dimensions. The corresponding parameters will have no
            effect on function values.
        use_map_metric: Whether to use a ``BenchmarkMapMetric`` (rather than a
            ``BenchmarkMetric``).
        n_steps: Number of steps (progression values) in each evaluation. The
            default of 1 reflects a normal synthetic function evaluation. A
            higher number results in repeating the evaluation and getting the
            same result ``n_steps`` times (before IID noise is added).

    Example:
        >>> from ax.benchmark.benchmark_problem import create_problem_from_botorch
        >>> from botorch.test_functions.synthetic import Branin
        >>> problem = create_problem_from_botorch(
        ...    test_problem_class=Branin,
        ...    test_problem_kwargs={},
        ...    noise_std=0.1,
        ...    num_trials=10,
        ...    observe_noise_sd=True,
        ...    step_runtime_function=lambda params: 1 / params["fidelity"],
        ... )
    """
    # pyre-fixme [45]: Invalid class instantiation
    test_problem = test_problem_class(**test_problem_kwargs)
    is_constrained = isinstance(test_problem, ConstrainedBaseTestProblem)

    if search_space is None:
        if use_shifted_function:
            search_space = get_continuous_search_space(
                bounds=[(2 * b[0], 2 * b[1]) for b in test_problem._bounds],
                n_dummy_dimensions=n_dummy_dimensions,
            )
        else:
            search_space = get_continuous_search_space(
                bounds=test_problem._bounds, n_dummy_dimensions=n_dummy_dimensions
            )

    dim = test_problem_kwargs.get("dim", None)

    n_obj = test_problem.num_objectives
    if name is None:
        name = _get_name(
            test_problem=test_problem,
            observe_noise_sd=observe_noise_sd,
            dim=dim,
            n_dummy_dimensions=n_dummy_dimensions,
        )

    num_constraints = test_problem.num_constraints if is_constrained else 0
    if isinstance(test_problem, MultiObjectiveTestProblem):
        # pyre-fixme[6]: For 1st argument expected `SupportsIndex` but got
        #  `Union[Tensor, Module]`.
        objective_names = [f"{name}_{i}" for i in range(n_obj)]
    else:
        objective_names = [name]

    # pyre-fixme[6]: For 1st argument expected `SupportsIndex` but got `Union[int,
    #  Tensor, Module]`.
    constraint_names = [f"constraint_slack_{i}" for i in range(num_constraints)]
    outcome_names = objective_names + constraint_names

    test_function = BoTorchTestFunction(
        botorch_problem=test_problem,
        outcome_names=outcome_names,
        use_shifted_function=use_shifted_function,
        dummy_param_names={
            n for n in search_space.parameters if "embedding_dummy_" in n
        },
        n_steps=n_steps,
    )

    if isinstance(test_problem, MultiObjectiveTestProblem):
        optimization_config = get_moo_opt_config(
            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[int,
            #  Tensor, Module]`.
            num_constraints=num_constraints,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            outcome_names=test_function.outcome_names,
            ref_point=test_problem._ref_point,
            use_map_metric=use_map_metric,
        )
    else:
        optimization_config = get_soo_opt_config(
            outcome_names=test_function.outcome_names,
            lower_is_better=lower_is_better,
            observe_noise_sd=observe_noise_sd,
            use_map_metric=use_map_metric,
        )

    optimal_value = (
        test_problem.max_hv
        if isinstance(test_problem, MultiObjectiveTestProblem)
        else test_problem.optimal_value
    )
    if len(optimization_config.outcome_constraints) > 0:
        worst_feasible_value = assert_is_instance(
            test_problem.worst_feasible_value, float
        )
    else:
        worst_feasible_value = None  # Not needed for unconstrained problems
    baseline_value = (
        BOTORCH_BASELINE_VALUES[(test_problem_class.__name__, dim)]
        if baseline_value is None
        else baseline_value
    )

    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        test_function=test_function,
        noise_std=noise_std,
        num_trials=num_trials,
        optimal_value=assert_is_instance(optimal_value, float),
        baseline_value=baseline_value,
        worst_feasible_value=worst_feasible_value,
        report_inference_value_as_trace=report_inference_value_as_trace,
        step_runtime_function=step_runtime_function,
        status_quo_params=status_quo_params,
        auxiliary_experiments_by_purpose=auxiliary_experiments_by_purpose,
    )


def get_augmented_branin_search_space(
    fidelity_or_task: Literal["fidelity", "task"],
) -> SearchSpace:
    """
    Get the ``SearchSpace`` that matches the ``AugmentedBranin`` test problem.

    ``AugmentedBranin`` has an extra parameter beyond the normal two which has
    been treated as a fidelity parameter.

    Args:
        fidelity_or_task: If "fidelity", the extra parameter is a fidelity
            parameter and will be continuous, because fidelity ChoiceParameters
            can't be used with the ``OrderedChoiceToIntegerRange`` transform. If
            "task", the extra parameter is a task parameter and is discrete,
            because a ``RangeParameter`` cannot be a task.
    """
    if fidelity_or_task == "fidelity":
        extra_parameter = RangeParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
            is_fidelity=True,
            target_value=1,
        )
    else:
        extra_parameter = ChoiceParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            values=[0, 1],
            is_fidelity=False,
            is_task=True,
            target_value=1,
        )
    parameters = [
        RangeParameter(
            name=f"x{i}",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        )
        for i in range(2)
    ] + [extra_parameter]
    return SearchSpace(parameters=parameters)


def get_augmented_branin_problem(
    fidelity_or_task: Literal["fidelity", "task"],
    report_inference_value_as_trace: bool = True,
) -> BenchmarkProblem:
    """
    Get a Branin problem with a fidelity or task parameter.

    Args:
        fidelity_or_task: If "fidelity", the extra parameter is a fidelity
            parameter. If "task", the extra parameter is a task parameter.
        report_inference_value_as_trace: Passed to
            ``create_problem_from_botorch`` then to ``BenchmarkProblem``.
    """

    return create_problem_from_botorch(
        test_problem_class=AugmentedBranin,
        test_problem_kwargs={},
        search_space=get_augmented_branin_search_space(
            fidelity_or_task=fidelity_or_task
        ),
        num_trials=3,
        baseline_value=3.0,
        report_inference_value_as_trace=report_inference_value_as_trace,
    )
