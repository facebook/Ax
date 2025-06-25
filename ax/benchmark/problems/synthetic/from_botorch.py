# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Functions that create ``BenchmarkProblem``s based on BoTorch test functions.
"""

from typing import Literal

from ax.benchmark.benchmark_problem import BenchmarkProblem, create_problem_from_botorch
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from botorch.test_functions.multi_fidelity import AugmentedBranin


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
