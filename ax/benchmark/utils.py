#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union, cast

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.modelbridge.generation_strategy import GenerationStrategy


def get_problems_and_methods(
    problems: Optional[Union[List[BenchmarkProblem], List[str]]] = None,
    methods: Optional[Union[List[GenerationStrategy], List[str]]] = None,
) -> Tuple[List[BenchmarkProblem], List[GenerationStrategy]]:
    """Validate problems and methods; find them by string keys if passed as
    strings.
    """
    if (
        problems is None
        or methods is None
        or not all(isinstance(p, BenchmarkProblem) for p in problems)
        or not all(isinstance(m, GenerationStrategy) for m in methods)
    ):
        raise NotImplementedError  # TODO (done in D18009570)
    return (
        cast(List[BenchmarkProblem], problems),
        cast(List[GenerationStrategy], methods),
    )


def get_corresponding(
    value_or_matrix: Union[int, List[List[int]]], row: int, col: int
) -> int:
    """If `value_or_matrix` is a matrix, extract the value in cell specified by
    `row` and `col`. If `value_or_matrix` is a scalar, just return it.
    """
    if isinstance(value_or_matrix, list):
        assert all(isinstance(x, list) for x in value_or_matrix)
        return value_or_matrix[row][col]
    assert isinstance(value_or_matrix, int)
    return value_or_matrix
