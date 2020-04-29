#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing  # noqa F401, this is to enable type-checking

from ax.exceptions.core import AxError


class MaxParallelismReachedException(AxError):
    """Special exception indicating that maximum number of trials running in
    parallel set on a given step (as `GenerationStep.max_parallelism`) has been
    reached. Upon getting this exception, users should wait until more trials
    are completed with data, to generate new trials.
    """

    def __init__(self, step_index: int, model_name: str, num_running: int) -> None:
        super().__init__(
            f"Maximum parallelism for generation step #{step_index} ({model_name})"
            f" has been reached: {num_running} trials are currently 'running'. Some "
            "trials need to be completed before more trials can be generated. See "
            "https://ax.dev/docs/bayesopt.html to understand why limited parallelism "
            "improves performance of Bayesian optimization."
        )


class GenerationStrategyCompleted(AxError):
    """Special exception indicating that the generation strategy has been
    completed.
    """

    pass
