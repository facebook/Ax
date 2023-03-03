# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationStrategyCreatedRecord:
    """
    Record of the GenerationStrategy creation event. This can be used for telemetry in
    settings where many GenerationStrategy are being created either manually or
    programatically. In order to facilitate easy serialization only include simple
    types: numbers, strings, bools, and None.
    """

    generation_strategy_name: str

    # -1 indicates unlimited trials requested, 0 indicates no trials requested
    num_requested_initialization_trials: int  # Typically the number of Sobol trials
    num_requested_bayesopt_trials: int
    num_requested_other_trials: int

    max_parallelism: int  # Minimum `max_parallelism` across GenerationSteps
