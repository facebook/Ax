# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

EXPERIMENT_DESIGN_KEY: str = "experiment_design"


@dataclass
class ExperimentDesign:
    """Struct that holds "experiment design" configuration: these are
    experiment-level settings that pertain to "how the experiment will be
    run or conducted", but are agnostic to the specific evaluation
    backend, to which the trials will be deployed.

    NOTE: In the future, we might treat concurrency limit as expressed
    in terms of "full arm equivalents" as opposed to just "number of arms",
    to cover for the multi-fidelity cases.

    NOTE: in ax/storage/sqa_store/encoder.py, attributes of this class
    are automatically serialized and stored in experiment.properties

    Args:
        concurrency_limit: Maximum number of arms to run within one or
            multiple trials, in parallel. In experiments that consist of
            Trials, this is equivalent to the total number of trials
            that should run in parallel. In experiments with BatchTrials,
            total number of arms can be spread across one or
            multiple BatchTrials.
    """

    concurrency_limit: int | None = None
