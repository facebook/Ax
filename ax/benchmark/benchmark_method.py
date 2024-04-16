# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from dataclasses import dataclass
from typing import Any

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.utils.scheduler_options import SchedulerOptions, TrialType
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger: logging.Logger = get_logger("BenchmarkMethod")


@dataclass(frozen=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and scheduler options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.).

    Note: If `BenchmarkMethod.scheduler_options.total_trials` is less than
    `BenchmarkProblem.num_trials` then only the number of trials specified in the
    former will be run.

    Note: The `generation_strategy` passed in is assumed to be in its "base state",
    as it will be cloned and reset.
    """

    name: str
    generation_strategy: GenerationStrategy
    scheduler_options: SchedulerOptions
    distribute_replications: bool = False

    def __post_init__(self) -> None:
        # We (I think?) in general don't want to fit tracking metrics during our
        # benchmarks. Further, not setting `fit_tracking_metrics=False`causes
        # issues with the ground truth metrics created automatically when running
        # the benchmark - in fact, things will error out deep inside the modeling
        # stack since the model gets both noisy (benchmark) and noiseless (ground
        # truth) observations. While support for this is something we shold add
        # for models, in the context of benchmarking we actually want to avoid
        # fitting the ground truth metrics at all.

        # Clone the GS so as to not modify the original one in-place below.
        # Note that this assumes that the GS passed in is in its base state.
        gs_cloned = self.generation_strategy.clone_reset()

        for node in gs_cloned._nodes:
            if isinstance(node, GenerationStep):
                if node.model_kwargs is None:
                    node.model_kwargs = {}
                if node.model_kwargs.get("fit_tracking_metrics", True):
                    logger.info(
                        "Setting `fit_tracking_metrics` in a GenerationStep to False.",
                    )
                    not_none(node.model_kwargs)["fit_tracking_metrics"] = False
            for model_spec in node.model_specs:
                if model_spec.model_kwargs is None:
                    model_spec.model_kwargs = {}
                elif model_spec.model_kwargs.get("fit_tracking_metrics", True):
                    logger.info(
                        "Setting `fit_tracking_metrics` in a GenerationNode's "
                        "model_spec to False."
                    )
                    not_none(model_spec.model_kwargs)["fit_tracking_metrics"] = False

        # hack around not being able to update frozen attribute of a dataclass
        _assign_frozen_attr(self, name="generation_strategy", value=gs_cloned)


def get_benchmark_scheduler_options(
    timeout_hours: int = 4,
    batch_size: int = 1,
) -> SchedulerOptions:
    """The typical SchedulerOptions used in benchmarking.

    Currently, regardless of batch size, all pending trials must complete before
    new ones are generated. That is, when batch_size > 1, the design is "batch
    sequential", and when batch_size = 1, the design is "fully sequential."

    Args:
        timeout_hours: The maximum amount of time (in hours) to run each
            benchmark replication. Defaults to 4 hours.
        batch_size: Number of trials to generate at once.
    """

    return SchedulerOptions(
        # No new candidates can be generated while any are pending.
        # If batched, an entire batch must finish before the next can be
        # generated.
        max_pending_trials=1,
        # Do not throttle, as is often necessary when polling real endpoints
        init_seconds_between_polls=0,
        min_seconds_before_poll=0,
        timeout_hours=timeout_hours,
        trial_type=TrialType.TRIAL if batch_size == 1 else TrialType.BATCH_TRIAL,
        batch_size=batch_size,
    )


def _assign_frozen_attr(obj: Any, name: str, value: Any) -> None:  # pyre-ignore [2]
    """Assign a new value to an attribute of a frozen dataclass.
    This is an ugly hack and shouldn't be used broadly.
    """
    object.__setattr__(obj, name, value)
