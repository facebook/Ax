#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import functools
import time
from logging import WARN
from queue import Queue
from threading import Event, Lock

import numpy as np

from ax.core.types import TEvaluationOutcome, TParameterization
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.service.interactive_loop import (
    ax_client_data_attacher,
    interactive_optimize,
    interactive_optimize_with_client,
)
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import hartmann6


class TestInteractiveLoop(TestCase):
    def setUp(self) -> None:
        generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, max_parallelism=1, num_trials=-1)]
        )
        self.ax_client = AxClient(generation_strategy=generation_strategy)
        self.ax_client.create_experiment(
            name="hartmann_test_experiment",
            # pyre-fixme[6]
            parameters=[
                {
                    "name": f"x{i}",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                }
                for i in range(1, 7)
            ],
            objectives={"hartmann6": ObjectiveProperties(minimize=True)},
            tracking_metric_names=["l2norm"],
        )

    def _elicit(
        self,
        parameterization_with_trial_index: tuple[TParameterization, int],
    ) -> tuple[int, TEvaluationOutcome] | None:
        parameterization, trial_index = parameterization_with_trial_index
        x = np.array([parameterization.get(f"x{i + 1}") for i in range(6)])

        return (
            trial_index,
            {
                "hartmann6": (hartmann6(x), 0.0),
                "l2norm": (np.sqrt((x**2).sum()), 0.0),
            },
        )

    def test_interactive_loop(self) -> None:
        optimization_completed = interactive_optimize_with_client(
            ax_client=self.ax_client,
            num_trials=15,
            candidate_queue_maxsize=3,
            # pyre-fixme[6]
            elicitation_function=self._elicit,
        )

        self.assertTrue(optimization_completed)
        self.assertEqual(len(self.ax_client.experiment.trials), 15)

    def test_interactive_loop_aborted(self) -> None:
        # Abort from elicitation function
        def _aborted_elicit(
            parameterization_with_trial_index: tuple[TParameterization, int],
        ) -> tuple[int, TEvaluationOutcome] | None:
            return None

        optimization_completed = interactive_optimize_with_client(
            ax_client=self.ax_client,
            num_trials=15,
            candidate_queue_maxsize=3,
            # pyre-fixme[6]
            elicitation_function=_aborted_elicit,
        )
        self.assertFalse(optimization_completed)

        # Abort from candidate_generator
        def ax_client_candidate_generator(
            queue: Queue[tuple[TParameterization, int] | None],
            stop_event: Event,
            num_trials: int,
            lock: Lock,
        ) -> None:
            with lock:
                queue.put(None)
                stop_event.set()

        ax_client_lock = Lock()
        optimization_completed = interactive_optimize(
            num_trials=15,
            candidate_queue_maxsize=3,
            candidate_generator_function=ax_client_candidate_generator,
            candidate_generator_kwargs={"lock": ax_client_lock},
            data_attacher_function=ax_client_data_attacher,
            data_attacher_kwargs={"ax_client": self.ax_client, "lock": ax_client_lock},
            elicitation_function=self._elicit,
        )
        self.assertFalse(optimization_completed)

    def test_candidate_pregeneration_errors_raised(self) -> None:
        def _sleep_elicit(
            parameterization_with_trial_index: tuple[TParameterization, int],
        ) -> tuple[int, TEvaluationOutcome]:
            parameterization, trial_index = parameterization_with_trial_index
            time.sleep(0.15)  # Sleep to induce MaxParallelismException in loop

            x = np.array([parameterization.get(f"x{i + 1}") for i in range(6)])

            return (
                trial_index,
                {
                    "hartmann6": (hartmann6(x), 0.0),
                    "l2norm": (np.sqrt((x**2).sum()), 0.0),
                },
            )

        with self.assertLogs(logger="ax", level=WARN) as logger:
            interactive_optimize_with_client(
                ax_client=self.ax_client,
                num_trials=3,
                candidate_queue_maxsize=3,
                # pyre-fixme[6]
                elicitation_function=_sleep_elicit,
            )

            # Assert sleep and retry warning is somewhere in the logs
            self.assertIn(
                "sleeping for 0.1 seconds and trying again.",
                functools.reduce(lambda left, right: left + right, logger.output),
            )
