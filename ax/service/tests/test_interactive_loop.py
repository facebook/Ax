#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import functools
import time
from logging import WARN
from typing import Optional, Tuple

import numpy as np
from ax.core.types import TEvaluationOutcome
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, TParameterization
from ax.service.interactive_loop import interactive_optimize_with_client
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.testing.mock import fast_botorch_optimize


class TestInteractiveLoop(TestCase):
    @fast_botorch_optimize
    def test_interactive_loop(self) -> None:
        def _elicit(
            parameterization_with_trial_index: Tuple[TParameterization, int]
        ) -> Optional[Tuple[int, TEvaluationOutcome]]:
            parameterization, trial_index = parameterization_with_trial_index
            x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])

            return (
                trial_index,
                {
                    "hartmann6": (hartmann6(x), 0.0),
                    "l2norm": (np.sqrt((x**2).sum()), 0.0),
                },
            )

        def _aborted_elicit(
            parameterization_with_trial_index: Tuple[TParameterization, int]
        ) -> Optional[Tuple[int, TEvaluationOutcome]]:
            return None

        ax_client = AxClient()
        ax_client.create_experiment(
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

        optimization_completed = interactive_optimize_with_client(
            ax_client=ax_client,
            num_trials=15,
            candidate_queue_maxsize=3,
            # pyre-fixme[6]
            elicitation_function=_elicit,
        )

        self.assertTrue(optimization_completed)
        self.assertEqual(len(ax_client.experiment.trials), 15)

        # test failed experiment
        optimization_completed = interactive_optimize_with_client(
            ax_client=ax_client,
            num_trials=15,
            candidate_queue_maxsize=3,
            # pyre-fixme[6]
            elicitation_function=_aborted_elicit,
        )
        self.assertFalse(optimization_completed)

    def test_candidate_pregeneration_errors_raised(self) -> None:
        def _elicit(
            parameterization_with_trial_index: Tuple[TParameterization, int]
        ) -> Tuple[int, TEvaluationOutcome]:
            parameterization, trial_index = parameterization_with_trial_index
            time.sleep(0.15)  # Sleep to induce MaxParallelismException in loop

            x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])

            return (
                trial_index,
                {
                    "hartmann6": (hartmann6(x), 0.0),
                    "l2norm": (np.sqrt((x**2).sum()), 0.0),
                },
            )

        # GS with low max parallelismm to induce MaxParallelismException:
        generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, max_parallelism=1, num_trials=-1)]
        )
        ax_client = AxClient(generation_strategy=generation_strategy)
        ax_client.create_experiment(
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

        with self.assertLogs(logger="ax", level=WARN) as logger:
            interactive_optimize_with_client(
                ax_client=ax_client,
                num_trials=3,
                candidate_queue_maxsize=3,
                # pyre-fixme[6]
                elicitation_function=_elicit,
            )

            # Assert sleep and retry warning is somewhere in the logs
            self.assertIn(
                "sleeping for 0.1 seconds and trying again.",
                functools.reduce(lambda left, right: left + right, logger.output),
            )
