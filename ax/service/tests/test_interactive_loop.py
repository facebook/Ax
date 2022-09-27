#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import time
from logging import WARN

import numpy as np
from ax.core.types import TEvaluationOutcome
from ax.service.ax_client import AxClient, TParameterization
from ax.service.interactive_loop import interactive_optimize_with_client
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.testing.mock import fast_botorch_optimize


class TestInteractiveLoop(TestCase):
    def setUp(self) -> None:
        super().setUp()

    @fast_botorch_optimize
    def test_interactive_loop(self) -> None:
        def _elicit(parameterization: TParameterization) -> TEvaluationOutcome:
            x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])

            return {
                "hartmann6": (hartmann6(x), 0.0),
                "l2norm": (np.sqrt((x**2).sum()), 0.0),
            }

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
            objective_name="hartmann6",
            tracking_metric_names=["l2norm"],
            minimize=True,
        )

        interactive_optimize_with_client(
            ax_client=ax_client,
            num_trials=15,
            candidate_queue_maxsize=3,
            elicitation_function=_elicit,
        )

        self.assertEqual(len(ax_client.experiment.trials), 15)

    def test_candidate_pregeneration_errors_raised(self) -> None:
        def _elicit(parameterization: TParameterization) -> TEvaluationOutcome:
            time.sleep(0.15)  # Sleep to induce MaxParallelismException in loop

            x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])

            return {
                "hartmann6": (hartmann6(x), 0.0),
                "l2norm": (np.sqrt((x**2).sum()), 0.0),
            }

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
            objective_name="hartmann6",
            tracking_metric_names=["l2norm"],
            minimize=True,
        )

        # Lower max parallelism to induce MaxParallelismException
        ax_client.generation_strategy._steps[0].max_parallelism = 1

        with self.assertLogs(logger="ax", level=WARN) as logger:
            interactive_optimize_with_client(
                ax_client=ax_client,
                num_trials=3,
                candidate_queue_maxsize=3,
                elicitation_function=_elicit,
            )

            # Assert sleep and retry warning is somewhere in the logs
            self.assertIn(
                "sleeping for 0.1 seconds and trying again.",
                functools.reduce(lambda left, right: left + right, logger.output),
            )
