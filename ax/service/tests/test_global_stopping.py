# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import numpy as np
from ax.core.types import TParameterization
from ax.exceptions.core import OptimizationShouldStop
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.service.ax_client import AxClient
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import branin
from ax.utils.testing.core_stubs import DummyGlobalStoppingStrategy


class TestGlobalStoppingIntegration(TestCase):
    def get_ax_client_for_branin(
        self,
        global_stopping_strategy: BaseGlobalStoppingStrategy,
    ) -> AxClient:
        """
        Instantiates an AxClient for the branin experiment with the specified
        global stopping strategy.
        """
        ax_client = AxClient(global_stopping_strategy=global_stopping_strategy)
        ax_client.create_experiment(
            name="branin_test_experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [0.0, 15.0],
                },
            ],
            objective_name="branin",
            minimize=True,
        )
        return ax_client

    def evaluate(self, parameters: TParameterization) -> Dict[str, Tuple[float, float]]:
        """Evaluates the parameters for branin experiment."""
        x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
        # pyre-fixme[7]: Expected `Dict[str, Tuple[float, float]]` but got
        #  `Dict[str, Tuple[Union[float, ndarray], float]]`.
        return {"branin": (branin(x), 0.0)}

    def test_global_stopping_integration(self) -> None:
        """
        Specifying a dummy global stopping strategy which stops
        the optimization after 3 trials are completed.
        """
        global_stopping_strategy = DummyGlobalStoppingStrategy(
            min_trials=2, trial_to_stop=3
        )
        ax_client = self.get_ax_client_for_branin(
            global_stopping_strategy=global_stopping_strategy
        )

        # Running the first 3 iterations.
        for _ in range(3):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index,
                # pyre-fixme[6]: For 2nd param expected `Union[Dict[str, Union[Tuple[...
                raw_data=self.evaluate(parameters),
            )

        # Trying to run the 4th iteration, which should raise
        exception = OptimizationShouldStop(message="Stop the optimization.")
        with self.assertRaises(OptimizationShouldStop) as cm:
            parameters, trial_index = ax_client.get_next_trial()
        # Assert Exception's message is unchanged.
        self.assertEqual(cm.exception.message, exception.message)

        # Trying to run the 4th iteration by overruling the stopping strategy.
        parameters, trial_index = ax_client.get_next_trial(force=True)
        self.assertIsNotNone(parameters)

        # Test the property & setter.
        self.assertIs(ax_client.global_stopping_strategy, global_stopping_strategy)
        new_gss = DummyGlobalStoppingStrategy(min_trials=5, trial_to_stop=5)
        ax_client.global_stopping_strategy = new_gss
        self.assertIs(ax_client.global_stopping_strategy, new_gss)
        self.assertIs(ax_client._global_stopping_strategy, new_gss)

    def test_min_trials(self) -> None:
        """
        Tests the min_trials mechanism of the stopping strategy; that is,
        the stopping strategy should not take effect before min_trials trials
        are completed.
        """
        global_stopping_strategy = DummyGlobalStoppingStrategy(
            min_trials=3, trial_to_stop=2
        )
        ax_client = self.get_ax_client_for_branin(
            global_stopping_strategy=global_stopping_strategy
        )

        # Running the first 2 iterations.
        for _ in range(2):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index,
                # pyre-fixme[6]: For 2nd param expected `Union[Dict[str, Union[Tuple[...
                raw_data=self.evaluate(parameters),
            )

        # Since min_trials=3, GSS should not stop creating the 3rd iteration.
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index,
            # pyre-fixme[6]: For 2nd param expected `Union[Dict[str, Union[Tuple[Unio...
            raw_data=self.evaluate(parameters),
        )
        self.assertIsNotNone(parameters)

        # Now, GSS should stop creating the 4th iteration.
        exception = OptimizationShouldStop(message="Stop the optimization.")
        with self.assertRaises(OptimizationShouldStop) as cm:
            parameters, trial_index = ax_client.get_next_trial()
        # Assert Exception's message is unchanged.
        self.assertEqual(cm.exception.message, exception.message)
