#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock

from ax.adapter.random import RandomAdapter

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_sobol,
)
from pyre_extensions import none_throws


class DummyNode(ExternalGenerationNode):
    def __init__(self, should_deduplicate: bool = False) -> None:
        super().__init__(node_name="dummy", should_deduplicate=should_deduplicate)
        self.update_count = 0
        self.gen_count = 0
        self.generator: RandomAdapter | None = None
        self.last_pending: list[TParameterization] = []

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        self.update_count += 1
        self.generator = get_sobol(experiment.search_space)

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        self.gen_count += 1
        self.last_pending = deepcopy(pending_parameters)
        return none_throws(self.generator).gen(n=1).arms[0].parameters


class TestExternalGenerationNode(TestCase):
    def test_properties(self) -> None:
        node = DummyNode()
        self.assertEqual(node.node_name, "dummy")
        self.assertGreater(node.fit_time_since_gen, 0.0)
        self.assertIsNone(node._fitted_adapter)
        self.assertIsNone(node.generator_spec_to_gen_from)
        self.assertFalse(node.should_deduplicate)
        with self.assertRaisesRegex(UnsupportedError, "Unexpected arguments"):
            node._fit(
                experiment=MagicMock(), data=MagicMock(), search_space=MagicMock()
            )

    def test_generation(self) -> None:
        node = DummyNode()
        gs = GenerationStrategy(name="test gs", nodes=[node])
        self.assertEqual(node.gen_count, 0)
        self.assertEqual(node.update_count, 0)
        experiment = get_branin_experiment()

        # Sequential generation.
        for _ in range(3):
            gr = gs.gen_single_trial(
                n=1, experiment=experiment, data=experiment.lookup_data()
            )
            trial = experiment.new_trial(generator_run=gr)
            trial.mark_running(no_runner_required=True)
            experiment.attach_data(get_branin_data(trials=[trial]))
            trial.mark_completed()
        self.assertEqual(node.gen_count, 3)
        self.assertEqual(node.update_count, 3)
        self.assertEqual(node.last_pending, [])

        # Test pending point handling.
        pending_observations = {
            "some_metric": [ObservationFeatures(parameters={"x1": 0.123, "x2": 0.456})]
        }
        gr = gs.gen_single_trial(
            n=1,
            experiment=experiment,
            data=experiment.lookup_data(),
            pending_observations=pending_observations,
        )
        trial = experiment.new_trial(generator_run=gr)
        trial.mark_running(no_runner_required=True)
        experiment.attach_data(get_branin_data(trials=[trial]))
        trial.mark_completed()
        self.assertEqual(node.gen_count, 4)
        self.assertEqual(node.update_count, 4)
        self.assertEqual(node.last_pending, [{"x1": 0.123, "x2": 0.456}])

        # Batch generation.
        gr = gs.gen_single_trial(
            n=5, experiment=experiment, data=experiment.lookup_data()
        )
        self.assertEqual(node.gen_count, 9)
        self.assertEqual(node.update_count, 5)
        self.assertEqual(len(gr.arms), 5)
        self.assertGreater(none_throws(gr.fit_time), 0.0)
        self.assertGreater(none_throws(gr.gen_time), 0.0)
        self.assertEqual(gr._model_key, "dummy")
        self.assertEqual(len(node.last_pending), 4)

    def test_fallback(self) -> None:
        node = DummyNode(should_deduplicate=True)
        gs = GenerationStrategy(name="test gs", nodes=[node])
        experiment = get_branin_experiment()
        params = {"x1": 0.0, "x2": 0.0}
        with mock.patch(
            "ax.adapter.random.RandomAdapter.gen",
            return_value=GeneratorRun(arms=[Arm(parameters=params)], model_key="Sobol"),
        ) as mock_gen:
            experiment.new_trial(
                generator_run=gs.gen_single_trial(n=1, experiment=experiment)
            ).run()
            experiment.new_trial(
                generator_run=gs.gen_single_trial(n=1, experiment=experiment)
            ).run()
        self.assertEqual(node.gen_count, 6)  # first trial + 5 deduplication attempts
        self.assertEqual(mock_gen.call_count, 7)  # also called for the fallback
        self.assertEqual(node.update_count, 2)
        # First trial was generated by the node, second by the fallback.
        self.assertEqual(experiment.trials[0].generator_runs[0]._model_key, "dummy")
        self.assertEqual(experiment.trials[1].generator_runs[0]._model_key, "Sobol")
