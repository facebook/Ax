#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Optional
from unittest.mock import MagicMock

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.external_generation_node import ExternalGenerationNode
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_sobol,
)


class DummyNode(ExternalGenerationNode):
    def __init__(self) -> None:
        super().__init__(node_name="dummy")
        self.update_count = 0
        self.gen_count = 0
        self.generator: Optional[RandomModelBridge] = None

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        self.update_count += 1
        self.generator = get_sobol(experiment.search_space)

    def get_next_candidate(
        self, pending_parameters: List[TParameterization]
    ) -> TParameterization:
        self.gen_count += 1
        return not_none(self.generator).gen(n=1).arms[0].parameters


class TestExternalGenerationNode(TestCase):
    def test_properties(self) -> None:
        node = DummyNode()
        self.assertEqual(node.node_name, "dummy")
        self.assertGreater(node.fit_time_since_gen, 0.0)
        self.assertIsNone(node._fitted_model)
        self.assertIsNone(node.model_spec_to_gen_from)
        with self.assertRaisesRegex(UnsupportedError, "Unexpected arguments"):
            node.fit(experiment=MagicMock(), data=MagicMock(), search_space=MagicMock())

    def test_generation(self) -> None:
        node = DummyNode()
        gs = GenerationStrategy(name="test gs", nodes=[node])
        self.assertEqual(node.gen_count, 0)
        self.assertEqual(node.update_count, 0)
        experiment = get_branin_experiment()
        # Sequential generation.
        for _ in range(3):
            gr = gs.gen(n=1, experiment=experiment, data=experiment.lookup_data())
            trial = experiment.new_trial(generator_run=gr)
            trial.mark_running(no_runner_required=True)
            experiment.attach_data(get_branin_data(trials=[trial]))
            trial.mark_completed()
        self.assertEqual(node.gen_count, 3)
        self.assertEqual(node.update_count, 3)
        # Batch generation.
        gr = gs.gen(n=5, experiment=experiment, data=experiment.lookup_data())
        self.assertEqual(node.gen_count, 8)
        self.assertEqual(node.update_count, 4)
        self.assertEqual(len(gr.arms), 5)
        self.assertGreater(not_none(gr.fit_time), 0.0)
        self.assertGreater(not_none(gr.gen_time), 0.0)
        self.assertEqual(gr._model_key, "dummy")
