# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.testing.benchmark_stubs import (
    DeterministicGenerationNode,
    get_discrete_search_space,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.utils.common.testutils import TestCase
from pyre_extensions import assert_is_instance


class TestDeterministicGenerationNode(TestCase):
    def test_deterministic_node(self) -> None:
        search_space = get_discrete_search_space(n_values=5)
        node = DeterministicGenerationNode(search_space=search_space)
        gs = GenerationStrategy(name="test gs", nodes=[node])

        # Create a simple experiment for testing
        experiment = Experiment(
            name="test",
            search_space=search_space,
            optimization_config=None,
            is_test=True,
        )

        # Should be None before generating candidates
        self.assertIsNone(
            assert_is_instance(gs._nodes[0], DeterministicGenerationNode)._iterator_gs
        )
        # Generate some candidates to advance the iterator
        gr1 = gs.gen_single_trial(experiment=experiment)

        # Should be set to current GS after generating candidates
        self.assertIsInstance(
            assert_is_instance(gs._nodes[0], DeterministicGenerationNode)._iterator_gs,
            GenerationStrategy,
        )
        gr2 = gs.gen_single_trial(experiment=experiment)
        self.assertEqual(gr1.arms[0].parameters, {"x0": 0})
        self.assertEqual(gr2.arms[0].parameters, {"x0": 1})

        # Clone and reset the generation strategy
        cloned_gs = gs.clone_reset()

        # Generate candidates from the cloned strategy - should start from the beginning
        cloned_gr1 = cloned_gs.gen_single_trial(experiment=experiment)
        cloned_gr2 = cloned_gs.gen_single_trial(experiment=experiment)

        self.assertEqual(cloned_gr1.arms[0].parameters, {"x0": 0})
        self.assertEqual(cloned_gr2.arms[0].parameters, {"x0": 1})
