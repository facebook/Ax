#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_discrete_search_space,
    get_experiment,
    get_factorial_search_space,
)


class TestDispatchUtils(TestCase):
    """Tests that dispatching utilities correctly select generation strategies.
    """

    def test_choose_generation_strategy(self):
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())
        self.assertEqual(sobol_gpei._steps[0].model.value, "Sobol")
        self.assertEqual(sobol_gpei._steps[1].model.value, "GPEI")
        sobol = choose_generation_strategy(search_space=get_factorial_search_space())
        self.assertEqual(sobol._steps[0].model.value, "Sobol")
        self.assertEqual(len(sobol._steps), 1)

    def test_setting_random_seed(self):
        sobol = choose_generation_strategy(
            search_space=get_factorial_search_space(), random_seed=9
        )
        sobol.gen(experiment=get_experiment())
        # First model is actually a bridge, second is the Sobol engine.
        self.assertEqual(sobol.model.model.seed, 9)

    def test_enforce_sequential_optimization(self):
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())
        self.assertEqual(sobol_gpei._steps[0].num_arms, 5)
        self.assertTrue(sobol_gpei._steps[0].enforce_num_arms)
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            enforce_sequential_optimization=False,
        )
        self.assertEqual(sobol_gpei._steps[0].num_arms, 5)
        self.assertFalse(sobol_gpei._steps[0].enforce_num_arms)

    def test_winsorization(self):
        winsorized = choose_generation_strategy(
            search_space=get_branin_search_space(),
            winsorize_botorch_model=True,
            winsorization_limits=(None, 0, 2),
        )
        self.assertIn(
            "Winsorize", winsorized._steps[1].model_kwargs.get("transform_configs")
        )

    def test_num_trials(self):
        ss = get_discrete_search_space()
        # Check that with budget that is lower than exhaustive, BayesOpt is used.
        sobol_gpei = choose_generation_strategy(search_space=ss, num_trials=11)
        self.assertEqual(sobol_gpei._steps[0].model.value, "Sobol")
        self.assertEqual(sobol_gpei._steps[1].model.value, "GPEI")
        # Check that with budget that is exhaustive, Sobol is used.
        sobol = choose_generation_strategy(search_space=ss, num_trials=12)
        self.assertEqual(sobol._steps[0].model.value, "Sobol")
        self.assertEqual(len(sobol._steps), 1)
