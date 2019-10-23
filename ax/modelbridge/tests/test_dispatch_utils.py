#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
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
