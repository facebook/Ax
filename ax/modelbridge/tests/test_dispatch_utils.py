#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.dispatch_utils import (
    DEFAULT_BAYESIAN_PARALLELISM,
    choose_generation_strategy,
)
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
        self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
        self.assertEqual(sobol_gpei._steps[1].model.value, "GPEI")
        sobol = choose_generation_strategy(search_space=get_factorial_search_space())
        self.assertEqual(sobol._steps[0].model.value, "Sobol")
        self.assertEqual(len(sobol._steps), 1)
        sobol_gpei_batched = choose_generation_strategy(
            search_space=get_branin_search_space(), use_batch_trials=3
        )
        self.assertEqual(sobol_gpei_batched._steps[0].num_trials, 1)

    def test_setting_random_seed(self):
        sobol = choose_generation_strategy(
            search_space=get_factorial_search_space(), random_seed=9
        )
        sobol.gen(experiment=get_experiment())
        # First model is actually a bridge, second is the Sobol engine.
        self.assertEqual(sobol.model.model.seed, 9)

    def test_enforce_sequential_optimization(self):
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())
        self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
        self.assertTrue(sobol_gpei._steps[0].enforce_num_trials)
        self.assertIsNotNone(sobol_gpei._steps[1].max_parallelism)
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            enforce_sequential_optimization=False,
        )
        self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
        self.assertFalse(sobol_gpei._steps[0].enforce_num_trials)
        self.assertIsNone(sobol_gpei._steps[1].max_parallelism)

    def test_max_parallelism_override(self):
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertTrue(all(s.max_parallelism == 10 for s in sobol_gpei._steps))

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

    def test_use_batch_trials(self):
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), use_batch_trials=True
        )
        self.assertEqual(sobol_gpei._steps[0].num_trials, 1)

    def test_fixed_num_initialization_trials(self):
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
        )
        self.assertEqual(sobol_gpei._steps[0].num_trials, 3)

    def test_max_parallelism_adjustments(self):
        # No adjustment.
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())
        self.assertIsNone(sobol_gpei._steps[0].max_parallelism)
        self.assertEqual(
            sobol_gpei._steps[1].max_parallelism, DEFAULT_BAYESIAN_PARALLELISM
        )
        # Impose a cap of 1 on max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_cap=1
        )
        self.assertEqual(
            sobol_gpei._steps[0].max_parallelism,
            sobol_gpei._steps[1].max_parallelism,
            1,
        )
        # Disable enforcing max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=-1
        )
        self.assertIsNone(sobol_gpei._steps[0].max_parallelism)
        self.assertIsNone(sobol_gpei._steps[1].max_parallelism)
        # Override max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertEqual(sobol_gpei._steps[0].max_parallelism, 10)
        self.assertEqual(sobol_gpei._steps[1].max_parallelism, 10)
