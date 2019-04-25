#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import (
    get_empirical_bayes_thompson,
    get_factorial,
    get_GPEI,
    get_MTGP,
    get_sobol,
    get_thompson,
    get_uniform,
)
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import (
    get_branin_experiment,
    get_branin_optimization_config,
    get_factorial_experiment,
    get_multi_type_experiment,
)


class ModelBridgeFactoryTest(TestCase):
    def test_sobol_GPEI(self):
        """Tests sobol + GPEI instantiation."""
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol modelbridge.
        sobol = get_sobol(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        # Check that factory generates a valid GP+EI modelbridge.
        exp.optimization_config = get_branin_optimization_config()
        gpei = get_GPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, TorchModelBridge)
        gpei = get_GPEI(
            experiment=exp, data=exp.fetch_data(), search_space=exp.search_space
        )
        self.assertIsInstance(gpei, TorchModelBridge)

    def test_MTGP(self):
        """Tests MTGP instantiation."""
        exp = get_multi_type_experiment(add_trials=True)
        mtgp = get_MTGP(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(mtgp, TorchModelBridge)

    def test_model_kwargs(self):
        """Tests that model kwargs are passed correctly."""
        exp = get_branin_experiment()
        sobol = get_sobol(
            search_space=exp.search_space, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        with self.assertRaises(TypeError):
            get_sobol(search_space=exp.search_space, nonexistent=True)

    def test_factorial(self):
        """Tests factorial instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_empirical_bayes_thompson(self):
        """Tests EB/TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        eb_thompson = get_empirical_bayes_thompson(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteModelBridge)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_thompson(self):
        """Tests TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        thompson = get_thompson(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_uniform(self):
        exp = get_branin_experiment()
        uniform = get_uniform(exp.search_space)
        self.assertIsInstance(uniform, RandomModelBridge)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)
