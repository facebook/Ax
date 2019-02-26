#!/usr/bin/env python3

from ae.lazarus.ae.generator.discrete import DiscreteGenerator
from ae.lazarus.ae.generator.factory import (
    get_empirical_bayes_thompson,
    get_factorial,
    get_GPEI,
    get_GPyGPEI,
    get_sobol,
    get_thompson,
    get_uniform,
)
from ae.lazarus.ae.generator.numpy import NumpyGenerator
from ae.lazarus.ae.generator.random import RandomGenerator
from ae.lazarus.ae.generator.torch import TorchGenerator
from ae.lazarus.ae.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ae.lazarus.ae.models.discrete.thompson import ThompsonSampler
from ae.lazarus.ae.tests.fake import (
    get_branin_experiment,
    get_branin_optimization_config,
    get_factorial_experiment,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class GeneratorFactoryTest(TestCase):
    def test_sobol_GPEI(self):
        """Tests sobol + GPEI instantiation."""
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol generator.
        sobol = get_sobol(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomGenerator)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        # Check that factory generates a valid GP+EI generator.
        exp.optimization_config = get_branin_optimization_config()
        gpei = get_GPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, TorchGenerator)
        gpei = get_GPEI(
            experiment=exp, data=exp.fetch_data(), search_space=exp.search_space
        )
        self.assertIsInstance(gpei, TorchGenerator)
        gpei = get_GPyGPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, NumpyGenerator)
        gpei_run = gpei.gen(n=2)
        self.assertEqual(len(gpei_run.arms), 2)

    def test_model_kwargs(self):
        """Tests that model kwargs are passed correctly."""
        exp = get_branin_experiment()
        sobol = get_sobol(
            search_space=exp.search_space, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomGenerator)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        with self.assertRaises(TypeError):
            get_sobol(search_space=exp.search_space, nonexistent=True)
        gpei = get_GPyGPEI(experiment=exp, data=exp.fetch_data(), refit_on_cv=True)
        self.assertIsInstance(gpei, NumpyGenerator)
        gpei_run = gpei.gen(2)
        self.assertEqual(len(gpei_run.arms), 2)
        with self.assertRaises(TypeError):
            get_GPyGPEI(experiment=exp, data=exp.fetch_data(), nonexistent=True)
        with self.assertRaises(TypeError):
            get_GPEI(experiment=exp, data=exp.fetch_data(), nonexistent=True)

    def test_factorial(self):
        """Tests factorial instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteGenerator)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_empirical_bayes_thompson(self):
        """Tests EB/TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteGenerator)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        eb_thompson = get_empirical_bayes_thompson(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteGenerator)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_thompson(self):
        """Tests TS instantiation."""
        exp = get_factorial_experiment()
        factorial = get_factorial(exp.search_space)
        self.assertIsInstance(factorial, DiscreteGenerator)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        thompson = get_thompson(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_uniform(self):
        exp = get_branin_experiment()
        uniform = get_uniform(exp.search_space)
        self.assertIsInstance(uniform, RandomGenerator)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)
