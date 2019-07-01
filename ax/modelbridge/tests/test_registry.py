#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import Cont_X_trans, Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import (
    get_branin_experiment,
    get_branin_optimization_config,
    get_factorial_experiment,
)


class ModelRegistryTest(TestCase):
    def test_enum_sobol_GPEI(self):
        """Tests Sobol instantiation through the Models enum."""
        exp = get_branin_experiment()
        # Check that factory generates a valid sobol modelbridge.
        sobol = Models.SOBOL(search_space=exp.search_space)
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(n=1)
            self.assertEqual(sobol_run._model_key, "Sobol")
            exp.new_batch_trial().add_generator_run(sobol_run).run()
        # Check that factory generates a valid GP+EI modelbridge.
        exp.optimization_config = get_branin_optimization_config()
        gpei = Models.GPEI(experiment=exp, data=exp.fetch_data())
        self.assertIsInstance(gpei, TorchModelBridge)
        gpei = Models.GPEI(
            experiment=exp, data=exp.fetch_data(), search_space=exp.search_space
        )
        self.assertIsInstance(gpei, TorchModelBridge)

    def test_enum_model_kwargs(self):
        """Tests that kwargs are passed correctly when instantiating through the
        Models enum."""
        exp = get_branin_experiment()
        sobol = Models.SOBOL(
            search_space=exp.search_space, init_position=2, scramble=False, seed=239
        )
        self.assertIsInstance(sobol, RandomModelBridge)
        for _ in range(5):
            sobol_run = sobol.gen(1)
            exp.new_batch_trial().add_generator_run(sobol_run).run()

    def test_enum_factorial(self):
        """Tests factorial instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        self.assertEqual(len(factorial_run.arms), 24)

    def test_enum_empirical_bayes_thompson(self):
        """Tests EB/TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        eb_thompson = Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=exp, data=data, min_weight=0.0
        )
        self.assertIsInstance(eb_thompson, DiscreteModelBridge)
        self.assertIsInstance(eb_thompson.model, EmpiricalBayesThompsonSampler)
        thompson_run = eb_thompson.gen(n=5)
        self.assertEqual(len(thompson_run.arms), 5)

    def test_enum_thompson(self):
        """Tests TS instantiation through the Models enum."""
        exp = get_factorial_experiment()
        factorial = Models.FACTORIAL(exp.search_space)
        self.assertIsInstance(factorial, DiscreteModelBridge)
        factorial_run = factorial.gen(n=-1)
        exp.new_batch_trial().add_generator_run(factorial_run).run()
        data = exp.fetch_data()
        thompson = Models.THOMPSON(experiment=exp, data=data)
        self.assertIsInstance(thompson.model, ThompsonSampler)

    def test_enum_uniform(self):
        """Tests uniform random instantiation through the Models enum."""
        exp = get_branin_experiment()
        uniform = Models.UNIFORM(exp.search_space)
        self.assertIsInstance(uniform, RandomModelBridge)
        uniform_run = uniform.gen(n=5)
        self.assertEqual(len(uniform_run.arms), 5)

    def test_view_defaults(self):
        """Checks that kwargs are correctly constructed from default kwargs +
        standard kwargs."""
        self.assertEqual(
            Models.SOBOL.view_defaults(),
            (
                {
                    "seed": None,
                    "deduplicate": False,
                    "init_position": 0,
                    "scramble": True,
                },
                {
                    "optimization_config": None,
                    "transforms": Cont_X_trans,
                    "transform_configs": None,
                    "status_quo_name": None,
                    "status_quo_features": None,
                },
            ),
        )
        self.assertTrue(
            all(
                kw in Models.SOBOL.view_kwargs()[0]
                for kw in ["seed", "deduplicate", "init_position", "scramble"]
            ),
            all(
                kw in Models.SOBOL.view_kwargs()[1]
                for kw in [
                    "search_space",
                    "model",
                    "transforms",
                    "experiment",
                    "data",
                    "transform_configs",
                    "status_quo_name",
                    "status_quo_features",
                ]
            ),
        )
