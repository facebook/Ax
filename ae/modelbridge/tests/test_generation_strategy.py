#!/usr/bin/env python3
from unittest import mock

from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.modelbridge.discrete import DiscreteModelBridge
from ae.lazarus.ae.modelbridge.factory import (
    get_factorial,
    get_GPEI,
    get_sobol,
    get_thompson,
)
from ae.lazarus.ae.modelbridge.generation_strategy import GenerationStrategy
from ae.lazarus.ae.modelbridge.random import RandomModelBridge
from ae.lazarus.ae.modelbridge.torch import TorchModelBridge
from ae.lazarus.ae.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ae.lazarus.ae.models.discrete.full_factorial import FullFactorialGenerator
from ae.lazarus.ae.models.discrete.thompson import ThompsonSampler
from ae.lazarus.ae.tests.fake import get_branin_experiment
from ae.lazarus.ae.utils.common.testutils import TestCase


class TestGenerationStrategy(TestCase):
    def test_validation(self):
        # GenerationStrategy should require as many arms_per_model
        # as model_factories.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                model_factories=[get_sobol, get_sobol, get_sobol], arms_per_model=[5]
            )

        # arms_per_model can't be negative.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                model_factories=[get_sobol, get_GPEI], arms_per_model=[5, -1]
            )

        # GenerationStrategy should require that there be no more
        # arms_per_model than there are model factories.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                model_factories=[get_sobol, get_sobol], arms_per_model=[5, 5, 5]
            )

        # Must specify arms_per_model
        with self.assertRaises(TypeError):
            GenerationStrategy(model_factories=[get_sobol, get_sobol])

    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{RandomModelBridge.__module__}.RandomModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    def test_sobol_GPEI_strategy(self, mock_sobol, mock_GPEI):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            model_factories=[get_sobol, get_GPEI], arms_per_model=[5, 2]
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.generator_changes, [5])
        prev_g, g = None, None
        for i in range(7):
            prev_g = g
            g = sobol_GPEI_generation_strategy.get_model(
                exp, exp.fetch_data(), exp.search_space
            )
            exp.new_batch_trial().add_arm(Arm(params={"x1": i, "x2": i}))
            if i > 0 and i < 5:
                self.assertTrue(g is prev_g)
            else:
                self.assertFalse(g is prev_g)
            if i < 5:
                mock_sobol.assert_called()
                exp.new_batch_trial()
            else:
                mock_GPEI.assert_called()
        with self.assertRaises(ValueError):
            sobol_GPEI_generation_strategy.get_model(
                exp, exp.fetch_data(), exp.search_space
            )

    @mock.patch(
        f"{DiscreteModelBridge.__module__}.DiscreteModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    def test_factorial_thompson_strategy(self, mock_discrete):
        exp = get_branin_experiment()
        factorial_thompson_generation_strategy = GenerationStrategy(
            model_factories=[get_factorial, get_thompson], arms_per_model=[1, 2]
        )
        self.assertEqual(
            factorial_thompson_generation_strategy.name, "factorial+thompson"
        )
        self.assertEqual(factorial_thompson_generation_strategy.generator_changes, [1])
        for i in range(3):
            factorial_thompson_generation_strategy.get_model(
                exp, exp.fetch_data(), exp.search_space
            )
            exp.new_batch_trial().add_arm(Arm(params={"x1": i, "x2": i}))
            if i < 1:
                mock_discrete.assert_called()
                args, kwargs = mock_discrete.call_args
                self.assertIsInstance(kwargs.get("model"), FullFactorialGenerator)
                exp.new_batch_trial()
            else:
                mock_discrete.assert_called()
                args, kwargs = mock_discrete.call_args
                self.assertIsInstance(
                    kwargs.get("model"),
                    (ThompsonSampler, EmpiricalBayesThompsonSampler),
                )
