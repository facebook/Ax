#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from unittest import mock

from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.search_space import SearchSpace
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.torch import TorchModelBridge
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import get_branin_experiment, get_choice_parameter, get_data


class TestGenerationStrategy(TestCase):
    def test_validation(self):
        # num_arms can be positive or -1.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_arms=5),
                    GenerationStep(model=Models.GPEI, num_arms=-10),
                ]
            )

        # only last num_arms can be -1.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_arms=-1),
                    GenerationStep(model=Models.GPEI, num_arms=10),
                ]
            )

        exp = Experiment(
            name="test", search_space=SearchSpace(parameters=[get_choice_parameter()])
        )
        factorial_thompson_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_arms=1),
                GenerationStep(model=Models.THOMPSON, num_arms=2),
            ]
        )
        with self.assertRaises(ValueError):
            factorial_thompson_generation_strategy.gen(exp)

    def test_min_observed(self):
        # We should fail to transition the next model if there is not
        # enough data observed.
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_arms=5, min_arms_observed=5),
                GenerationStep(model=Models.GPEI, num_arms=1),
            ]
        )
        for _ in range(5):
            gs.gen(exp)
        with self.assertRaises(ValueError):
            gs.gen(exp)

    def test_do_not_enforce_min_observations(self):
        # We should be able to move on to the next model if there is not
        # enough data observed if `enforce_num_arms` setting is False, in which
        # case the previous model should be used until there is enough data.
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_arms=5,
                    min_arms_observed=5,
                    enforce_num_arms=False,
                ),
                GenerationStep(model=Models.GPEI, num_arms=1),
            ]
        )
        for _ in range(5):
            gs.gen(exp)
            sobol = gs._model
        gs.gen(exp)
        # Make sure the same model is used to generate the 6th point.
        self.assertIs(gs._model, sobol)

    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.update",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.gen",
        autospec=True,
        return_value=GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 2})]),
    )
    def test_sobol_GPEI_strategy(self, mock_GPEI_gen, mock_GPEI_update, mock_GPEI_init):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_arms=5),
                GenerationStep(model=Models.GPEI, num_arms=2),
            ],
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "Sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.generator_changes, [5])
        exp.new_trial(generator_run=sobol_GPEI_generation_strategy.gen(exp)).run()
        for i in range(1, 8):
            if i == 7:
                # Check completeness error message.
                with self.assertRaisesRegex(ValueError, "Generation strategy"):
                    g = sobol_GPEI_generation_strategy.gen(
                        exp, exp._fetch_trial_data(trial_index=i - 1)
                    )
            else:
                g = sobol_GPEI_generation_strategy.gen(
                    exp, exp._fetch_trial_data(trial_index=i - 1)
                )
                exp.new_trial(generator_run=g).run()
                if i > 4:
                    mock_GPEI_init.assert_called()
        # Check for "seen data" error message.
        with self.assertRaisesRegex(ValueError, "Data for arm"):
            sobol_GPEI_generation_strategy.gen(exp, exp.fetch_data())

    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.update",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.gen",
        autospec=True,
        return_value=GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 2})]),
    )
    def test_sobol_GPEI_strategy_keep_generating(
        self, mock_GPEI_gen, mock_GPEI_update, mock_GPEI_init
    ):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_arms=5),
                GenerationStep(model=Models.GPEI, num_arms=-1),
            ]
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.generator_changes, [5])
        exp.new_trial(generator_run=sobol_GPEI_generation_strategy.gen(exp)).run()
        for i in range(1, 15):
            # Passing in all experiment data should cause an error as only
            # new data should be passed into `gen`.
            if i > 1:
                with self.assertRaisesRegex(ValueError, "Data for arm"):
                    g = sobol_GPEI_generation_strategy.gen(exp, exp.fetch_data())
            g = sobol_GPEI_generation_strategy.gen(
                exp, exp._fetch_trial_data(trial_index=i - 1)
            )
            exp.new_trial(generator_run=g).run()
            if i > 4:
                mock_GPEI_init.assert_called()

    @mock.patch(
        f"{DiscreteModelBridge.__module__}.DiscreteModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{DiscreteModelBridge.__module__}.DiscreteModelBridge.gen",
        autospec=True,
        return_value=GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 2})]),
    )
    @mock.patch(
        f"{DiscreteModelBridge.__module__}.DiscreteModelBridge.update",
        autospec=True,
        return_value=None,
    )
    def test_factorial_thompson_strategy(self, mock_update, mock_gen, mock_discrete):
        exp = get_branin_experiment()
        factorial_thompson_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_arms=1),
                GenerationStep(model=Models.THOMPSON, num_arms=-1),
            ]
        )
        self.assertEqual(
            factorial_thompson_generation_strategy.name, "factorial+thompson"
        )
        self.assertEqual(factorial_thompson_generation_strategy.generator_changes, [1])
        for i in range(2):
            data = get_data() if i > 0 else None
            factorial_thompson_generation_strategy.gen(experiment=exp, new_data=data)
            exp.new_batch_trial().add_arm(Arm(parameters={"x1": i, "x2": i}))
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

    def test_clone_reset(self):
        ftgs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_arms=1),
                GenerationStep(model=Models.THOMPSON, num_arms=2),
            ]
        )
        ftgs._curr = ftgs._steps[1]
        self.assertEqual(ftgs._curr.index, 1)
        self.assertEqual(ftgs.clone_reset()._curr.index, 0)

    def test_kwargs_passed(self):
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_arms=1, model_kwargs={"scramble": False}
                )
            ]
        )
        exp = get_branin_experiment()
        gs.gen(exp, exp.fetch_data())
        self.assertFalse(gs._model.model.scramble)

    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.update",
        autospec=True,
        return_value=None,
    )
    @mock.patch(
        f"{TorchModelBridge.__module__}.TorchModelBridge.gen",
        autospec=True,
        return_value=GeneratorRun(
            arms=[
                Arm(parameters={"x1": 1, "x2": 2}),
                Arm(parameters={"x1": 3, "x2": 4}),
            ]
        ),
    )
    def test_sobol_GPEI_strategy_batches(
        self, mock_GPEI_gen, mock_GPEI_update, mock_GPEI_init
    ):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_arms=5),
                GenerationStep(model=Models.GPEI, num_arms=8),
            ],
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "Sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.generator_changes, [5])
        exp.new_batch_trial(
            generator_run=sobol_GPEI_generation_strategy.gen(exp, n=2)
        ).run()
        for i in range(1, 8):
            if i == 2:
                with self.assertRaisesRegex(ValueError, "Cannot generate 2 new"):
                    g = sobol_GPEI_generation_strategy.gen(
                        exp, exp._fetch_trial_data(trial_index=i - 1), n=2
                    )
                g = sobol_GPEI_generation_strategy.gen(
                    exp, exp._fetch_trial_data(trial_index=i - 1)
                )
            elif i == 7:
                # Check completeness error message.
                with self.assertRaisesRegex(ValueError, "Generation strategy"):
                    g = sobol_GPEI_generation_strategy.gen(
                        exp, exp._fetch_trial_data(trial_index=i - 1), n=2
                    )
            else:
                g = sobol_GPEI_generation_strategy.gen(
                    exp, exp._fetch_trial_data(trial_index=i - 1), n=2
                )
            exp.new_batch_trial(generator_run=g).run()
            if i > 4:
                mock_GPEI_init.assert_called()
        with self.assertRaises(ValueError):
            sobol_GPEI_generation_strategy.gen(exp, exp.fetch_data())
