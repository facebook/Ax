#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.exceptions.generation_strategy import GenerationStrategyCompleted
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
    MaxParallelismReachedException,
)
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import MODEL_KEY_TO_MODEL_SETUP, Cont_X_trans, Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_choice_parameter,
    get_data,
)


class TestGenerationStrategy(TestCase):
    def setUp(self):
        self.gr = GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 2})])

        # Mock out slow GPEI.
        self.torch_model_bridge_patcher = patch(
            f"{TorchModelBridge.__module__}.TorchModelBridge", spec=True
        )
        self.mock_torch_model_bridge = self.torch_model_bridge_patcher.start()
        self.mock_torch_model_bridge.return_value.gen.return_value = self.gr

        # Mock out slow TS.
        self.discrete_model_bridge_patcher = patch(
            f"{DiscreteModelBridge.__module__}.DiscreteModelBridge", spec=True
        )
        self.mock_discrete_model_bridge = self.discrete_model_bridge_patcher.start()
        self.mock_discrete_model_bridge.return_value.gen.return_value = self.gr

        # Mock in `Models` registry
        self.registry_setup_dict_patcher = patch.dict(
            f"{Models.__module__}.MODEL_KEY_TO_MODEL_SETUP",
            {
                "Factorial": MODEL_KEY_TO_MODEL_SETUP["Factorial"]._replace(
                    bridge_class=self.mock_discrete_model_bridge
                ),
                "Thompson": MODEL_KEY_TO_MODEL_SETUP["Thompson"]._replace(
                    bridge_class=self.mock_discrete_model_bridge
                ),
                "GPEI": MODEL_KEY_TO_MODEL_SETUP["GPEI"]._replace(
                    bridge_class=self.mock_torch_model_bridge
                ),
            },
        )
        self.mock_in_registry = self.registry_setup_dict_patcher.start()

    def tearDown(self):
        self.torch_model_bridge_patcher.stop()
        self.discrete_model_bridge_patcher.stop()
        self.registry_setup_dict_patcher.stop()

    def test_validation(self):
        # num_trials can be positive or -1.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=5),
                    GenerationStep(model=Models.GPEI, num_trials=-10),
                ]
            )

        # only last num_trials can be -1.
        with self.assertRaises(ValueError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=-1),
                    GenerationStep(model=Models.GPEI, num_trials=10),
                ]
            )

        exp = Experiment(
            name="test", search_space=SearchSpace(parameters=[get_choice_parameter()])
        )
        factorial_thompson_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_trials=1),
                GenerationStep(model=Models.THOMPSON, num_trials=2),
            ]
        )
        self.assertTrue(factorial_thompson_generation_strategy._uses_registered_models)
        self.assertFalse(
            factorial_thompson_generation_strategy.uses_non_registered_models
        )
        with self.assertRaises(ValueError):
            factorial_thompson_generation_strategy.gen(exp)
        self.assertEqual(GenerationStep(model=sum, num_trials=1).model_name, "sum")

    def test_custom_callables_for_models(self):
        exp = get_branin_experiment()
        sobol_factory_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=get_sobol, num_trials=-1)]
        )
        self.assertFalse(sobol_factory_generation_strategy._uses_registered_models)
        self.assertTrue(sobol_factory_generation_strategy.uses_non_registered_models)
        gr = sobol_factory_generation_strategy.gen(experiment=exp, n=1)
        self.assertEqual(len(gr.arms), 1)

    def test_string_representation(self):
        gs1 = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        self.assertEqual(
            str(gs1),
            (
                "GenerationStrategy(name='Sobol+GPEI', steps=[Sobol for 5 trials,"
                " GPEI for subsequent trials])"
            ),
        )
        gs2 = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)]
        )
        self.assertEqual(
            str(gs2), "GenerationStrategy(name='Sobol', steps=[Sobol for all trials])"
        )

    def test_equality(self):
        gs1 = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ],
        )
        gs2 = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ],
        )
        self.assertEqual(gs1, gs2)

        # Clone_reset() doesn't clone exactly, so they won't be equal.
        gs3 = gs1.clone_reset()
        self.assertEqual(gs1, gs3)

    def test_restore_from_generator_run(self):
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        with self.assertRaises(ValueError):
            gs._restore_model_from_generator_run()
        gs.gen(experiment=get_branin_experiment())
        model = gs.model
        gs._restore_model_from_generator_run()
        # Model should be reset.
        self.assertIsNot(model, gs.model)

    def test_min_observed(self):
        # We should fail to transition the next model if there is not
        # enough data observed.
        exp = get_branin_experiment(get_branin_experiment())
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
                GenerationStep(model=Models.GPEI, num_trials=1),
            ]
        )
        self.assertFalse(gs.uses_non_registered_models)
        for _ in range(5):
            exp.new_trial(gs.gen(exp))
        with self.assertRaises(DataRequiredError):
            gs.gen(exp)

    def test_do_not_enforce_min_observations(self):
        # We should be able to move on to the next model if there is not
        # enough data observed if `enforce_num_trials` setting is False, in which
        # case the previous model should be used until there is enough data.
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=1,
                    min_trials_observed=5,
                    enforce_num_trials=False,
                ),
                GenerationStep(model=Models.GPEI, num_trials=1),
            ]
        )
        for _ in range(2):
            gs.gen(exp)
        # Make sure Sobol is used to generate the 6th point.
        self.assertIsInstance(gs._model, RandomModelBridge)

    def test_sobol_GPEI_strategy(self):
        exp = get_branin_experiment()
        sobol_GPEI = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=2),
            ],
        )
        self.assertEqual(sobol_GPEI.name, "Sobol+GPEI")
        self.assertEqual(sobol_GPEI.model_transitions, [5])
        # exp.new_trial(generator_run=sobol_GPEI.gen(exp)).run()
        for i in range(7):
            g = sobol_GPEI.gen(exp)
            exp.new_trial(generator_run=g).run()
            if i > 4:
                self.mock_torch_model_bridge.assert_called()
            else:
                self.assertEqual(g._model_key, "Sobol")
                self.assertEqual(
                    g._model_kwargs,
                    {
                        "seed": None,
                        "deduplicate": False,
                        "init_position": i,
                        "scramble": True,
                        "generated_points": None,
                    },
                )
                self.assertEqual(
                    g._bridge_kwargs,
                    {
                        "optimization_config": None,
                        "status_quo_features": None,
                        "status_quo_name": None,
                        "transform_configs": None,
                        "transforms": Cont_X_trans,
                        "fit_out_of_design": False,
                    },
                )
                self.assertEqual(g._model_state_after_gen, {"init_position": i + 1})
        # Check completeness error message when GS should be done.
        with self.assertRaises(GenerationStrategyCompleted):
            g = sobol_GPEI.gen(exp)

    def test_sobol_GPEI_strategy_keep_generating(self):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "Sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.model_transitions, [5])
        exp.new_trial(generator_run=sobol_GPEI_generation_strategy.gen(exp)).run()
        for i in range(1, 15):
            g = sobol_GPEI_generation_strategy.gen(exp)
            exp.new_trial(generator_run=g).run()
            if i > 4:
                self.assertIsInstance(
                    sobol_GPEI_generation_strategy.model, TorchModelBridge
                )

    @patch(f"{Experiment.__module__}.Experiment.fetch_data", return_value=get_data())
    def test_factorial_thompson_strategy(self, _):
        exp = get_branin_experiment()
        factorial_thompson_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_trials=1),
                GenerationStep(model=Models.THOMPSON, num_trials=-1),
            ]
        )
        self.assertEqual(
            factorial_thompson_generation_strategy.name, "Factorial+Thompson"
        )
        self.assertEqual(factorial_thompson_generation_strategy.model_transitions, [1])
        mock_model_bridge = self.mock_discrete_model_bridge.return_value

        # Initial factorial batch.
        exp.new_batch_trial(factorial_thompson_generation_strategy.gen(experiment=exp))
        args, kwargs = mock_model_bridge._set_kwargs_to_save.call_args
        self.assertEqual(kwargs.get("model_key"), "Factorial")

        # Subsequent Thompson sampling batch.
        exp.new_batch_trial(factorial_thompson_generation_strategy.gen(experiment=exp))
        args, kwargs = mock_model_bridge._set_kwargs_to_save.call_args
        self.assertEqual(kwargs.get("model_key"), "Thompson")

    def test_clone_reset(self):
        ftgs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_trials=1),
                GenerationStep(model=Models.THOMPSON, num_trials=2),
            ]
        )
        ftgs._curr = ftgs._steps[1]
        self.assertEqual(ftgs._curr.index, 1)
        self.assertEqual(ftgs.clone_reset()._curr.index, 0)

    def test_kwargs_passed(self):
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=1, model_kwargs={"scramble": False}
                )
            ]
        )
        exp = get_branin_experiment()
        gs.gen(exp)
        self.assertFalse(gs._model.model.scramble)

    def test_sobol_GPEI_strategy_batches(self):
        mock_GPEI_gen = self.mock_torch_model_bridge.return_value.gen
        mock_GPEI_gen.return_value = GeneratorRun(
            arms=[
                Arm(parameters={"x1": 1, "x2": 2}),
                Arm(parameters={"x1": 3, "x2": 4}),
            ]
        )
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=1),
                GenerationStep(model=Models.GPEI, num_trials=6),
            ],
        )
        self.assertEqual(sobol_GPEI_generation_strategy.name, "Sobol+GPEI")
        self.assertEqual(sobol_GPEI_generation_strategy.model_transitions, [1])
        gr = sobol_GPEI_generation_strategy.gen(exp, n=2)
        exp.new_batch_trial(generator_run=gr).run()
        for i in range(1, 8):
            if i == 7:
                # Check completeness error message.
                with self.assertRaises(GenerationStrategyCompleted):
                    g = sobol_GPEI_generation_strategy.gen(exp, n=2)
            else:
                g = sobol_GPEI_generation_strategy.gen(exp, n=2)
            exp.new_batch_trial(generator_run=g).run()
        self.assertIsInstance(sobol_GPEI_generation_strategy.model, TorchModelBridge)

    def test_with_factory_function(self):
        """Checks that generation strategy works with custom factory functions.
        No information about the model should be saved on generator run."""

        def get_sobol(search_space: SearchSpace) -> RandomModelBridge:
            return RandomModelBridge(
                search_space=search_space,
                model=SobolGenerator(),
                transforms=Cont_X_trans,
            )

        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=get_sobol, num_trials=5)]
        )
        g = sobol_generation_strategy.gen(exp)
        self.assertIsInstance(sobol_generation_strategy.model, RandomModelBridge)
        self.assertIsNone(g._model_key)
        self.assertIsNone(g._model_kwargs)
        self.assertIsNone(g._bridge_kwargs)

    def test_store_experiment(self):
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5)]
        )
        self.assertIsNone(sobol_generation_strategy._experiment)
        sobol_generation_strategy.gen(exp)
        self.assertIsNotNone(sobol_generation_strategy._experiment)

    def test_trials_as_df(self):
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5)]
        )
        # No trials yet, so the DF will be None.
        self.assertIsNone(sobol_generation_strategy.trials_as_df)
        # Now the trial should appear in the DF.
        trial = exp.new_trial(sobol_generation_strategy.gen(experiment=exp))
        self.assertFalse(sobol_generation_strategy.trials_as_df.empty)
        self.assertEqual(
            sobol_generation_strategy.trials_as_df.head()["Trial Status"][0],
            "CANDIDATE",
        )
        # Changes in trial status should be reflected in the DF.
        trial._status = TrialStatus.RUNNING
        self.assertEqual(
            sobol_generation_strategy.trials_as_df.head()["Trial Status"][0], "RUNNING"
        )

    def test_max_parallelism_reached(self):
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5, max_parallelism=1)]
        )
        exp.new_trial(
            generator_run=sobol_generation_strategy.gen(experiment=exp)
        ).mark_running(no_runner_required=True)
        with self.assertRaises(MaxParallelismReachedException):
            sobol_generation_strategy.gen(experiment=exp)

    @patch(f"{RandomModelBridge.__module__}.RandomModelBridge.update")
    @patch(f"{Experiment.__module__}.Experiment.fetch_trials_data")
    def test_use_update(self, mock_fetch_trials_data, mock_update):
        exp = get_branin_experiment()
        sobol_gs_with_update = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1, use_update=True)]
        )
        # Try without passing data (generation strategy fetches data from experiment).
        trial = exp.new_trial(generator_run=sobol_gs_with_update.gen(experiment=exp))
        mock_update.assert_not_called()
        trial._status = TrialStatus.COMPLETED
        for i in range(3):
            trial = exp.new_trial(
                generator_run=sobol_gs_with_update.gen(experiment=exp)
            )
            self.assertEqual(
                mock_fetch_trials_data.call_args[1].get("trial_indices"), {i}
            )
            trial._status = TrialStatus.COMPLETED
        # Try with passing data.
        sobol_gs_with_update.gen(
            experiment=exp, data=get_branin_data(trial_indices=range(4))
        )
        # Only the data for the last completed trial should be considered new and passed
        # to `update`.
        self.assertEqual(
            set(mock_update.call_args[1].get("new_data").df["trial_index"].values), {3}
        )
