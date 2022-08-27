#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List
from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.parameter import ChoiceParameter, FixedParameter, Parameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.exceptions.generation_strategy import (
    GenerationStrategyCompleted,
    GenerationStrategyRepeatedPoints,
)
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
    MaxParallelismReachedException,
)
from ax.modelbridge.modelbridge_utils import (
    get_pending_observation_features_based_on_trial_status as get_pending,
)
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import Cont_X_trans, MODEL_KEY_TO_MODEL_SETUP, Models
from ax.modelbridge.torch import TorchModelBridge
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_choice_parameter,
    get_data,
    get_hierarchical_search_space_experiment,
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

        # model bridges are mocked, which makes kwargs' validation difficult,
        # so for now we will skip it in the generation strategy tests.
        # NOTE: Starting with Python3.8 this is not a problem as `autospec=True`
        # ensures that the mocks have correct signatures, but in earlier
        # versions kwarg validation on mocks does not really work.
        self.step_model_kwargs = {"silently_filter_kwargs": True}
        self.hss_experiment = get_hierarchical_search_space_experiment()
        self.sobol_GPEI_GS = GenerationStrategy(
            name="Sobol+GPEI",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.GPEI, num_trials=2, model_kwargs=self.step_model_kwargs
                ),
            ],
        )
        self.sobol_GS = GenerationStrategy(
            steps=[
                GenerationStep(
                    Models.SOBOL,
                    num_trials=-1,
                    should_deduplicate=True,
                )
            ]
        )

    def tearDown(self):
        self.torch_model_bridge_patcher.stop()
        self.discrete_model_bridge_patcher.stop()
        self.registry_setup_dict_patcher.stop()

    def test_name(self):
        self.sobol_GS.name = "SomeGSName"
        self.assertEqual(self.sobol_GS.name, "SomeGSName")

    def test_validation(self):
        # num_trials can be positive or -1.
        with self.assertRaises(UserInputError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=5),
                    GenerationStep(model=Models.GPEI, num_trials=-10),
                ]
            )

        # only last num_trials can be -1.
        with self.assertRaises(UserInputError):
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
        with self.assertRaisesRegex(UserInputError, "Maximum parallelism should be"):
            GenerationStrategy(
                steps=[
                    GenerationStep(
                        model=Models.SOBOL, num_trials=5, max_parallelism=-1
                    ),
                    GenerationStep(model=Models.GPEI, num_trials=-1),
                ]
            )

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
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5)]
        )
        # No generator runs on GS, so can't restore from one.
        with self.assertRaises(ValueError):
            gs._restore_model_from_generator_run()
        exp = get_branin_experiment(with_batch=True)
        gs.gen(experiment=exp)
        model = gs.model
        # Create a copy of the generation strategy and check that when
        # we restore from last generator run, the model will be set
        # correctly and that `_seen_trial_indices_by_status` is filled.
        new_gs = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5)]
        )
        new_gs._experiment = exp
        new_gs._generator_runs = gs._generator_runs
        self.assertIsNone(new_gs._seen_trial_indices_by_status)
        new_gs._restore_model_from_generator_run()
        self.assertEqual(gs._seen_trial_indices_by_status, exp.trial_indices_by_status)
        # Model should be reset, but it should be the same model with same data.
        self.assertIsNot(model, new_gs.model)
        self.assertEqual(model.__class__, new_gs.model.__class__)  # Model bridge.
        self.assertEqual(model.model.__class__, new_gs.model.model.__class__)  # Model.
        self.assertEqual(model._training_data, new_gs.model._training_data)

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
        self.assertEqual(self.sobol_GPEI_GS.name, "Sobol+GPEI")
        self.assertEqual(self.sobol_GPEI_GS.model_transitions, [5])
        for i in range(7):
            g = self.sobol_GPEI_GS.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_GPEI_GS._generator_runs), i + 1)
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
                        "fallback_to_sample_polytope": False,
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
                        "fit_abandoned": False,
                    },
                )
                self.assertEqual(g._model_state_after_gen, {"init_position": i + 1})
        # Check completeness error message when GS should be done.
        with self.assertRaises(GenerationStrategyCompleted):
            g = self.sobol_GPEI_GS.gen(exp)

    def test_sobol_GPEI_strategy_keep_generating(self):
        exp = get_branin_experiment()
        sobol_GPEI_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=-1,
                    model_kwargs=self.step_model_kwargs,
                ),
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

    def test_sobol_strategy(self):
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,
                    max_parallelism=10,
                    use_update=False,
                    enforce_num_trials=False,
                )
            ]
        )
        for i in range(1, 6):
            sobol_generation_strategy.gen(exp, n=1)
            self.assertEqual(len(sobol_generation_strategy._generator_runs), i)

    @patch(f"{Experiment.__module__}.Experiment.fetch_data", return_value=get_data())
    def test_factorial_thompson_strategy(self, _):
        exp = get_branin_experiment()
        factorial_thompson_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.FACTORIAL,
                    num_trials=1,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.THOMPSON,
                    num_trials=-1,
                    model_kwargs=self.step_model_kwargs,
                ),
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
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=1,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.GPEI, num_trials=6, model_kwargs=self.step_model_kwargs
                ),
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
    @patch(f"{Experiment.__module__}.Experiment.lookup_data")
    def test_use_update(self, mock_lookup_data, mock_update):
        exp = get_branin_experiment()
        sobol_gs_with_update = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1, use_update=True)]
        )
        sobol_gs_with_update._experiment = exp
        self.assertEqual(
            sobol_gs_with_update._find_trials_completed_since_last_gen(),
            set(),
        )
        with self.assertRaises(NotImplementedError):
            # `BraninMetric` is available while running by default, which should
            # raise an error when use with `use_update=True` on a generation step, as we
            # have not yet properly addressed that edge case (for lack of use case).
            sobol_gs_with_update.gen(experiment=exp)

        core_stubs_module = get_branin_experiment.__module__
        with patch(
            f"{core_stubs_module}.BraninMetric.is_available_while_running",
            return_value=False,
        ):
            # Try without passing data (GS looks up data on experiment).
            trial = exp.new_trial(
                generator_run=sobol_gs_with_update.gen(experiment=exp)
            )
            mock_update.assert_not_called()
            trial._status = TrialStatus.COMPLETED
            for i in range(3):
                gr = sobol_gs_with_update.gen(experiment=exp)
                self.assertEqual(
                    mock_lookup_data.call_args[1].get("trial_indices"), {i}
                )
                trial = exp.new_trial(generator_run=gr)
                trial._status = TrialStatus.COMPLETED
            # `_seen_trial_indices_by_status` is set during `gen`, to the experiment's
            # `trial_indices_by_Status` at the time of candidate generation.
            self.assertNotEqual(
                sobol_gs_with_update._seen_trial_indices_by_status,
                exp.trial_indices_by_status,
            )
            # Try with passing data.
            sobol_gs_with_update.gen(
                experiment=exp, data=get_branin_data(trial_indices=range(4))
            )
        # Now `_seen_trial_indices_by_status` should be set to experiment's,
        self.assertEqual(
            sobol_gs_with_update._seen_trial_indices_by_status,
            exp.trial_indices_by_status,
        )
        # Only the data for the last completed trial should be considered new and passed
        # to `update`.
        self.assertEqual(
            set(mock_update.call_args[1].get("new_data").df["trial_index"].values), {3}
        )
        # Try with passing same data as before; no update should be performed.
        with patch.object(sobol_gs_with_update, "_update_current_model") as mock_update:
            sobol_gs_with_update.gen(
                experiment=exp, data=get_branin_data(trial_indices=range(4))
            )
            mock_update.assert_not_called()

    def test_deduplication(self):
        tiny_parameters = [
            FixedParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                value=1.0,
            ),
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                values=[float(x) for x in range(2)],
            ),
        ]
        tiny_search_space = SearchSpace(
            parameters=cast(List[Parameter], tiny_parameters)
        )
        exp = get_branin_experiment(search_space=tiny_search_space)
        sobol = GenerationStrategy(
            name="Sobol",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    model_kwargs=self.step_model_kwargs,
                    should_deduplicate=True,
                ),
            ],
        )
        for _ in range(2):
            g = sobol.gen(exp)
            exp.new_trial(generator_run=g).run()

        self.assertEqual(len(exp.arms_by_signature), 2)

        with self.assertRaisesRegex(
            GenerationStrategyRepeatedPoints, "exceeded `MAX_GEN_DRAWS`"
        ):
            g = sobol.gen(exp)

    def test_current_generator_run_limit(self):
        NUM_INIT_TRIALS = 5
        SECOND_STEP_PARALLELISM = 3
        NUM_ROUNDS = 4
        exp = get_branin_experiment()
        sobol_gs_with_parallelism_limits = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=NUM_INIT_TRIALS,
                    min_trials_observed=3,
                ),
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=(NUM_ROUNDS - 1) * SECOND_STEP_PARALLELISM,
                    max_parallelism=SECOND_STEP_PARALLELISM,
                ),
            ]
        )
        sobol_gs_with_parallelism_limits._experiment = exp
        could_gen = self._run_GS_for_N_rounds(
            gs=sobol_gs_with_parallelism_limits, exp=exp, num_rounds=NUM_ROUNDS
        )

        # Optimization should now be complete.
        (
            num_trials_to_gen,
            opt_complete,
        ) = sobol_gs_with_parallelism_limits.current_generator_run_limit()
        self.assertTrue(opt_complete)
        self.assertEqual(num_trials_to_gen, 0)

        # We expect trials from first generation step + trials from remaining rounds in
        # batches limited by parallelism setting in the second step.
        self.assertEqual(
            len(exp.trials),
            NUM_INIT_TRIALS + (NUM_ROUNDS - 1) * SECOND_STEP_PARALLELISM,
        )
        self.assertTrue(all(t.status.is_completed for t in exp.trials.values()))
        self.assertEqual(
            could_gen, [NUM_INIT_TRIALS] + [SECOND_STEP_PARALLELISM] * (NUM_ROUNDS - 1)
        )

    def test_current_generator_run_limit_unlimited_second_step(self):
        NUM_INIT_TRIALS = 5
        SECOND_STEP_PARALLELISM = 3
        NUM_ROUNDS = 4
        exp = get_branin_experiment()
        sobol_gs_with_parallelism_limits = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=NUM_INIT_TRIALS,
                    min_trials_observed=3,
                ),
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    max_parallelism=SECOND_STEP_PARALLELISM,
                ),
            ]
        )
        sobol_gs_with_parallelism_limits._experiment = exp
        could_gen = self._run_GS_for_N_rounds(
            gs=sobol_gs_with_parallelism_limits, exp=exp, num_rounds=NUM_ROUNDS
        )
        # We expect trials from first generation step + trials from remaining rounds in
        # batches limited by parallelism setting in the second step.
        self.assertEqual(
            len(exp.trials),
            NUM_INIT_TRIALS + (NUM_ROUNDS - 1) * SECOND_STEP_PARALLELISM,
        )
        self.assertTrue(all(t.status.is_completed for t in exp.trials.values()))
        self.assertEqual(
            could_gen, [NUM_INIT_TRIALS] + [SECOND_STEP_PARALLELISM] * (NUM_ROUNDS - 1)
        )

    def test_hierarchical_search_space(self):
        experiment = get_hierarchical_search_space_experiment()
        self.assertTrue(experiment.search_space.is_hierarchical)
        self.sobol_GS.gen(experiment=experiment)
        for _ in range(10):
            # During each iteration, check that all transformed observation features
            # contain all parameters of the flat search space.
            with patch.object(
                RandomModelBridge, "_fit"
            ) as mock_model_fit, patch.object(RandomModelBridge, "gen"):
                self.sobol_GS.gen(experiment=experiment)
                mock_model_fit.assert_called_once()
                obs_feats = mock_model_fit.call_args[1].get("observation_features")
                all_parameter_names = (
                    experiment.search_space._all_parameter_names.copy()
                )
                # One of the parameter names is modified by transforms (because it's
                # one-hot encoded).
                all_parameter_names.remove("model")
                all_parameter_names.add("model_OH_PARAM_")
                for obsf in obs_feats:
                    for p_name in all_parameter_names:
                        self.assertIn(p_name, obsf.parameters)

            trial = (
                experiment.new_trial(
                    generator_run=self.sobol_GS.gen(experiment=experiment)
                )
                .mark_running(no_runner_required=True)
                .mark_completed()
            )
            experiment.attach_data(
                get_data(
                    metric_name="m1",
                    trial_index=trial.index,
                    num_non_sq_arms=1,
                    include_sq=False,
                )
            )
            experiment.attach_data(
                get_data(
                    metric_name="m2",
                    trial_index=trial.index,
                    num_non_sq_arms=1,
                    include_sq=False,
                )
            )

    # ------------- Testing helpers (put tests above this line) -------------

    def _run_GS_for_N_rounds(
        self, gs: GenerationStrategy, exp: Experiment, num_rounds: int
    ) -> List[int]:
        could_gen = []
        for _ in range(num_rounds):
            (
                num_trials_to_gen,
                opt_complete,
            ) = gs.current_generator_run_limit()
            self.assertFalse(opt_complete)
            could_gen.append(num_trials_to_gen)
            trials = []

            for _ in range(num_trials_to_gen):
                gr = gs.gen(
                    experiment=exp,
                    pending_observations=get_pending(experiment=exp),
                )
                trials.append(exp.new_trial(gr).mark_running(no_runner_required=True))

            for trial in trials:
                exp.attach_data(get_branin_data(trial_indices=[trial.index]))
                trial.mark_completed()

        return could_gen
