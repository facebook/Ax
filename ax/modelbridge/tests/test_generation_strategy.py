#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import cast
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, Parameter, ParameterType
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.utils import (
    get_pending_observation_features_based_on_trial_status as get_pending,
)
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.exceptions.generation_strategy import (
    GenerationStrategyCompleted,
    GenerationStrategyMisconfiguredException,
    GenerationStrategyRepeatedPoints,
    MaxParallelismReachedException,
)
from ax.modelbridge.best_model_selector import SingleDiagnosticBestModelSelector
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    _extract_model_state_after_gen,
    Cont_X_trans,
    MODEL_KEY_TO_MODEL_SETUP,
    Models,
    ST_MTGP_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transition_criterion import (
    AutoTransitionAfterGen,
    MaxGenerationParallelism,
    MaxTrials,
    MinTrials,
)
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.equality import same_elements
from ax.utils.common.mock import mock_patch_method_original
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_choice_parameter,
    get_data,
    get_experiment_with_multi_objective,
    get_hierarchical_search_space_experiment,
)
from ax.utils.testing.mock import fast_botorch_optimize


class TestGenerationStrategy(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gr = GeneratorRun(arms=[Arm(parameters={"x1": 1, "x2": 2})])

        # Mock out slow model fitting.
        self.torch_model_bridge_patcher = patch(
            f"{TorchModelBridge.__module__}.TorchModelBridge", spec=True
        )
        self.mock_torch_model_bridge = self.torch_model_bridge_patcher.start()
        mock_mb = self.mock_torch_model_bridge.return_value
        mock_mb.gen.return_value = self.gr
        mock_mb._process_and_transform_data.return_value = (None, None)

        # Mock out slow TS.
        self.discrete_model_bridge_patcher = patch(
            f"{DiscreteModelBridge.__module__}.DiscreteModelBridge", spec=True
        )
        self.mock_discrete_model_bridge = self.discrete_model_bridge_patcher.start()
        self.mock_discrete_model_bridge.return_value.gen.return_value = self.gr

        # Mock in `Models` registry.
        self.registry_setup_dict_patcher = patch.dict(
            f"{Models.__module__}.MODEL_KEY_TO_MODEL_SETUP",
            {
                "Factorial": MODEL_KEY_TO_MODEL_SETUP["Factorial"]._replace(
                    bridge_class=self.mock_discrete_model_bridge
                ),
                "Thompson": MODEL_KEY_TO_MODEL_SETUP["Thompson"]._replace(
                    bridge_class=self.mock_discrete_model_bridge
                ),
                "BoTorch": MODEL_KEY_TO_MODEL_SETUP["BoTorch"]._replace(
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
        self.sobol_GS = GenerationStrategy(
            steps=[
                GenerationStep(
                    Models.SOBOL,
                    num_trials=-1,
                    should_deduplicate=True,
                )
            ]
        )
        self.sobol_MBM_step_GS = self._get_sobol_mbm_step_gs()

        # Set up the node-based generation strategy for testing.
        self.sobol_criterion = [
            MaxTrials(
                threshold=5,
                transition_to="MBM_node",
                block_gen_if_met=True,
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            )
        ]
        self.mbm_criterion = [
            MaxTrials(
                threshold=2,
                # this self-pointing isn't representative of real-world, but is
                # useful for testing attributes likes repr etc
                transition_to="MBM_node",
                block_gen_if_met=True,
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            )
        ]
        self.single_running_trial_criterion = [
            MaxTrials(
                threshold=1,
                transition_to="mbm",
                block_transition_if_unmet=True,
                only_in_statuses=[TrialStatus.RUNNING],
            )
        ]
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        self.mbm_model_spec = ModelSpec(
            model_enum=Models.BOTORCH_MODULAR,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        self.sobol_node = GenerationNode(
            node_name="sobol_node",
            transition_criteria=self.sobol_criterion,
            model_specs=[self.sobol_model_spec],
        )
        self.mbm_node = GenerationNode(
            node_name="MBM_node",
            transition_criteria=self.mbm_criterion,
            model_specs=[self.mbm_model_spec],
        )
        self.sobol_MBM_GS_nodes = GenerationStrategy(
            name="Sobol+MBM_Nodes",
            nodes=[self.sobol_node, self.mbm_node],
        )
        self.mbm_to_sobol2_max = MaxTrials(
            threshold=1,
            transition_to="sobol_2",
            block_transition_if_unmet=True,
            only_in_statuses=[TrialStatus.RUNNING],
            use_all_trials_in_exp=True,
        )
        self.mbm_to_sobol2_min = MinTrials(
            threshold=1,
            transition_to="sobol_2",
            block_transition_if_unmet=True,
            only_in_statuses=[TrialStatus.COMPLETED],
            use_all_trials_in_exp=True,
        )
        self.mbm_to_sobol_auto = AutoTransitionAfterGen(transition_to="sobol_3")
        self.competing_tc_gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="sobol",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=self.single_running_trial_criterion,
                ),
                GenerationNode(
                    node_name="mbm",
                    model_specs=[self.mbm_model_spec],
                    transition_criteria=[
                        self.mbm_to_sobol2_max,
                        self.mbm_to_sobol2_min,
                        self.mbm_to_sobol_auto,
                    ],
                ),
                GenerationNode(
                    node_name="sobol_2",
                    model_specs=[self.sobol_model_spec],
                ),
                GenerationNode(
                    node_name="sobol_3",
                    model_specs=[self.sobol_model_spec],
                ),
            ],
        )
        self.complex_multinode_per_trial_gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="sobol",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=self.single_running_trial_criterion,
                ),
                GenerationNode(
                    node_name="mbm",
                    model_specs=[self.mbm_model_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(
                            transition_to="sobol_2",
                        )
                    ],
                ),
                GenerationNode(
                    node_name="sobol_2",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(transition_to="sobol_3")
                    ],
                ),
                GenerationNode(
                    node_name="sobol_3",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=[
                        MaxTrials(
                            threshold=2,
                            transition_to="sobol_4",
                            block_transition_if_unmet=True,
                            only_in_statuses=[TrialStatus.RUNNING],
                            use_all_trials_in_exp=True,
                        ),
                        AutoTransitionAfterGen(
                            transition_to="mbm",
                            block_transition_if_unmet=True,
                            continue_trial_generation=False,
                        ),
                    ],
                ),
                GenerationNode(
                    node_name="sobol_4",
                    model_specs=[self.sobol_model_spec],
                ),
            ],
        )

    def tearDown(self) -> None:
        self.torch_model_bridge_patcher.stop()
        self.discrete_model_bridge_patcher.stop()
        self.registry_setup_dict_patcher.stop()

    def _get_sobol_mbm_step_gs(
        self, num_sobol_trials: int = 5, num_mbm_trials: int = -1
    ) -> GenerationStrategy:
        return GenerationStrategy(
            name="Sobol+MBM",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=num_sobol_trials,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=num_mbm_trials,
                    model_kwargs=self.step_model_kwargs,
                ),
            ],
        )

    def test_unique_step_names(self) -> None:
        """This tests the name of the steps on generation strategy. The name is
        inherited from the GenerationNode class, and for GenerationSteps the
        name should follow the format "GenerationNode"+Stepidx.
        """
        gs = self.sobol_MBM_step_GS
        self.assertEqual(gs._steps[0].node_name, "GenerationStep_0")
        self.assertEqual(gs._steps[1].node_name, "GenerationStep_1")

    def test_name(self) -> None:
        self.assertEqual(self.sobol_GS._name, "Sobol")
        self.assertEqual(
            self.sobol_MBM_step_GS.name,
            "Sobol+MBM",
        )
        self.sobol_GS._name = "SomeGSName"
        self.assertEqual(self.sobol_GS.name, "SomeGSName")

    def test_validation(self) -> None:
        # num_trials can be positive or -1.
        with self.assertRaises(UserInputError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=5),
                    GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-10),
                ]
            )

        # only last num_trials can be -1.
        with self.assertRaises(UserInputError):
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=-1),
                    GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=10),
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
                    GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1),
                ]
            )

    def test_custom_callables_for_models(self) -> None:
        exp = get_branin_experiment()
        sobol_factory_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=get_sobol, num_trials=-1)]
        )
        self.assertFalse(sobol_factory_generation_strategy._uses_registered_models)
        self.assertTrue(sobol_factory_generation_strategy.uses_non_registered_models)
        gr = sobol_factory_generation_strategy.gen(experiment=exp, n=1)
        self.assertEqual(len(gr.arms), 1)

    def test_string_representation(self) -> None:
        gs1 = self.sobol_MBM_step_GS
        self.assertEqual(
            str(gs1),
            (
                "GenerationStrategy(name='Sobol+MBM', steps=[Sobol for 5 trials,"
                " BoTorch for subsequent trials])"
            ),
        )
        gs2 = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)]
        )
        self.assertEqual(
            str(gs2), "GenerationStrategy(name='Sobol', steps=[Sobol for all trials])"
        )

        gs3 = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="test",
                    model_specs=[
                        ModelSpec(
                            model_enum=Models.SOBOL,
                            model_kwargs={},
                            model_gen_kwargs={},
                        ),
                    ],
                )
            ]
        )
        self.assertEqual(
            str(gs3),
            "GenerationStrategy(name='test', nodes=[GenerationNode("
            "model_specs=[ModelSpec(model_enum=Sobol, "
            "model_kwargs={}, model_gen_kwargs={}, model_cv_kwargs={},"
            " )], node_name=test, "
            "transition_criteria=[])])",
        )

    def test_equality(self) -> None:
        gs1 = self.sobol_MBM_step_GS
        gs2 = self.sobol_MBM_step_GS
        self.assertEqual(gs1, gs2)

        # Clone_reset() doesn't clone exactly, so they won't be equal.
        gs3 = gs1.clone_reset()
        self.assertEqual(gs1, gs3)

    def test_min_observed(self) -> None:
        # We should fail to transition the next model if there is not
        # enough data observed.
        # pyre-fixme[6]: For 1st param expected `bool` but got `Experiment`.
        exp = get_branin_experiment(get_branin_experiment())
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5, min_trials_observed=5),
                GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=1),
            ]
        )
        self.assertFalse(gs.uses_non_registered_models)
        for _ in range(5):
            exp.new_trial(gs.gen(exp))
        with self.assertRaises(DataRequiredError):
            gs.gen(exp)

    def test_do_not_enforce_min_observations(self) -> None:
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
                GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=1),
            ]
        )
        for _ in range(2):
            gs.gen(exp)
        # Make sure Sobol is used to generate the 6th point.
        self.assertIsInstance(gs._model, RandomModelBridge)

    def test_sobol_MBM_strategy(self) -> None:
        exp = get_branin_experiment()
        # New GS to test for GS completed error below.
        gs = self._get_sobol_mbm_step_gs(num_mbm_trials=2)
        expected_seed = None
        for i in range(7):
            g = gs.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(gs._generator_runs), i + 1)
            if i > 4:
                self.mock_torch_model_bridge.assert_called()
            else:
                self.assertEqual(g._model_key, "Sobol")
                mkw = g._model_kwargs
                self.assertIsNotNone(mkw)
                if i > 0:
                    # Generated points are randomized, so checking that they're there.
                    self.assertIsNotNone(mkw.get("generated_points"))
                else:
                    # This is the first GR, there should be no generated points yet.
                    self.assertIsNone(mkw.get("generated_points"))
                # Remove the randomized generated points to compare the rest.
                mkw = mkw.copy()
                del mkw["generated_points"]
                self.assertEqual(
                    mkw,
                    {
                        "seed": expected_seed,
                        "deduplicate": True,
                        "init_position": i,
                        "scramble": True,
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
                        "fit_tracking_metrics": True,
                        "fit_on_init": True,
                    },
                )
                ms = not_none(g._model_state_after_gen).copy()
                # Compare the model state to Sobol state.
                sobol_model = not_none(gs.model).model
                self.assertTrue(
                    np.array_equal(
                        ms.pop("generated_points"), sobol_model.generated_points
                    )
                )
                # Replace expected seed with the one generated in __init__.
                expected_seed = sobol_model.seed
                self.assertEqual(ms, {"init_position": i + 1, "seed": expected_seed})
        # Check completeness error message when GS should be done.
        with self.assertRaises(GenerationStrategyCompleted):
            g = gs.gen(exp)

    def test_sobol_MBM_strategy_keep_generating(self) -> None:
        exp = get_branin_experiment()
        exp.new_trial(generator_run=self.sobol_MBM_step_GS.gen(exp)).run()
        for i in range(1, 15):
            g = self.sobol_MBM_step_GS.gen(exp)
            exp.new_trial(generator_run=g).run()
            if i > 4:
                self.assertIsInstance(self.sobol_MBM_step_GS.model, TorchModelBridge)

    def test_sobol_strategy(self) -> None:
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=5,
                    max_parallelism=10,
                    enforce_num_trials=False,
                )
            ]
        )
        for i in range(1, 6):
            sobol_generation_strategy.gen(exp, n=1)
            self.assertEqual(len(sobol_generation_strategy._generator_runs), i)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `ax.utils.testing.core_stubs.get_data()` to decorator `unittest.mock.patch`.
    @patch(f"{Experiment.__module__}.Experiment.fetch_data", return_value=get_data())
    def test_factorial_thompson_strategy(self, _: MagicMock) -> None:
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
        mock_model_bridge = self.mock_discrete_model_bridge.return_value

        # Initial factorial batch.
        exp.new_batch_trial(factorial_thompson_generation_strategy.gen(experiment=exp))
        args, kwargs = mock_model_bridge._set_kwargs_to_save.call_args
        self.assertEqual(kwargs.get("model_key"), "Factorial")

        # Subsequent Thompson sampling batch.
        exp.new_batch_trial(factorial_thompson_generation_strategy.gen(experiment=exp))
        args, kwargs = mock_model_bridge._set_kwargs_to_save.call_args
        self.assertEqual(kwargs.get("model_key"), "Thompson")

    def test_clone_reset(self) -> None:
        ftgs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.FACTORIAL, num_trials=1),
                GenerationStep(model=Models.THOMPSON, num_trials=2),
            ]
        )
        ftgs._curr = ftgs._steps[1]
        self.assertEqual(ftgs.current_step_index, 1)
        self.assertEqual(ftgs.clone_reset().current_step_index, 0)

    def test_kwargs_passed(self) -> None:
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=1, model_kwargs={"scramble": False}
                )
            ]
        )
        exp = get_branin_experiment()
        gs.gen(exp)
        # pyre-fixme[16]: Optional type has no attribute `model`.
        self.assertFalse(gs._model.model.scramble)

    def test_sobol_MBM_strategy_batches(self) -> None:
        mock_MBM_gen = self.mock_torch_model_bridge.return_value.gen
        mock_MBM_gen.return_value = GeneratorRun(
            arms=[
                Arm(parameters={"x1": 1, "x2": 2}),
                Arm(parameters={"x1": 3, "x2": 4}),
            ]
        )
        exp = get_branin_experiment()
        sobol_MBM_generation_strategy = self._get_sobol_mbm_step_gs(
            num_sobol_trials=1, num_mbm_trials=6
        )
        gr = sobol_MBM_generation_strategy.gen(exp, n=2)
        exp.new_batch_trial(generator_run=gr).run()
        for i in range(1, 8):
            if i == 7:
                # Check completeness error message.
                with self.assertRaises(GenerationStrategyCompleted):
                    g = sobol_MBM_generation_strategy.gen(exp, n=2)
            else:
                g = sobol_MBM_generation_strategy.gen(exp, n=2)
            exp.new_batch_trial(generator_run=g).run()
        self.assertIsInstance(sobol_MBM_generation_strategy.model, TorchModelBridge)

    def test_with_factory_function(self) -> None:
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

    def test_store_experiment(self) -> None:
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5)]
        )
        self.assertIsNone(sobol_generation_strategy._experiment)
        sobol_generation_strategy.gen(exp)
        self.assertIsNotNone(sobol_generation_strategy._experiment)

    def test_trials_as_df(self) -> None:
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=2),
                GenerationStep(model=Models.SOBOL, num_trials=3),
            ]
        )
        # No trials yet, so the DF will be None.
        self.assertIsNone(sobol_generation_strategy.trials_as_df)
        # Now the trial should appear in the DF.
        trial = exp.new_trial(sobol_generation_strategy.gen(experiment=exp))
        trials_df = not_none(sobol_generation_strategy.trials_as_df)
        self.assertFalse(trials_df.empty)
        self.assertEqual(trials_df.head()["Trial Status"][0], "CANDIDATE")
        # Changes in trial status should be reflected in the DF.
        trial._status = TrialStatus.RUNNING
        trials_df = not_none(sobol_generation_strategy.trials_as_df)
        self.assertEqual(trials_df.head()["Trial Status"][0], "RUNNING")
        # Check that rows are present for step 0 and 1 after moving to step 1
        for _i in range(3):
            # attach necessary trials to fill up the Generation Strategy
            trial = exp.new_trial(sobol_generation_strategy.gen(experiment=exp))
        trials_df = not_none(sobol_generation_strategy.trials_as_df)
        self.assertEqual(trials_df.head()["Generation Step"][0], ["GenerationStep_0"])
        self.assertEqual(trials_df.head()["Generation Step"][2], ["GenerationStep_1"])

    def test_max_parallelism_reached(self) -> None:
        exp = get_branin_experiment()
        sobol_generation_strategy = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=5, max_parallelism=1)]
        )
        exp.new_trial(
            generator_run=sobol_generation_strategy.gen(experiment=exp)
        ).mark_running(no_runner_required=True)
        with self.assertRaises(MaxParallelismReachedException):
            sobol_generation_strategy.gen(experiment=exp)

    def test_deduplication(self) -> None:
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
            parameters=cast(list[Parameter], tiny_parameters)
        )
        exp = get_branin_experiment(search_space=tiny_search_space)
        sobol = GenerationStrategy(
            name="Sobol",
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    # Disable model-level deduplication.
                    model_kwargs={"deduplicate": False},
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
        ), mock.patch("ax.modelbridge.generation_node.logger.info") as mock_logger:
            g = sobol.gen(exp)
        self.assertEqual(mock_logger.call_count, 5)
        self.assertIn(
            "The generator run produced duplicate arms.", mock_logger.call_args[0][0]
        )

    def test_current_generator_run_limit(self) -> None:
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

    def test_current_generator_run_limit_unlimited_second_step(self) -> None:
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

    def test_hierarchical_search_space(self) -> None:
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
                observations = mock_model_fit.call_args[1].get("observations")
                all_parameter_names = checked_cast(
                    HierarchicalSearchSpace, experiment.search_space
                )._all_parameter_names.copy()
                for obs in observations:
                    for p_name in all_parameter_names:
                        self.assertIn(p_name, obs.features.parameters)

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

    def test_gen_multiple(self) -> None:
        exp = get_experiment_with_multi_objective()
        sobol_MBM_gs = self.sobol_MBM_step_GS

        with mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.gen",
            original_method=ModelSpec.gen,
        ) as model_spec_gen_mock, mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.fit",
            original_method=ModelSpec.fit,
        ) as model_spec_fit_mock:
            # Generate first four Sobol GRs (one more to gen after that if
            # first four become trials.
            grs = sobol_MBM_gs._gen_multiple(experiment=exp, num_generator_runs=3)
            self.assertEqual(len(grs), 3)
            # We should only fit once; refitting for each `gen` would be
            # wasteful as there is no new data.
            model_spec_fit_mock.assert_called_once()
            self.assertEqual(model_spec_gen_mock.call_count, 3)
            pending_in_each_gen = enumerate(
                args_and_kwargs.kwargs.get("pending_observations")
                for args_and_kwargs in model_spec_gen_mock.call_args_list
            )
            for gr, (idx, pending) in zip(grs, pending_in_each_gen):
                exp.new_trial(generator_run=gr).mark_running(no_runner_required=True)
                if idx > 0:
                    prev_gr = grs[idx - 1]
                    for arm in prev_gr.arms:
                        for m in pending:
                            self.assertIn(ObservationFeatures.from_arm(arm), pending[m])
            model_spec_gen_mock.reset_mock()

            # Check case with pending features initially specified; we should get two
            # GRs now (remaining in Sobol step) even though we requested 3.
            original_pending = not_none(get_pending(experiment=exp))
            first_3_trials_obs_feats = [
                ObservationFeatures.from_arm(arm=a, trial_index=idx)
                for idx, trial in exp.trials.items()
                for a in trial.arms
            ]
            for m in original_pending:
                self.assertTrue(
                    same_elements(original_pending[m], first_3_trials_obs_feats)
                )

            grs = sobol_MBM_gs._gen_multiple(
                experiment=exp,
                num_generator_runs=3,
                pending_observations=get_pending(experiment=exp),
            )
            self.assertEqual(len(grs), 2)

            pending_in_each_gen = enumerate(
                args_and_kwargs[1].get("pending_observations")
                for args_and_kwargs in model_spec_gen_mock.call_args_list
            )
            for gr, (idx, pending) in zip(grs, pending_in_each_gen):
                exp.new_trial(generator_run=gr).mark_running(no_runner_required=True)
                if idx > 0:
                    prev_gr = grs[idx - 1]
                    for arm in prev_gr.arms:
                        for m in pending:
                            # In this case, we should see both the originally-pending
                            # and the new arms as pending observation features.
                            self.assertIn(ObservationFeatures.from_arm(arm), pending[m])
                            for p in original_pending[m]:
                                self.assertIn(p, pending[m])

    def test_gen_for_multiple_trials_with_multiple_models(self) -> None:
        exp = get_experiment_with_multi_objective()
        sobol_MBM_gs = self.sobol_MBM_step_GS
        with mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.gen",
            original_method=ModelSpec.gen,
        ) as model_spec_gen_mock, mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.fit",
            original_method=ModelSpec.fit,
        ) as model_spec_fit_mock:
            # Generate first four Sobol GRs (one more to gen after that if
            # first four become trials.
            grs = sobol_MBM_gs.gen_for_multiple_trials_with_multiple_models(
                experiment=exp, num_generator_runs=3
            )
        self.assertEqual(len(grs), 3)
        for gr in grs:
            self.assertEqual(len(gr), 1)
            self.assertIsInstance(gr[0], GeneratorRun)

        # We should only fit once; refitting for each `gen` would be
        # wasteful as there is no new data.
        model_spec_fit_mock.assert_called_once()
        self.assertEqual(model_spec_gen_mock.call_count, 3)
        pending_in_each_gen = enumerate(
            args_and_kwargs.kwargs.get("pending_observations")
            for args_and_kwargs in model_spec_gen_mock.call_args_list
        )
        for gr, (idx, pending) in zip(grs, pending_in_each_gen):
            exp.new_trial(generator_run=gr[0]).mark_running(no_runner_required=True)
            if idx > 0:
                prev_grs = grs[idx - 1]
                for arm in prev_grs[0].arms:
                    for m in pending:
                        self.assertIn(ObservationFeatures.from_arm(arm), pending[m])
        model_spec_gen_mock.reset_mock()

        # Check case with pending features initially specified; we should get two
        # GRs now (remaining in Sobol step) even though we requested 3.
        original_pending = not_none(get_pending(experiment=exp))
        first_3_trials_obs_feats = [
            ObservationFeatures.from_arm(arm=a, trial_index=idx)
            for idx, trial in exp.trials.items()
            for a in trial.arms
        ]
        for m in original_pending:
            self.assertTrue(
                same_elements(original_pending[m], first_3_trials_obs_feats)
            )

        grs = sobol_MBM_gs.gen_for_multiple_trials_with_multiple_models(
            experiment=exp,
            num_generator_runs=3,
        )
        self.assertEqual(len(grs), 2)
        for gr in grs:
            self.assertEqual(len(gr), 1)
            self.assertIsInstance(gr[0], GeneratorRun)

        pending_in_each_gen = enumerate(
            args_and_kwargs[1].get("pending_observations")
            for args_and_kwargs in model_spec_gen_mock.call_args_list
        )
        for gr, (idx, pending) in zip(grs, pending_in_each_gen):
            exp.new_trial(generator_run=gr[0]).mark_running(no_runner_required=True)
            if idx > 0:
                prev_grs = grs[idx - 1]
                for arm in prev_grs[0].arms:
                    for m in pending:
                        # In this case, we should see both the originally-pending
                        # and the new arms as pending observation features.
                        self.assertIn(ObservationFeatures.from_arm(arm), pending[m])
                        for p in original_pending[m]:
                            self.assertIn(p, pending[m])

    @fast_botorch_optimize
    def test_gen_for_multiple_trials_with_multiple_models_with_fixed_features(
        self,
    ) -> None:
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=1,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.BO_MIXED,
                    num_trials=1,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    model_kwargs={
                        # this will cause an error if the model
                        # doesn't get fixed features
                        "transforms": ST_MTGP_trans,
                        **self.step_model_kwargs,
                    },
                    num_trials=1,
                ),
            ]
        )
        for _ in range(3):
            grs = gs.gen_for_multiple_trials_with_multiple_models(
                experiment=exp,
                num_generator_runs=1,
                n=2,
            )
            exp.new_batch_trial(generator_runs=grs[0]).mark_running(
                no_runner_required=True
            ).mark_completed()
            exp.fetch_data()

        # This is to ensure it generated from all nodes
        self.assertTrue(gs.optimization_complete)
        self.assertEqual(len(exp.trials), 3)

    # ---------- Tests for GenerationStrategies composed of GenerationNodes --------
    def test_gs_setup_with_nodes(self) -> None:
        """Test GS initialization and validation with nodes"""
        node_1_criterion = [
            MaxTrials(
                threshold=4,
                block_gen_if_met=False,
                transition_to="node_2",
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            ),
            MinTrials(
                only_in_statuses=[TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED],
                threshold=2,
                transition_to="node_2",
            ),
            MaxGenerationParallelism(
                threshold=1,
                only_in_statuses=[TrialStatus.RUNNING],
                block_gen_if_met=True,
                block_transition_if_unmet=False,
            ),
        ]
        node_1 = GenerationNode(
            node_name="node_1",
            transition_criteria=node_1_criterion,
            model_specs=[self.sobol_model_spec],
        )
        node_3 = GenerationNode(
            node_name="node_3",
            model_specs=[self.sobol_model_spec],
        )
        node_2 = GenerationNode(
            node_name="node_2",
            model_specs=[self.sobol_model_spec],
        )

        # check error raised if node names are not unique
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "All node names"
        ):
            GenerationStrategy(
                nodes=[
                    node_1,
                    node_1,
                ],
            )
        # check error raised if transition to argument is not valid
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "`transition_to` argument"
        ):
            GenerationStrategy(
                nodes=[
                    node_1,
                    node_3,
                ],
            )
        # check error raised if provided both steps and nodes
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "contain either steps or nodes"
        ):
            GenerationStrategy(
                nodes=[
                    node_1,
                    node_3,
                ],
                steps=[
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=5,
                        model_kwargs=self.step_model_kwargs,
                    ),
                    GenerationStep(
                        model=Models.BOTORCH_MODULAR,
                        num_trials=-1,
                        model_kwargs=self.step_model_kwargs,
                    ),
                ],
            )

        # check error raised if two transition criterion defining a single edge have
        # differing `continue_trial_generation` settings
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException,
            "should have the same `continue_trial_generation`",
        ):
            GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="node_1",
                        transition_criteria=[
                            MaxTrials(
                                threshold=4,
                                block_gen_if_met=False,
                                transition_to="node_2",
                                only_in_statuses=None,
                                not_in_statuses=[
                                    TrialStatus.FAILED,
                                    TrialStatus.ABANDONED,
                                ],
                                continue_trial_generation=False,
                            ),
                            MinTrials(
                                only_in_statuses=[
                                    TrialStatus.COMPLETED,
                                    TrialStatus.EARLY_STOPPED,
                                ],
                                threshold=2,
                                transition_to="node_2",
                                continue_trial_generation=True,
                            ),
                        ],
                        model_specs=[self.sobol_model_spec],
                    ),
                    node_2,
                ],
            )

        # check error raised if provided both steps and nodes under node list
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "`GenerationStrategy` inputs are:"
        ):
            GenerationStrategy(
                nodes=[
                    node_1,
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=5,
                        model_kwargs=self.step_model_kwargs,
                    ),
                    node_2,
                ],
            )
        # check that warning is logged if no nodes have transition arguments
        with self.assertLogs(GenerationStrategy.__module__, logging.WARNING) as logger:
            warning_msg = (
                "None of the nodes in this GenerationStrategy "
                "contain a `transition_to` argument in their transition_criteria. "
            )
            GenerationStrategy(
                nodes=[
                    node_2,
                    node_3,
                ],
            )
            self.assertTrue(
                any(warning_msg in output for output in logger.output),
                logger.output,
            )

    def test_gs_with_generation_nodes(self) -> None:
        "Simple test of a SOBOL + MBM GenerationStrategy composed of GenerationNodes"
        exp = get_branin_experiment()
        self.assertEqual(self.sobol_MBM_GS_nodes.name, "Sobol+MBM_Nodes")
        expected_seed = None

        for i in range(7):
            g = self.sobol_MBM_GS_nodes.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_MBM_GS_nodes._generator_runs), i + 1)
            if i > 4:
                self.mock_torch_model_bridge.assert_called()
            else:
                self.assertEqual(g._model_key, "Sobol")
                mkw = g._model_kwargs
                self.assertIsNotNone(mkw)
                if i > 0:
                    # Generated points are randomized, so checking that they're there.
                    self.assertIsNotNone(mkw.get("generated_points"))
                else:
                    # This is the first GR, there should be no generated points yet.
                    self.assertIsNone(mkw.get("generated_points"))
                # Remove the randomized generated points to compare the rest.
                mkw = mkw.copy()
                del mkw["generated_points"]
                self.assertEqual(
                    mkw,
                    {
                        "seed": expected_seed,
                        "deduplicate": True,
                        "init_position": i,
                        "scramble": True,
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
                        "fit_tracking_metrics": True,
                        "fit_on_init": True,
                    },
                )
                ms = not_none(g._model_state_after_gen).copy()
                # Compare the model state to Sobol state.
                sobol_model = not_none(self.sobol_MBM_GS_nodes.model).model
                self.assertTrue(
                    np.array_equal(
                        ms.pop("generated_points"), sobol_model.generated_points
                    )
                )
                # Replace expected seed with the one generated in __init__.
                expected_seed = sobol_model.seed
                self.assertEqual(ms, {"init_position": i + 1, "seed": expected_seed})

    def test_clone_reset_nodes(self) -> None:
        """Test that node-based generation strategy is appropriately reset
        when cloned with `clone_reset`.
        """
        exp = get_branin_experiment()
        for i in range(7):
            g = self.sobol_MBM_GS_nodes.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_MBM_GS_nodes._generator_runs), i + 1)
        gs_clone = self.sobol_MBM_GS_nodes.clone_reset()
        self.assertEqual(gs_clone.name, self.sobol_MBM_GS_nodes.name)
        self.assertEqual(gs_clone._generator_runs, [])

    def test_gs_with_nodes_and_blocking_criteria(self) -> None:
        sobol_node_with_criteria = GenerationNode(
            node_name="test",
            model_specs=[self.sobol_model_spec],
            transition_criteria=[
                MaxTrials(
                    threshold=3,
                    block_gen_if_met=True,
                    block_transition_if_unmet=True,
                    transition_to="MBM_node",
                ),
                MinTrials(
                    threshold=2,
                    only_in_statuses=[TrialStatus.COMPLETED],
                    block_gen_if_met=False,
                    block_transition_if_unmet=True,
                    transition_to="MBM_node",
                ),
            ],
        )
        mbm_node = GenerationNode(
            node_name="MBM_node",
            model_specs=[self.mbm_model_spec],
        )
        gs = GenerationStrategy(
            name="Sobol+MBM_Nodes",
            nodes=[sobol_node_with_criteria, mbm_node],
        )
        exp = get_branin_experiment()
        for _ in range(5):
            trial = exp.new_trial(
                generator_run=gs.gen(n=1, experiment=exp, data=exp.lookup_data())
            )
            trial.mark_running(no_runner_required=True)
            exp.attach_data(get_branin_data(trials=[trial]))
            trial.mark_completed()

    def test_step_based_gs_only(self) -> None:
        """Test the step_based_gs_only decorator"""
        gs_test = self.sobol_MBM_GS_nodes
        with self.assertRaisesRegex(
            UnsupportedError, "is not supported for GenerationNode based"
        ):
            gs_test.current_step_index

    def test_generation_strategy_eq_print(self) -> None:
        """
        Calling a GenerationStrategy's ``__repr__`` method should not alter
        its ``__dict__`` attribute.
        In the past, ``__repr__``  was causing ``name`` to change
        under the hood, resulting in
        ``RuntimeError: dictionary changed size during iteration.``
        This test ensures this issue doesn't reappear.
        """
        gs1 = self.sobol_MBM_step_GS
        gs2 = self.sobol_MBM_step_GS
        self.assertEqual(gs1, gs2)

    def test_gs_with_competing_transition_edges(self) -> None:
        """Test that a ``GenerationStrategy`` with a node with competing transition
        edges correctly transitions.
        """
        # this gs has a single sobol node which transitions to mbm. If the MaxTrials
        # and MinTrials criterion are met, the transition to sobol_2 should occur,
        # otherwise, should transition to sobol_3
        gs = self.competing_tc_gs
        exp = get_branin_experiment()

        # check that mbm will move to sobol_3 when MaxTrials and MinTrials are unmet
        exp.new_trial(generator_run=gs.gen(exp)).run()
        gs.gen(exp)
        self.assertEqual(gs.current_node_name, "mbm")
        gs.gen(exp)
        self.assertEqual(gs.current_node_name, "sobol_3")

    def test_transition_edges(self) -> None:
        """Test transition_edges property of ``GenerationNode``"""
        # this gs has a single sobol node which transitions to mbm. If the MaxTrials
        # and MinTrials criterion are met, the transition to sobol_2 should occur,
        # otherwise, should transition back to sobol.
        mbm_to_sobol_auto = AutoTransitionAfterGen(transition_to="sobol")
        gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="sobol",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=self.single_running_trial_criterion,
                ),
                GenerationNode(
                    node_name="mbm",
                    model_specs=[self.mbm_model_spec],
                    transition_criteria=[
                        self.mbm_to_sobol2_max,
                        self.mbm_to_sobol2_min,
                        mbm_to_sobol_auto,
                    ],
                ),
                GenerationNode(
                    node_name="sobol_2",
                    model_specs=[self.sobol_model_spec],
                ),
            ],
        )
        exp = get_branin_experiment()
        self.assertEqual(
            gs._curr.transition_edges, {"mbm": self.single_running_trial_criterion}
        )
        exp.new_trial(generator_run=gs.gen(exp)).run()
        gs.gen(exp)
        self.assertEqual(gs.current_node_name, "mbm")
        self.assertEqual(
            gs._curr.transition_edges,
            {
                "sobol_2": [self.mbm_to_sobol2_max, self.mbm_to_sobol2_min],
                "sobol": [mbm_to_sobol_auto],
            },
        )

    def test_multiple_arms_per_node(self) -> None:
        """Test that a ``GenerationStrategy`` which expects some trials to be composed
        of multiple nodes can generate multiple arms per node using `arms_per_node`.
        """
        exp = get_branin_experiment()
        gs = self.complex_multinode_per_trial_gs
        gs.experiment = exp
        # first check that arms_per node validation works
        arms_per_node = {
            "sobol": 3,
            "sobol_2": 2,
            "sobol_3": 1,
            "sobol_4": 4,
        }
        with self.assertRaisesRegex(UserInputError, "defined in `arms_per_node`"):
            gs.gen_with_multiple_nodes(exp, arms_per_node=arms_per_node)

        # now we will check that the first trial contains 3 arms, the sconed trial
        # contains 6 arms (2 from mbm, 1 from sobol_2, 3 from sobol_3), and all
        # remaining trials contain 4 arms
        arms_per_node = {
            "sobol": 3,
            "mbm": 1,
            "sobol_2": 2,
            "sobol_3": 3,
            "sobol_4": 4,
        }
        # for the first trial, we start on sobol, we generate the trial, but it hasn't
        # been run yet, so we remain on sobol
        trial0 = exp.new_batch_trial(
            generator_runs=gs.gen_with_multiple_nodes(exp, arms_per_node=arms_per_node)
        )
        self.assertEqual(len(trial0.arms_by_name), 3)
        self.assertEqual(trial0.generator_runs[0]._generation_node_name, "sobol")
        trial0.run()

        # after trial 0 is run, we create a trial with nodes mbm, sobol_2, and sobol_3
        # However, the sobol_3 criterion requires that we have two running trials. We
        # don't move onto sobol_4 until we have two running trials, instead we reset
        # to the last first node in a trial.
        for _i in range(0, 2):
            trial = exp.new_batch_trial(
                generator_runs=gs.gen_with_multiple_nodes(
                    exp, arms_per_node=arms_per_node
                )
            )
            self.assertEqual(gs.current_node_name, "sobol_3")
            self.assertEqual(len(trial.arms_by_name), 6)
            self.assertEqual(len(trial.generator_runs), 3)
            self.assertEqual(trial.generator_runs[0]._generation_node_name, "mbm")
            self.assertEqual(len(trial.generator_runs[0].arms), 1)
            self.assertEqual(trial.generator_runs[1]._generation_node_name, "sobol_2")
            self.assertEqual(len(trial.generator_runs[1].arms), 2)
            self.assertEqual(trial.generator_runs[2]._generation_node_name, "sobol_3")
            self.assertEqual(len(trial.generator_runs[2].arms), 3)

        # after running the next trial should be made from sobol 4
        trial.run()
        trial = exp.new_batch_trial(
            generator_runs=gs.gen_with_multiple_nodes(exp, arms_per_node=arms_per_node)
        )
        self.assertEqual(trial.generator_runs[0]._generation_node_name, "sobol_4")
        self.assertEqual(len(trial.generator_runs[0].arms), 4)

    def test_node_gs_with_auto_transitions(self) -> None:
        """Test that node-based generation strategies which leverage
        AutoTransitionAfterGen criterion correctly transition and create trials.
        """
        gs = GenerationStrategy(
            nodes=[
                # first node should be our exploration node and only grs from this node
                # should be on the first trial
                GenerationNode(
                    node_name="sobol",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=self.single_running_trial_criterion,
                ),
                # node 2,3,4 will be out iteration nodes, and grs from all 3 nodes
                # should be used to make the subsequent trials
                GenerationNode(
                    node_name="mbm",
                    model_specs=[self.mbm_model_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(transition_to="sobol_2")
                    ],
                ),
                GenerationNode(
                    node_name="sobol_2",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(transition_to="sobol_3")
                    ],
                ),
                GenerationNode(
                    node_name="sobol_3",
                    model_specs=[self.sobol_model_spec],
                    transition_criteria=[
                        AutoTransitionAfterGen(
                            transition_to="mbm",
                            block_transition_if_unmet=True,
                            continue_trial_generation=False,
                        )
                    ],
                ),
            ],
        )
        exp = get_branin_experiment()
        gs.experiment = exp

        # for the first trial, we start on sobol, we generate the trial, but it hasn't
        # been run yet, so we remain on sobol, after the trial  is run, the subsequent
        # trials should be from node mbm, sobol_2, and sobol_3
        self.assertEqual(gs.current_node_name, "sobol")
        trial0 = exp.new_batch_trial(generator_runs=gs.gen_with_multiple_nodes(exp))
        self.assertEqual(gs.current_node_name, "sobol")
        # while here, test the last generator run property on node
        self.assertEqual(gs.current_node.node_that_generated_last_gr, "sobol")

        trial0.run()
        for _i in range(0, 2):
            trial = exp.new_batch_trial(generator_runs=gs.gen_with_multiple_nodes(exp))
            self.assertEqual(gs.current_node_name, "sobol_3")
            self.assertEqual(len(trial.generator_runs), 3)
            self.assertEqual(trial.generator_runs[0]._generation_node_name, "mbm")
            self.assertEqual(trial.generator_runs[1]._generation_node_name, "sobol_2")
            self.assertEqual(trial.generator_runs[2]._generation_node_name, "sobol_3")

    def test_node_gs_with_auto_transitions_three_phase(self) -> None:
        exp = get_branin_experiment()
        gs_2 = self.complex_multinode_per_trial_gs
        gs_2.experiment = exp

        # for the first trial, we start on sobol, we generate the trial, but it hasn't
        # been run yet, so we remain on sobol
        self.assertEqual(gs_2.current_node_name, "sobol")
        trial0 = exp.new_batch_trial(generator_runs=gs_2.gen_with_multiple_nodes(exp))
        self.assertEqual(gs_2.current_node_name, "sobol")
        trial0.run()

        # after trial 0 is run, we create a trial with nodes mbm, sobol_2, and sobol_3
        # However, the sobol_3 criterion requires that we have two running trials. We
        # don't move onto sobol_4 until we have two running trials, instead we reset
        # to the last first node in a trial.
        for _i in range(0, 2):
            trial = exp.new_batch_trial(
                generator_runs=gs_2.gen_with_multiple_nodes(exp)
            )
            self.assertEqual(gs_2.current_node_name, "sobol_3")
            self.assertEqual(len(trial.generator_runs), 3)
            self.assertEqual(trial.generator_runs[0]._generation_node_name, "mbm")
            self.assertEqual(trial.generator_runs[1]._generation_node_name, "sobol_2")
            self.assertEqual(trial.generator_runs[2]._generation_node_name, "sobol_3")

        # after running the next trial should be made from sobol 4
        trial.run()
        trial = exp.new_batch_trial(generator_runs=gs_2.gen_with_multiple_nodes(exp))
        self.assertEqual(trial.generator_runs[0]._generation_node_name, "sobol_4")

    def test_trials_as_df_node_gs(self) -> None:
        exp = get_branin_experiment()
        gs = self.complex_multinode_per_trial_gs
        arms_per_node = {
            "sobol": 3,
            "mbm": 1,
            "sobol_2": 2,
            "sobol_3": 3,
            "sobol_4": 4,
        }
        gs.experiment = exp
        self.assertIsNone(gs.trials_as_df)
        # Now the trial should appear in the DF.
        trial = exp.new_batch_trial(
            generator_runs=gs.gen_with_multiple_nodes(exp, arms_per_node=arms_per_node)
        )
        trials_df = not_none(gs.trials_as_df)
        self.assertFalse(trials_df.empty)
        self.assertEqual(trials_df.head()["Trial Status"][0], "CANDIDATE")
        self.assertEqual(trials_df.head()["Generation Model(s)"][0], ["Sobol"])
        # Changes in trial status should be reflected in the DF.
        trial.run()
        self.assertEqual(not_none(gs.trials_as_df).head()["Trial Status"][0], "RUNNING")
        # Add a new trial which will be generated from multiple nodes, and check that
        # is properly reflected in the DF.
        trial = exp.new_batch_trial(
            generator_runs=gs.gen_with_multiple_nodes(exp, arms_per_node=arms_per_node)
        )
        self.assertEqual(
            not_none(gs.trials_as_df).head()["Generation Nodes"][1],
            ["mbm", "sobol_2", "sobol_3"],
        )

    # ------------- Testing helpers (put tests above this line) -------------

    def _run_GS_for_N_rounds(
        self, gs: GenerationStrategy, exp: Experiment, num_rounds: int
    ) -> list[int]:
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


class TestGenerationStrategyWithoutModelBridgeMocks(TestCase):
    """The test class above heavily mocks the modelbridge. This makes it
    difficult to test certain aspects of the GS. This is an alternative
    test class that makes use of mocking rather sparingly.
    """

    @fast_botorch_optimize
    @patch(
        "ax.modelbridge.generation_node._extract_model_state_after_gen",
        wraps=_extract_model_state_after_gen,
    )
    def test_with_model_selection(self, mock_model_state: Mock) -> None:
        """Test that a GS with a model selection node functions correctly."""
        best_model_selector = MagicMock(autospec=SingleDiagnosticBestModelSelector)
        best_model_idx = 0
        best_model_selector.best_model.side_effect = lambda model_specs: model_specs[
            best_model_idx
        ]
        gs = GenerationStrategy(
            name="Sobol+MBM/BO_MIXED",
            nodes=[
                GenerationNode(
                    node_name="Sobol",
                    model_specs=[ModelSpec(model_enum=Models.SOBOL)],
                    transition_criteria=[
                        MaxTrials(threshold=2, transition_to="MBM/BO_MIXED")
                    ],
                ),
                GenerationNode(
                    node_name="MBM/BO_MIXED",
                    model_specs=[
                        ModelSpec(model_enum=Models.BOTORCH_MODULAR),
                        ModelSpec(model_enum=Models.BO_MIXED),
                    ],
                    best_model_selector=best_model_selector,
                ),
            ],
        )
        exp = get_branin_experiment(with_completed_trial=True)
        # Gen with Sobol.
        exp.new_trial(gs.gen(experiment=exp))
        # Model state is not extracted since there is no past GR.
        mock_model_state.assert_not_called()
        exp.new_trial(gs.gen(experiment=exp))
        # Model state is extracted since there is a past GR.
        mock_model_state.assert_called_once()
        mock_model_state.reset_mock()
        # Gen with MBM/BO_MIXED.
        mbm_gr_1 = gs.gen(experiment=exp)
        # Model state is not extracted since there is no past GR from this node.
        mock_model_state.assert_not_called()
        mbm_gr_2 = gs.gen(experiment=exp)
        # Model state is extracted only once, since there is a GR from only
        # one of these models.
        mock_model_state.assert_called_once()
        # Verify that it was extracted from the previous GR.
        self.assertIs(mock_model_state.call_args.kwargs["generator_run"], mbm_gr_1)
        # Change the best model and verify that it generates as well.
        best_model_idx = 1
        mixed_gr_1 = gs.gen(experiment=exp)
        # Only one new call for the MBM model.
        self.assertEqual(mock_model_state.call_count, 2)
        gs.gen(experiment=exp)
        # Two new calls, since we have a GR from the mixed model as well.
        self.assertEqual(mock_model_state.call_count, 4)
        self.assertIs(
            mock_model_state.call_args_list[-2].kwargs["generator_run"], mbm_gr_2
        )
        self.assertIs(
            mock_model_state.call_args_list[-1].kwargs["generator_run"], mixed_gr_1
        )
