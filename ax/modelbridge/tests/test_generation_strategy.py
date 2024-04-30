#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from typing import cast, List
from unittest import mock
from unittest.mock import MagicMock, patch

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
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    MODEL_KEY_TO_MODEL_SETUP,
    Models,
    ST_MTGP_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transition_criterion import (
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

        # Mock out slow GPEI.
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

        # Set up the node-based generation strategy for testing.
        self.sobol_criterion = [
            MaxTrials(
                threshold=5,
                transition_to="GPEI_node",
                block_gen_if_met=True,
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            )
        ]
        self.gpei_criterion = [
            MaxTrials(
                threshold=2,
                transition_to=None,
                block_gen_if_met=True,
                only_in_statuses=None,
                not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
            )
        ]
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        self.gpei_model_spec = ModelSpec(
            model_enum=Models.GPEI,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        self.sobol_node = GenerationNode(
            node_name="sobol_node",
            transition_criteria=self.sobol_criterion,
            model_specs=[self.sobol_model_spec],
        )
        self.gpei_node = GenerationNode(
            node_name="GPEI_node",
            transition_criteria=self.gpei_criterion,
            model_specs=[self.gpei_model_spec],
        )

        self.sobol_GPEI_GS_nodes = GenerationStrategy(
            name="Sobol+GPEI_Nodes",
            nodes=[self.sobol_node, self.gpei_node],
        )

    def tearDown(self) -> None:
        self.torch_model_bridge_patcher.stop()
        self.discrete_model_bridge_patcher.stop()
        self.registry_setup_dict_patcher.stop()

    def test_unique_step_names(self) -> None:
        """This tests the name of the steps on generation strategy. The name is
        inherited from the GenerationNode class, and for GenerationSteps the
        name should follow the format "GenerationNode"+Stepidx.
        """
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        self.assertEqual(gs._steps[0].node_name, "GenerationStep_0")
        self.assertEqual(gs._steps[1].node_name, "GenerationStep_1")

    def test_name(self) -> None:
        self.assertEqual(self.sobol_GS._name, "Sobol")
        self.assertEqual(
            GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=5),
                    GenerationStep(model=Models.GPEI, num_trials=-1),
                ],
            ).name,
            "Sobol+GPEI",
        )
        self.sobol_GS._name = "SomeGSName"
        self.assertEqual(self.sobol_GS.name, "SomeGSName")

    def test_validation(self) -> None:
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

    def test_min_observed(self) -> None:
        # We should fail to transition the next model if there is not
        # enough data observed.
        # pyre-fixme[6]: For 1st param expected `bool` but got `Experiment`.
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
                GenerationStep(model=Models.GPEI, num_trials=1),
            ]
        )
        for _ in range(2):
            gs.gen(exp)
        # Make sure Sobol is used to generate the 6th point.
        self.assertIsInstance(gs._model, RandomModelBridge)

    def test_sobol_GPEI_strategy(self) -> None:
        exp = get_branin_experiment()
        self.assertEqual(self.sobol_GPEI_GS.name, "Sobol+GPEI")
        for i in range(7):
            g = self.sobol_GPEI_GS.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_GPEI_GS._generator_runs), i + 1)
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
                        "seed": None,
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
                ms = g._model_state_after_gen
                self.assertIsNotNone(ms)
                # Generated points are randomized, so just checking that they are there.
                self.assertIn("generated_points", ms)
                # Remove the randomized generated points to compare the rest.
                ms = ms.copy()
                del ms["generated_points"]
                self.assertEqual(ms, {"init_position": i + 1})
        # Check completeness error message when GS should be done.
        with self.assertRaises(GenerationStrategyCompleted):
            g = self.sobol_GPEI_GS.gen(exp)

    def test_sobol_GPEI_strategy_keep_generating(self) -> None:
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
        exp.new_trial(generator_run=sobol_GPEI_generation_strategy.gen(exp)).run()
        for i in range(1, 15):
            g = sobol_GPEI_generation_strategy.gen(exp)
            exp.new_trial(generator_run=g).run()
            if i > 4:
                self.assertIsInstance(
                    sobol_GPEI_generation_strategy.model, TorchModelBridge
                )

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

    def test_sobol_GPEI_strategy_batches(self) -> None:
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
        self.assertEqual(trials_df.head()["Generation Step"][0], "GenerationStep_0")
        self.assertEqual(trials_df.head()["Generation Step"][2], "GenerationStep_1")

        # construct the same GS as above but directly with nodes
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={},
            model_gen_kwargs={},
        )
        node_gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="sobol_2_trial",
                    model_specs=[sobol_model_spec],
                    transition_criteria=[
                        MaxTrials(
                            threshold=2,
                            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                            block_gen_if_met=True,
                            block_transition_if_unmet=True,
                            transition_to="sobol_3_trial",
                        )
                    ],
                ),
                GenerationNode(
                    node_name="sobol_3_trial",
                    model_specs=[sobol_model_spec],
                    transition_criteria=[
                        MaxTrials(
                            threshold=2,
                            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                            block_gen_if_met=True,
                            block_transition_if_unmet=True,
                            transition_to=None,
                        )
                    ],
                ),
            ]
        )
        self.assertIsNone(node_gs.trials_as_df)
        # Now the trial should appear in the DF.
        trial = exp.new_trial(node_gs.gen(experiment=exp))
        trials_df = not_none(node_gs.trials_as_df)
        self.assertFalse(trials_df.empty)
        self.assertEqual(trials_df.head()["Trial Status"][0], "CANDIDATE")
        # Changes in trial status should be reflected in the DF.
        trial._status = TrialStatus.RUNNING
        trials_df = not_none(node_gs.trials_as_df)
        self.assertEqual(trials_df.head()["Trial Status"][0], "RUNNING")
        # Check that rows are present for step 0 and 1 after moving to step 1
        for _i in range(3):
            # attach necessary trials to fill up the Generation Strategy
            trial = exp.new_trial(node_gs.gen(experiment=exp))
        trials_df = not_none(node_gs.trials_as_df)
        self.assertEqual(trials_df.head()["Generation Node"][0], "sobol_2_trial")
        self.assertEqual(trials_df.head()["Generation Node"][2], "sobol_3_trial")

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
            parameters=cast(List[Parameter], tiny_parameters)
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
                # One of the parameter names is modified by transforms (because it's
                # one-hot encoded).
                all_parameter_names.remove("model")
                all_parameter_names.add("model_OH_PARAM_")
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
        sobol_GPEI_gs = GenerationStrategy(
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
        with mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.gen",
            original_method=ModelSpec.gen,
        ) as model_spec_gen_mock, mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.fit",
            original_method=ModelSpec.fit,
        ) as model_spec_fit_mock:
            # Generate first four Sobol GRs (one more to gen after that if
            # first four become trials.
            grs = sobol_GPEI_gs._gen_multiple(experiment=exp, num_generator_runs=3)
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

            grs = sobol_GPEI_gs._gen_multiple(
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
        sobol_GPEI_gs = GenerationStrategy(
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
        with mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.gen",
            original_method=ModelSpec.gen,
        ) as model_spec_gen_mock, mock_patch_method_original(
            mock_path=f"{ModelSpec.__module__}.ModelSpec.fit",
            original_method=ModelSpec.fit,
        ) as model_spec_fit_mock:
            # Generate first four Sobol GRs (one more to gen after that if
            # first four become trials.
            grs = sobol_GPEI_gs.gen_for_multiple_trials_with_multiple_models(
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

        grs = sobol_GPEI_gs.gen_for_multiple_trials_with_multiple_models(
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
                    model=Models.GPEI,
                    num_trials=1,
                    model_kwargs=self.step_model_kwargs,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    model_kwargs={
                        # this will cause and error if the model
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
        """Test GS initalization and validation with nodes"""
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={},
            model_gen_kwargs={"n": 2},
        )
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

        # check error raised if node names are not unique
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "All node names"
        ):
            GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="node_1",
                        transition_criteria=node_1_criterion,
                        model_specs=[sobol_model_spec],
                    ),
                    GenerationNode(
                        node_name="node_1",
                        model_specs=[sobol_model_spec],
                    ),
                ],
            )
        # check error raised if transition to argument is not valid
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "`transition_to` argument"
        ):
            GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="node_1",
                        transition_criteria=node_1_criterion,
                        model_specs=[sobol_model_spec],
                    ),
                    GenerationNode(
                        node_name="node_3",
                        model_specs=[sobol_model_spec],
                    ),
                ],
            )

        # check error raised if provided both steps and nodes
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "contain either steps or nodes"
        ):
            GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="node_1",
                        transition_criteria=node_1_criterion,
                        model_specs=[sobol_model_spec],
                    ),
                    GenerationNode(
                        node_name="node_3",
                        model_specs=[sobol_model_spec],
                    ),
                ],
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
                ],
            )

        # check error raised if provided both steps and nodes under node list
        with self.assertRaisesRegex(
            GenerationStrategyMisconfiguredException, "`GenerationStrategy` inputs are:"
        ):
            GenerationStrategy(
                nodes=[
                    GenerationNode(
                        node_name="node_1",
                        transition_criteria=node_1_criterion,
                        model_specs=[sobol_model_spec],
                    ),
                    GenerationStep(
                        model=Models.SOBOL,
                        num_trials=5,
                        model_kwargs=self.step_model_kwargs,
                    ),
                    GenerationNode(
                        node_name="node_2",
                        model_specs=[sobol_model_spec],
                    ),
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
                    GenerationNode(
                        node_name="node_1",
                        model_specs=[sobol_model_spec],
                    ),
                    GenerationNode(
                        node_name="node_3",
                        model_specs=[sobol_model_spec],
                    ),
                ],
            )
            self.assertTrue(
                any(warning_msg in output for output in logger.output),
                logger.output,
            )

    def test_gs_with_generation_nodes(self) -> None:
        "Simple test of a SOBOL + GPEI GenerationStrategy composed of GenerationNodes"
        exp = get_branin_experiment()
        self.assertEqual(self.sobol_GPEI_GS_nodes.name, "Sobol+GPEI_Nodes")

        for i in range(7):
            g = self.sobol_GPEI_GS_nodes.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_GPEI_GS_nodes._generator_runs), i + 1)
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
                        "seed": None,
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
                ms = g._model_state_after_gen
                self.assertIsNotNone(ms)
                # Generated points are randomized, so just checking that they are there.
                self.assertIn("generated_points", ms)
                # Remove the randomized generated points to compare the rest.
                ms = ms.copy()
                del ms["generated_points"]
                self.assertEqual(ms, {"init_position": i + 1})

    def test_clone_reset_nodes(self) -> None:
        """Test that node-based generation strategy is appropriately reset
        when cloned with `clone_reset`.
        """
        exp = get_branin_experiment()
        for i in range(7):
            g = self.sobol_GPEI_GS_nodes.gen(exp)
            exp.new_trial(generator_run=g).run()
            self.assertEqual(len(self.sobol_GPEI_GS_nodes._generator_runs), i + 1)
        gs_clone = self.sobol_GPEI_GS_nodes.clone_reset()
        self.assertEqual(gs_clone.name, self.sobol_GPEI_GS_nodes.name)
        self.assertEqual(gs_clone._generator_runs, [])

    def test_gs_with_nodes_and_blocking_criteria(self) -> None:
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        sobol_node_with_criteria = GenerationNode(
            node_name="test",
            model_specs=[sobol_model_spec],
            transition_criteria=[
                MaxTrials(
                    threshold=3,
                    block_gen_if_met=True,
                    block_transition_if_unmet=True,
                    transition_to="GPEI_node",
                ),
                MinTrials(
                    threshold=2,
                    only_in_statuses=[TrialStatus.COMPLETED],
                    block_gen_if_met=False,
                    block_transition_if_unmet=True,
                    transition_to="GPEI_node",
                ),
            ],
        )
        gpei_model_spec = ModelSpec(
            model_enum=Models.GPEI,
            model_kwargs=self.step_model_kwargs,
            model_gen_kwargs={},
        )
        gpei_node = GenerationNode(
            node_name="GPEI_node",
            model_specs=[gpei_model_spec],
        )
        gs = GenerationStrategy(
            name="Sobol+GPEI_Nodes",
            nodes=[sobol_node_with_criteria, gpei_node],
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
        sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={},
            model_gen_kwargs={"n": 2},
        )
        gs_test = GenerationStrategy(
            nodes=[
                GenerationNode(
                    node_name="node_1",
                    model_specs=[sobol_model_spec],
                ),
                GenerationNode(
                    node_name="node_2",
                    model_specs=[sobol_model_spec],
                ),
            ],
        )
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
        gs1 = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        gs2 = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        self.assertEqual(gs1, gs2)

    def test_generation_strategy_eq_no_print(self) -> None:
        """
        Calling a GenerationStrategy's ``__repr__`` method should not alter
        its ``__dict__`` attribute.
        In the past, ``__repr__``  was causing ``name`` to change
        under the hood, resulting in
        ``RuntimeError: dictionary changed size during iteration.``
        This test ensures this issue doesn't reappear.
        """
        gs1 = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        gs2 = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=5),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )
        self.assertEqual(gs1, gs2)

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
