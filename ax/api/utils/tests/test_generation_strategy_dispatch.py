#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from itertools import product
from typing import Any

import torch
from ax.adapter.registry import Generators
from ax.api.utils.generation_strategy_dispatch import choose_generation_strategy
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.dispatch_utils import get_derelativize_config
from ax.generation_strategy.transition_criterion import MinTrials
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.utils import run_trials_with_gs
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.map_saas import EnsembleMapSaasSingleTaskGP
from pyre_extensions import assert_is_instance, none_throws


class TestDispatchUtils(TestCase):
    def test_choose_gs_random_search(self) -> None:
        struct_kws_cases: dict[str, dict[str, Any]] = {
            "use_center_false": {"initialize_with_center": False},
            "use_center_true": {"initialize_with_center": True},
            "default": {},
        }
        use_center_cases = {
            "use_center_false": False,
            "use_center_true": True,
            "default": True,
        }
        for case, struct_kws in struct_kws_cases.items():
            with self.subTest(case=case):
                struct = GenerationStrategyDispatchStruct(
                    method="random_search", **struct_kws
                )
                use_center = use_center_cases[case]
                gs = choose_generation_strategy(struct=struct)
                self.assertEqual(len(gs._nodes), 1 + use_center)
                if use_center:
                    self.assertIsInstance(gs._nodes[0], CenterGenerationNode)
                    self.assertEqual(gs.name, "Center+QuasiRandomSearch")
                else:
                    self.assertEqual(gs.name, "QuasiRandomSearch")
                sobol_node = gs._nodes[-1]
                self.assertEqual(len(sobol_node.generator_specs), 1)
                sobol_spec = sobol_node.generator_specs[0]
                self.assertEqual(sobol_spec.generator_enum, Generators.SOBOL)
                self.assertEqual(sobol_spec.generator_kwargs, {"seed": None})
                self.assertEqual(sobol_node._transition_criteria, [])
                # Make sure it generates.
                run_trials_with_gs(
                    experiment=get_branin_experiment(), gs=gs, num_trials=3
                )

    @mock_botorch_optimize
    def test_choose_gs_fast_with_options(self) -> None:
        struct = GenerationStrategyDispatchStruct(
            method="fast",
            initialization_budget=3,
            initialization_random_seed=0,
            use_existing_trials_for_initialization=False,
            min_observed_initialization_trials=4,
            allow_exceeding_initialization_budget=True,
            torch_device="cpu",
            initialize_with_center=True,
        )
        gs = choose_generation_strategy(struct=struct)
        self.assertEqual(len(gs._nodes), 3)
        self.assertEqual(gs.name, "Center+Sobol+MBM:fast")
        # Check the center node.
        center_node = assert_is_instance(gs._nodes[0], CenterGenerationNode)
        self.assertEqual(center_node.next_node_name, "Sobol")
        # Check the Sobol node & TC.
        sobol_node = gs._nodes[1]
        self.assertTrue(sobol_node.should_deduplicate)
        self.assertEqual(len(sobol_node.generator_specs), 1)
        sobol_spec = sobol_node.generator_specs[0]
        self.assertEqual(sobol_spec.generator_enum, Generators.SOBOL)
        self.assertEqual(sobol_spec.generator_kwargs, {"seed": 0})
        expected_tc = [
            MinTrials(
                threshold=2,
                transition_to="MBM",
                block_gen_if_met=False,
                block_transition_if_unmet=True,
                use_all_trials_in_exp=False,
            ),
            MinTrials(
                threshold=4,
                transition_to="MBM",
                block_gen_if_met=False,
                block_transition_if_unmet=True,
                use_all_trials_in_exp=True,
                only_in_statuses=[TrialStatus.COMPLETED],
                count_only_trials_with_data=True,
            ),
        ]
        self.assertEqual(sobol_node._transition_criteria, expected_tc)
        # Check the MBM node.
        mbm_node = gs._nodes[2]
        self.assertTrue(mbm_node.should_deduplicate)
        self.assertEqual(len(mbm_node.generator_specs), 1)
        mbm_spec = mbm_node.generator_specs[0]
        self.assertEqual(mbm_spec.generator_enum, Generators.BOTORCH_MODULAR)
        expected_ss = SurrogateSpec(model_configs=[ModelConfig(name="MBM defaults")])
        self.assertEqual(
            mbm_spec.generator_kwargs,
            {
                "surrogate_spec": expected_ss,
                "torch_device": torch.device("cpu"),
                "transform_configs": get_derelativize_config(
                    derelativize_with_raw_status_quo=True
                ),
                "acquisition_options": {"prune_irrelevant_parameters": False},
            },
        )
        self.assertEqual(mbm_node._transition_criteria, [])
        # Experiment with 2 observations. We should still generate 4 Sobol trials.
        experiment = get_experiment_with_observations([[1.0], [2.0]])
        # Mark the existing trials as manual to prevent them from counting for Sobol.
        for trial in experiment.trials.values():
            none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._generator_key = "Manual"
        # Generate 5 trials and make sure they're from the correct nodes.
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=5)
        self.assertEqual(len(experiment.trials), 7)
        for trial in experiment.trials.values():
            model_key = none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._generator_key
            if trial.index < 2:
                self.assertEqual(model_key, "Manual")
            elif trial.index == 2:
                self.assertEqual(model_key, "CenterOfSearchSpace")
            elif trial.index < 5:
                self.assertEqual(model_key, "Sobol")
            else:
                self.assertEqual(model_key, "BoTorch")

    def test_choose_gs_quality_with_options(self) -> None:
        struct = GenerationStrategyDispatchStruct(
            method="quality",
            initialization_budget=3,
            initialization_random_seed=0,
            use_existing_trials_for_initialization=False,
            min_observed_initialization_trials=4,
            allow_exceeding_initialization_budget=True,
            torch_device="cpu",
            initialize_with_center=True,
        )
        gs = choose_generation_strategy(struct=struct)
        self.assertEqual(len(gs._nodes), 3)
        self.assertEqual(gs.name, "Center+Sobol+MBM:quality")

        # Check the MBM node.
        mbm_node = gs._nodes[2]
        self.assertTrue(mbm_node.should_deduplicate)
        self.assertEqual(len(mbm_node.generator_specs), 1)
        mbm_spec = mbm_node.generator_specs[0]
        self.assertEqual(mbm_spec.generator_enum, Generators.BOTORCH_MODULAR)
        expected_ss = SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=SaasFullyBayesianSingleTaskGP,
                    model_options={"use_input_warping": True},
                    mll_options={
                        "disable_progbar": True,
                    },
                    name="WarpedSAAS",
                )
            ]
        )
        self.assertEqual(
            mbm_spec.generator_kwargs,
            {
                "surrogate_spec": expected_ss,
                "torch_device": torch.device("cpu"),
                "transform_configs": get_derelativize_config(
                    derelativize_with_raw_status_quo=True
                ),
                "acquisition_options": {"prune_irrelevant_parameters": False},
            },
        )
        self.assertEqual(mbm_node._transition_criteria, [])

    def test_choose_gs_no_initialization(self) -> None:
        struct = GenerationStrategyDispatchStruct(
            method="fast", initialization_budget=0
        )
        gs = choose_generation_strategy(struct=struct)
        self.assertEqual(len(gs._nodes), 1)
        self.assertEqual(gs.name, "MBM:fast")
        mbm_node = gs._nodes[0]
        self.assertEqual(mbm_node.name, "MBM")

    def test_choose_gs_center_only_initialization(self) -> None:
        struct = GenerationStrategyDispatchStruct(
            method="fast", initialization_budget=1, initialize_with_center=True
        )
        gs = choose_generation_strategy(struct=struct)
        self.assertEqual(len(gs._nodes), 2)
        self.assertEqual(gs.name, "Center+MBM:fast")
        center_node = gs._nodes[0]
        self.assertEqual(center_node.name, "CenterOfSearchSpace")
        mbm_node = gs._nodes[1]
        self.assertEqual(mbm_node.name, "MBM")

    def test_choose_gs_single_sobol_initialization(self) -> None:
        struct = GenerationStrategyDispatchStruct(
            method="fast", initialization_budget=1, initialize_with_center=False
        )
        gs = choose_generation_strategy(struct=struct)
        self.assertEqual(len(gs._nodes), 2)
        self.assertEqual(gs.name, "Sobol+MBM:fast")
        sobol_node = gs._nodes[0]
        self.assertEqual(sobol_node.name, "Sobol")
        mbm_node = gs._nodes[1]
        self.assertEqual(mbm_node.name, "MBM")

    def test_gs_simplify_parameter_changes(self) -> None:
        for simplify, method in product((True, False), ("fast", "quality")):
            struct = GenerationStrategyDispatchStruct(
                # pyre-fixme [6]: In call
                # `GenerationStrategyDispatchStruct.__init__`, for argument
                # `method`, expected `Union[typing_extensions.Literal['fast'],
                # typing_extensions.Literal['quality'],
                # typing_extensions.Literal['random_search']]` but got `str`
                method=method,
                simplify_parameter_changes=simplify,
            )
            gs = choose_generation_strategy(struct=struct)
            self.assertEqual(gs.name, f"Center+Sobol+MBM:{method}")
            mbm_node = gs._nodes[2]
            mbm_spec = mbm_node.generator_specs[0]
            self.assertEqual(
                mbm_spec.generator_kwargs["acquisition_options"],
                {"prune_irrelevant_parameters": simplify},
            )

    def test_choose_gs_custom_with_model_config(self) -> None:
        """Test that custom method works with a provided ModelConfig."""
        custom_model_config = ModelConfig(
            botorch_model_class=EnsembleMapSaasSingleTaskGP,
            name="MAPSAAS",
        )
        struct = GenerationStrategyDispatchStruct(
            method="custom",
            initialization_budget=3,
            initialize_with_center=False,
            torch_device="cpu",
        )
        gs = choose_generation_strategy(struct=struct, model_config=custom_model_config)
        self.assertEqual(len(gs._nodes), 2)
        self.assertEqual(gs.name, "Sobol+MBM:MAPSAAS")

        # Check the MBM node uses the custom model config.
        mbm_node = gs._nodes[1]
        self.assertEqual(len(mbm_node.generator_specs), 1)
        mbm_spec = mbm_node.generator_specs[0]
        self.assertEqual(mbm_spec.generator_enum, Generators.BOTORCH_MODULAR)
        expected_ss = SurrogateSpec(model_configs=[custom_model_config])
        self.assertEqual(
            mbm_spec.generator_kwargs["surrogate_spec"],
            expected_ss,
        )
        self.assertEqual(
            mbm_spec.generator_kwargs["torch_device"],
            torch.device("cpu"),
        )

    def test_choose_gs_custom_without_name(self) -> None:
        """Test that custom method works with unnamed ModelConfig."""
        custom_model_config = ModelConfig(
            botorch_model_class=SaasFullyBayesianSingleTaskGP,
            # No name provided.
        )
        struct = GenerationStrategyDispatchStruct(
            method="custom",
            initialization_budget=3,
            initialize_with_center=False,
        )
        gs = choose_generation_strategy(struct=struct, model_config=custom_model_config)
        # Should use "custom_config" as the default name.
        self.assertEqual(gs.name, "Sobol+MBM:custom_config")

    def test_choose_gs_custom_model_config_validation(self) -> None:
        """Test validation of model_config and custom method pairing."""
        # Test that custom method raises an error when model_config is not provided.
        struct = GenerationStrategyDispatchStruct(method="custom")
        with self.assertRaisesRegex(
            UserInputError,
            "model_config must be provided when method='custom'.",
        ):
            choose_generation_strategy(struct=struct)

        # Test that providing model_config without custom method raises an error.
        custom_model_config = ModelConfig(name="SomeConfig")
        struct = GenerationStrategyDispatchStruct(method="fast")
        with self.assertRaisesRegex(
            UserInputError,
            "model_config should only be provided when method='custom'. "
            "Got method='fast'.",
        ):
            choose_generation_strategy(struct=struct, model_config=custom_model_config)

    def test_choose_gs_with_custom_botorch_acqf_class(self) -> None:
        """Test that custom botorch_acqf_class is properly passed to generator kwargs
        and appended to the node name. Tests both fast and custom methods.
        """
        for method, model_config, expected_name in [
            ("fast", None, "Sobol+MBM:fast+qLogNoisyExpectedImprovement"),
            (
                "custom",
                ModelConfig(
                    botorch_model_class=EnsembleMapSaasSingleTaskGP,
                    name="MAPSAAS",
                ),
                "Sobol+MBM:MAPSAAS+qLogNoisyExpectedImprovement",
            ),
        ]:
            with self.subTest(method=method):
                struct = GenerationStrategyDispatchStruct(
                    method=method,  # pyre-ignore [6]
                    initialization_budget=3,
                    initialize_with_center=False,
                )
                gs = choose_generation_strategy(
                    struct=struct,
                    model_config=model_config,
                    botorch_acqf_class=qLogNoisyExpectedImprovement,
                )
                # Check that the name includes the acquisition function class name.
                self.assertEqual(gs.name, expected_name)

                # Check that MBM node generator kwargs include the botorch_acqf_class.
                mbm_node = gs._nodes[1]
                self.assertEqual(len(mbm_node.generator_specs), 1)
                mbm_spec = mbm_node.generator_specs[0]
                self.assertEqual(mbm_spec.generator_enum, Generators.BOTORCH_MODULAR)
                self.assertEqual(
                    mbm_spec.generator_kwargs["botorch_acqf_class"],
                    qLogNoisyExpectedImprovement,
                )
                # Check surrogate spec uses expected model config.
                expected_model_config = (
                    model_config
                    if model_config is not None
                    else ModelConfig(name="MBM defaults")
                )
                expected_ss = SurrogateSpec(model_configs=[expected_model_config])
                self.assertEqual(
                    mbm_spec.generator_kwargs["surrogate_spec"], expected_ss
                )
