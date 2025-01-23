#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from ax.core.base_trial import TrialStatus
from ax.core.trial import Trial
from ax.modelbridge.registry import Models
from ax.modelbridge.transition_criterion import MinTrials
from ax.models.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.preview.api.configs import GenerationMethod, GenerationStrategyConfig
from ax.preview.modelbridge.dispatch_utils import choose_generation_strategy
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.utils import run_trials_with_gs
from botorch.models.transforms.input import Normalize, Warp
from gpytorch.kernels.linear_kernel import LinearKernel
from pyre_extensions import assert_is_instance, none_throws


class TestDispatchUtils(TestCase):
    def test_choose_gs_random_search(self) -> None:
        gs_config = GenerationStrategyConfig(
            method=GenerationMethod.RANDOM_SEARCH,
        )
        gs = choose_generation_strategy(gs_config=gs_config)
        self.assertEqual(len(gs._nodes), 1)
        sobol_node = gs._nodes[0]
        self.assertEqual(len(sobol_node.model_specs), 1)
        sobol_spec = sobol_node.model_specs[0]
        self.assertEqual(sobol_spec.model_enum, Models.SOBOL)
        self.assertEqual(sobol_spec.model_kwargs, {"seed": None})
        self.assertEqual(sobol_node._transition_criteria, [])
        # Make sure it generates.
        run_trials_with_gs(experiment=get_branin_experiment(), gs=gs, num_trials=3)

    @mock_botorch_optimize
    def test_choose_gs_fast_with_options(self) -> None:
        gs_config = GenerationStrategyConfig(
            method=GenerationMethod.FAST,
            initialization_budget=3,
            initialization_random_seed=0,
            use_existing_trials_for_initialization=False,
            min_observed_initialization_trials=4,
            allow_exceeding_initialization_budget=True,
            torch_device="cpu",
        )
        gs = choose_generation_strategy(gs_config=gs_config)
        self.assertEqual(len(gs._nodes), 2)
        # Check the Sobol node & TC.
        sobol_node = gs._nodes[0]
        self.assertTrue(sobol_node.should_deduplicate)
        self.assertEqual(len(sobol_node.model_specs), 1)
        sobol_spec = sobol_node.model_specs[0]
        self.assertEqual(sobol_spec.model_enum, Models.SOBOL)
        self.assertEqual(sobol_spec.model_kwargs, {"seed": 0})
        expected_tc = [
            MinTrials(
                threshold=3,
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
                use_all_trials_in_exp=False,
                only_in_statuses=[TrialStatus.COMPLETED],
                count_only_trials_with_data=True,
            ),
        ]
        self.assertEqual(sobol_node._transition_criteria, expected_tc)
        # Check the MBM node.
        mbm_node = gs._nodes[1]
        self.assertTrue(mbm_node.should_deduplicate)
        self.assertEqual(len(mbm_node.model_specs), 1)
        mbm_spec = mbm_node.model_specs[0]
        self.assertEqual(mbm_spec.model_enum, Models.BOTORCH_MODULAR)
        expected_ss = SurrogateSpec(model_configs=[ModelConfig(name="MBM defaults")])
        self.assertEqual(
            mbm_spec.model_kwargs,
            {"surrogate_spec": expected_ss, "torch_device": torch.device("cpu")},
        )
        self.assertEqual(mbm_node._transition_criteria, [])
        # Experiment with 2 observations. We should still generate 4 Sobol trials.
        experiment = get_experiment_with_observations([[1.0], [2.0]])
        # Mark the existing trials as manual to prevent them from counting for Sobol.
        for trial in experiment.trials.values():
            none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._model_key = "Manual"
        # Generate 5 trials and make sure they're from the correct nodes.
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=5)
        self.assertEqual(len(experiment.trials), 7)
        for trial in experiment.trials.values():
            model_key = none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._model_key
            if trial.index < 2:
                self.assertEqual(model_key, "Manual")
            elif trial.index < 6:
                self.assertEqual(model_key, "Sobol")
            else:
                self.assertEqual(model_key, "BoTorch")

    @mock_botorch_optimize
    def test_choose_gs_defaults(self) -> None:
        gs = choose_generation_strategy(gs_config=GenerationStrategyConfig())
        self.assertEqual(len(gs._nodes), 2)
        # Check the Sobol node & TC.
        sobol_node = gs._nodes[0]
        self.assertTrue(sobol_node.should_deduplicate)
        self.assertEqual(len(sobol_node.model_specs), 1)
        sobol_spec = sobol_node.model_specs[0]
        self.assertEqual(sobol_spec.model_enum, Models.SOBOL)
        self.assertEqual(sobol_spec.model_kwargs, {"seed": None})
        expected_tc = [
            MinTrials(
                threshold=5,
                transition_to="MBM",
                block_gen_if_met=True,
                block_transition_if_unmet=True,
                use_all_trials_in_exp=True,
            ),
            MinTrials(
                threshold=2,
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
        mbm_node = gs._nodes[1]
        self.assertTrue(mbm_node.should_deduplicate)
        self.assertEqual(len(mbm_node.model_specs), 1)
        mbm_spec = mbm_node.model_specs[0]
        self.assertEqual(mbm_spec.model_enum, Models.BOTORCH_MODULAR)
        expected_ss = SurrogateSpec(
            model_configs=[
                ModelConfig(name="MBM defaults"),
                ModelConfig(
                    covar_module_class=LinearKernel,
                    input_transform_classes=[Warp, Normalize],
                    input_transform_options={"Normalize": {"center": 0.0}},
                    name="LinearKernel with Warp",
                ),
            ]
        )
        self.assertEqual(
            mbm_spec.model_kwargs, {"surrogate_spec": expected_ss, "torch_device": None}
        )
        self.assertEqual(mbm_node._transition_criteria, [])
        # Experiment with 2 observations. We should generate 3 more Sobol trials.
        experiment = get_experiment_with_observations([[1.0], [2.0]])
        # Mark the existing trials as manual to prevent them from counting for Sobol.
        # They'll still count for TC, since we use all trials in the experiment.
        for trial in experiment.trials.values():
            none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._model_key = "Manual"
        # Generate 5 trials and make sure they're from the correct nodes.
        run_trials_with_gs(experiment=experiment, gs=gs, num_trials=5)
        self.assertEqual(len(experiment.trials), 7)
        for trial in experiment.trials.values():
            model_key = none_throws(
                assert_is_instance(trial, Trial).generator_run
            )._model_key
            if trial.index < 2:
                self.assertEqual(model_key, "Manual")
            elif trial.index < 5:
                self.assertEqual(model_key, "Sobol")
            else:
                self.assertEqual(model_key, "BoTorch")
