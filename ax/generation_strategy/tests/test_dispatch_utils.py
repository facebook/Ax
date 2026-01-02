#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from itertools import product
from typing import Any

import torch
from ax.adapter.registry import Generators
from ax.adapter.transforms.log_y import LogY
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.generation_strategy.dispatch_utils import (
    _make_botorch_step,
    calculate_num_initialization_trials,
    choose_generation_strategy_legacy,
    DEFAULT_BAYESIAN_PARALLELISM,
)
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.transition_criterion import (
    MaxGenerationParallelism,
    MinTrials,
)
from ax.generators.random.sobol import SobolGenerator
from ax.generators.winsorization_config import WinsorizationConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_search_space,
    get_discrete_search_space,
    get_experiment,
    get_factorial_search_space,
    get_large_factorial_search_space,
    get_large_ordinal_search_space,
    get_search_space_with_choice_parameters,
    run_branin_experiment_with_generation_strategy,
)
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from pyre_extensions import assert_is_instance, none_throws


class TestDispatchUtils(TestCase):
    """Tests that dispatching utilities correctly select generation strategies."""

    @mock_botorch_optimize
    def test_choose_generation_strategy_legacy(self) -> None:
        expected_transform_configs = {}
        with self.subTest("GPEI"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space()
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
            expected_generator_kwargs: dict[str, Any] = {
                "torch_device": None,
                "transform_configs": expected_transform_configs,
                "acquisition_options": {"prune_irrelevant_parameters": False},
            }
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_kwargs,
                expected_generator_kwargs,
            )
            device = torch.device("cpu")
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(), torch_device=device
            )
            expected_generator_kwargs["torch_device"] = device
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_kwargs,
                expected_generator_kwargs,
            )
            run_branin_experiment_with_generation_strategy(
                generation_strategy=sobol_gpei
            )
        with self.subTest("max initialization trials"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                2,
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("min sobol trials"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                min_sobol_trials_observed=1,
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            # Extract min_trials_observed from MinTrials TC with only_in_statuses
            min_trials_observed = None
            for tc in sobol_gpei._nodes[0].transition_criteria:
                if isinstance(tc, MinTrials) and tc.only_in_statuses is not None:
                    min_trials_observed = tc.threshold
                    break
            self.assertEqual(min_trials_observed, 1)
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("num_initialization_trials > max_initialization_trials"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
                num_initialization_trials=3,
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                3,
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("num_initialization_trials > max_initialization_trials"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
                num_initialization_trials=3,
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                3,
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("MOO"):
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=[])
            )
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                optimization_config=optimization_config,
                simplify_parameter_changes=True,
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
            generator_kwargs = none_throws(
                sobol_gpei._nodes[1].generator_specs[0].generator_kwargs
            )
            self.assertEqual(
                set(generator_kwargs.keys()),
                {"torch_device", "transform_configs", "acquisition_options"},
            )
            self.assertTrue(
                generator_kwargs["acquisition_options"]["prune_irrelevant_parameters"]
            )
        with self.subTest("Sobol (we can try every option)"):
            sobol = choose_generation_strategy_legacy(
                search_space=get_factorial_search_space(), num_trials=1000
            )
            self.assertEqual(
                sobol._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(len(sobol._nodes), 1)
        with self.subTest("Sobol (because of too many categories)"):
            sobol_large = choose_generation_strategy_legacy(
                search_space=get_large_factorial_search_space()
            )
            self.assertEqual(
                sobol_large._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(len(sobol_large._nodes), 1)
        with self.subTest("Sobol (because of too many categories) with saasbo"):
            with self.assertLogs(
                choose_generation_strategy_legacy.__module__, logging.WARNING
            ) as logger:
                sobol_large = choose_generation_strategy_legacy(
                    search_space=get_large_factorial_search_space(), use_saasbo=True
                )
                self.assertTrue(
                    any(
                        "SAASBO is incompatible with Sobol" in output
                        for output in logger.output
                    ),
                    logger.output,
                )
            self.assertEqual(
                sobol_large._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(len(sobol_large._nodes), 1)
        with self.subTest("SOBOL due to too many unordered choices"):
            # Search space with more unordered choices than ordered parameters.
            sobol = choose_generation_strategy_legacy(
                search_space=get_search_space_with_choice_parameters(
                    num_ordered_parameters=5,
                    num_unordered_choices=100,
                )
            )
            self.assertEqual(
                sobol._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(len(sobol._nodes), 1)
        with self.subTest("GPEI with more unordered choices than ordered parameters"):
            # Search space with more unordered choices than ordered parameters.
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_search_space_with_choice_parameters(
                    num_ordered_parameters=5,
                    num_unordered_choices=10,
                )
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("GPEI despite many unordered 2-value parameters"):
            gs = choose_generation_strategy_legacy(
                search_space=get_large_factorial_search_space(
                    num_levels=2, num_parameters=10
                ),
            )
            self.assertEqual(
                gs._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                gs._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("GPEI-Batched"):
            sobol_gpei_batched = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei_batched._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                1,
            )
        with self.subTest("BO_MIXED (purely categorical)"):
            bo_mixed = choose_generation_strategy_legacy(
                search_space=get_factorial_search_space()
            )
            self.assertEqual(
                bo_mixed._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    bo_mixed._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                6,
            )
            self.assertEqual(
                bo_mixed._nodes[1].generator_specs[0].generator_enum,
                Generators.BO_MIXED,
            )
            expected_generator_kwargs = {
                "torch_device": None,
                "transform_configs": expected_transform_configs,
                "acquisition_options": {"prune_irrelevant_parameters": False},
            }
            self.assertEqual(
                bo_mixed._nodes[1].generator_specs[0].generator_kwargs,
                expected_generator_kwargs,
            )
        with self.subTest("BO_MIXED (mixed search space)"):
            ss = get_branin_search_space(with_choice_parameter=True)
            # pyre-fixme[16]: `Parameter` has no attribute `_is_ordered`.
            ss.parameters["x2"]._is_ordered = False
            bo_mixed_2 = choose_generation_strategy_legacy(search_space=ss)
            self.assertEqual(
                bo_mixed_2._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    bo_mixed_2._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            self.assertEqual(
                bo_mixed_2._nodes[1].generator_specs[0].generator_enum,
                Generators.BO_MIXED,
            )
            expected_generator_kwargs = {
                "torch_device": None,
                "transform_configs": expected_transform_configs,
                "acquisition_options": {"prune_irrelevant_parameters": False},
            }
            self.assertEqual(
                bo_mixed._nodes[1].generator_specs[0].generator_kwargs,
                expected_generator_kwargs,
            )
        with self.subTest("BO_MIXED (mixed multi-objective optimization)"):
            search_space = get_branin_search_space(with_choice_parameter=True)
            search_space.parameters["x2"]._is_ordered = False
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=[])
            )
            moo_mixed = choose_generation_strategy_legacy(
                search_space=search_space,
                optimization_config=optimization_config,
                simplify_parameter_changes=True,
            )
            self.assertEqual(
                moo_mixed._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                assert_is_instance(
                    moo_mixed._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            self.assertEqual(
                moo_mixed._nodes[1].generator_specs[0].generator_enum,
                Generators.BO_MIXED,
            )
            generator_kwargs = none_throws(
                moo_mixed._nodes[1].generator_specs[0].generator_kwargs
            )
            self.assertEqual(
                set(generator_kwargs.keys()),
                {"torch_device", "transform_configs", "acquisition_options"},
            )
            self.assertEqual(
                generator_kwargs["acquisition_options"]["prune_irrelevant_parameters"],
                True,
            )
        with self.subTest("SAASBO"):
            sobol_fullybayesian = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
                num_initialization_trials=3,
                use_saasbo=True,
                simplify_parameter_changes=True,
            )
            self.assertEqual(
                sobol_fullybayesian._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_fullybayesian._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                3,
            )
            bo_step = sobol_fullybayesian._nodes[1]
            self.assertEqual(
                bo_step.generator_specs[0].generator_enum, Generators.BOTORCH_MODULAR
            )
            model_config = (
                bo_step.generator_specs[0]
                .generator_kwargs["surrogate_spec"]
                .model_configs[0]
            )
            self.assertEqual(
                model_config.botorch_model_class, SaasFullyBayesianSingleTaskGP
            )
            self.assertEqual(
                generator_kwargs["acquisition_options"]["prune_irrelevant_parameters"],
                True,
            )
        with self.subTest("SAASBO MOO"):
            sobol_fullybayesianmoo = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
                num_initialization_trials=3,
                use_saasbo=True,
                optimization_config=MultiObjectiveOptimizationConfig(
                    objective=MultiObjective(objectives=[])
                ),
            )
            self.assertEqual(
                sobol_fullybayesianmoo._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_fullybayesianmoo._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                3,
            )
            bo_step = sobol_fullybayesianmoo._nodes[1]
            self.assertEqual(
                bo_step.generator_specs[0].generator_enum, Generators.BOTORCH_MODULAR
            )
            model_config = (
                bo_step.generator_specs[0]
                .generator_kwargs["surrogate_spec"]
                .model_configs[0]
            )
            self.assertEqual(
                model_config.botorch_model_class, SaasFullyBayesianSingleTaskGP
            )
        with self.subTest("SAASBO"):
            sobol_fullybayesian_large = choose_generation_strategy_legacy(
                search_space=get_large_ordinal_search_space(
                    n_ordinal_choice_parameters=5, n_continuous_range_parameters=10
                ),
                use_saasbo=True,
            )
            self.assertEqual(
                sobol_fullybayesian_large._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_fullybayesian_large._nodes[0].transition_criteria[0],
                    MinTrials,
                ).threshold,
                30,
            )
            bo_step = sobol_fullybayesian_large._nodes[1]
            self.assertEqual(
                bo_step.generator_specs[0].generator_enum, Generators.BOTORCH_MODULAR
            )
            model_config = (
                bo_step.generator_specs[0]
                .generator_kwargs["surrogate_spec"]
                .model_configs[0]
            )
            self.assertEqual(
                model_config.botorch_model_class, SaasFullyBayesianSingleTaskGP
            )

        with self.subTest("num_initialization_trials"):
            ss = get_large_factorial_search_space()
            for _, param in ss.parameters.items():
                param._is_ordered = True
            # 2 * len(ss.parameters) init trials are performed if num_trials is large
            gs_12_init_trials = choose_generation_strategy_legacy(
                search_space=ss, num_trials=100
            )
            self.assertEqual(
                gs_12_init_trials._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    gs_12_init_trials._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                12,
            )
            self.assertEqual(
                gs_12_init_trials._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
            # at least 5 initialization trials are performed
            gs_5_init_trials = choose_generation_strategy_legacy(
                search_space=ss, num_trials=0
            )
            self.assertEqual(
                gs_5_init_trials._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    gs_5_init_trials._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            self.assertEqual(
                gs_5_init_trials._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
            # avoid spending >20% of budget on initialization trials if there are
            # more than 5 initialization trials
            gs_6_init_trials = choose_generation_strategy_legacy(
                search_space=ss, num_trials=30
            )
            self.assertEqual(
                gs_6_init_trials._nodes[0].generator_specs[0].generator_enum,
                Generators.SOBOL,
            )
            self.assertEqual(
                assert_is_instance(
                    gs_6_init_trials._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                6,
            )
            self.assertEqual(
                gs_6_init_trials._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
        with self.subTest("suggested_model_override"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space()
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BOTORCH_MODULAR,
            )
            sobol_saasbo = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                suggested_model_override=Generators.SAASBO,
            )
            self.assertEqual(
                sobol_saasbo._nodes[1].generator_specs[0].generator_enum,
                Generators.SAASBO,
            )

    def test_make_botorch_step_extra(self) -> None:
        # Test parts of _make_botorch_step that are not directly exposed in
        # choose_generation_strategy.
        generator_kwargs = {
            "transforms": [LogY],
            "transform_configs": {"LogY": {"metrics": ["metric_1"]}},
        }
        with self.assertRaises(AssertionError):
            bo_step = _make_botorch_step(generator_kwargs=generator_kwargs)
        generator_kwargs = {"transforms": [LogY]}
        bo_step = _make_botorch_step(generator_kwargs=generator_kwargs)
        self.assertEqual(
            # pyre-fixme[16]: Currently, Pyre doesn't recognize that `Generation
            #  Step.__new__` actually returns a `GenerationNode`.
            none_throws(bo_step.generator_spec.generator_kwargs)["transforms"],
            [LogY],
        )
        self.assertEqual(
            none_throws(bo_step.generator_spec.generator_kwargs)["transform_configs"],
            {},
        )
        # With derelativize_with_raw_status_quo.
        bo_step = _make_botorch_step(
            generator_kwargs=generator_kwargs, derelativize_with_raw_status_quo=True
        )
        self.assertEqual(
            none_throws(bo_step.generator_spec.generator_kwargs)["transform_configs"],
            {
                "Derelativize": {"use_raw_status_quo": True},
                "Winsorize": {"derelativize_with_raw_status_quo": True},
                "BilogY": {"derelativize_with_raw_status_quo": True},
            },
        )

    def test_setting_random_seed(self) -> None:
        sobol = choose_generation_strategy_legacy(
            search_space=get_factorial_search_space(), random_seed=9
        )
        sobol.gen_single_trial(experiment=get_experiment(), n=1)
        # First model is actually an adapter, second is the Sobol engine.
        self.assertEqual(
            assert_is_instance(
                none_throws(sobol.adapter).generator, SobolGenerator
            ).seed,
            9,
        )

        with self.subTest("warns if use_saasbo is true"):
            with self.assertLogs(
                choose_generation_strategy_legacy.__module__, logging.WARNING
            ) as logger:
                sobol = choose_generation_strategy_legacy(
                    search_space=get_factorial_search_space(),
                    random_seed=9,
                    use_saasbo=True,
                )
                self.assertTrue(
                    any(
                        "SAASBO is incompatible with `BO_MIXED`" in output
                        for output in logger.output
                    ),
                    logger.output,
                )

    def test_enforce_sequential_optimization(self) -> None:
        with self.subTest("True"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space()
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            # Check that enforce_num_trials is True by verifying the MinTrials
            # criterion has block_gen_if_met=True
            node0_min_trials = assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            )
            self.assertTrue(node0_min_trials.block_gen_if_met)
            # Check that max_parallelism is set by verifying MaxGenerationParallelism
            # criterion exists on node 1
            node1_max_parallelism = [
                tc
                for tc in sobol_gpei._nodes[1].transition_criteria
                if isinstance(tc, MaxGenerationParallelism)
            ]
            self.assertTrue(len(node1_max_parallelism) > 0)
        with self.subTest("False"):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=get_branin_search_space(),
                enforce_sequential_optimization=False,
            )
            self.assertEqual(
                assert_is_instance(
                    sobol_gpei._nodes[0].transition_criteria[0], MinTrials
                ).threshold,
                5,
            )
            # Check that enforce_num_trials is False by verifying the MinTrials
            # criterion has block_gen_if_met=False
            node0_min_trials = assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            )
            self.assertFalse(node0_min_trials.block_gen_if_met)
            # Check that max_parallelism is None by verifying no
            # MaxGenerationParallelism criterion exists on node 1
            node1_max_parallelism = [
                tc
                for tc in sobol_gpei._nodes[1].transition_criteria
                if isinstance(tc, MaxGenerationParallelism)
            ]
            self.assertEqual(len(node1_max_parallelism), 0)
        with self.subTest("False and max_parallelism_override"):
            with self.assertLogs(
                choose_generation_strategy_legacy.__module__, logging.INFO
            ) as logger:
                choose_generation_strategy_legacy(
                    search_space=get_branin_search_space(),
                    enforce_sequential_optimization=False,
                    max_parallelism_override=5,
                )
                self.assertTrue(
                    any(
                        "other max parallelism settings will be ignored" in output
                        for output in logger.output
                    ),
                    logger.output,
                )
        with self.subTest("False and max_parallelism_cap"):
            with self.assertLogs(
                choose_generation_strategy_legacy.__module__, logging.INFO
            ) as logger:
                choose_generation_strategy_legacy(
                    search_space=get_branin_search_space(),
                    enforce_sequential_optimization=False,
                    max_parallelism_cap=5,
                )
                self.assertTrue(
                    any(
                        "other max parallelism settings will be ignored" in output
                        for output in logger.output
                    ),
                    logger.output,
                )
        with self.subTest("False and max_parallelism_override and max_parallelism_cap"):
            with self.assertRaisesRegex(
                ValueError,
                (
                    "If `max_parallelism_override` specified, cannot also apply "
                    "`max_parallelism_cap`."
                ),
            ):
                choose_generation_strategy_legacy(
                    search_space=get_branin_search_space(),
                    enforce_sequential_optimization=False,
                    max_parallelism_override=5,
                    max_parallelism_cap=5,
                )

    def test_max_parallelism_override(self) -> None:
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertTrue(
            all(self._get_max_parallelism(s) == 10 for s in sobol_gpei._nodes)
        )

    def test_winsorization(self) -> None:
        winsorized = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            winsorization_config=WinsorizationConfig(upper_quantile_margin=2),
        )
        tc = none_throws(winsorized._nodes[1].generator_specs[0].generator_kwargs).get(
            "transform_configs"
        )
        self.assertIn("Winsorize", tc)
        self.assertDictEqual(
            tc["Winsorize"],
            {
                "winsorization_config": WinsorizationConfig(
                    lower_quantile_margin=0.0,
                    upper_quantile_margin=2,
                    lower_boundary=None,
                    upper_boundary=None,
                )
            },
        )
        # With derelativize_with_raw_status_quo.
        winsorized = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            derelativize_with_raw_status_quo=True,
        )
        tc = none_throws(winsorized._nodes[1].generator_specs[0].generator_kwargs).get(
            "transform_configs"
        )
        self.assertIn(
            "Winsorize",
            tc,
        )
        self.assertDictEqual(
            tc["Winsorize"],
            {"derelativize_with_raw_status_quo": True},
        )
        self.assertIn(
            "Derelativize",
            tc,
        )
        self.assertDictEqual(tc["Derelativize"], {"use_raw_status_quo": True})

    def test_num_trials(self) -> None:
        ss = get_discrete_search_space()
        with self.subTest(
            "with budget that is lower than exhaustive, BayesOpt is used"
        ):
            sobol_gpei = choose_generation_strategy_legacy(
                search_space=ss, num_trials=23
            )
            self.assertEqual(
                sobol_gpei._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(
                sobol_gpei._nodes[1].generator_specs[0].generator_enum,
                Generators.BO_MIXED,
            )
        with self.subTest("with budget that is exhaustive, Sobol is used"):
            sobol = choose_generation_strategy_legacy(search_space=ss, num_trials=36)
            self.assertEqual(
                sobol._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(len(sobol._nodes), 1)
        with self.subTest("with budget that is exhaustive and use_saasbo, it warns"):
            with self.assertLogs(
                choose_generation_strategy_legacy.__module__, logging.WARNING
            ) as logger:
                sobol = choose_generation_strategy_legacy(
                    search_space=ss,
                    num_trials=36,
                    use_saasbo=True,
                )
                self.assertTrue(
                    any(
                        "SAASBO is incompatible with Sobol" in output
                        for output in logger.output
                    ),
                    logger.output,
                )
            self.assertEqual(
                sobol._nodes[0].generator_specs[0].generator_enum, Generators.SOBOL
            )
            self.assertEqual(len(sobol._nodes), 1)

    def test_use_batch_trials(self) -> None:
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(), use_batch_trials=True
        )
        self.assertEqual(
            assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            ).threshold,
            1,
        )

    def test_fixed_num_initialization_trials(self) -> None:
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
        )
        self.assertEqual(
            assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            ).threshold,
            3,
        )

    def _get_max_parallelism(self, node: GenerationNode) -> int | None:
        """Helper to extract max_parallelism from transition criteria."""
        for tc in node.transition_criteria:
            if isinstance(tc, MaxGenerationParallelism):
                return tc.threshold
        return None

    def test_max_parallelism_adjustments(self) -> None:
        # No adjustment.
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space()
        )
        self.assertIsNone(self._get_max_parallelism(sobol_gpei._nodes[0]))
        self.assertEqual(
            self._get_max_parallelism(sobol_gpei._nodes[1]),
            DEFAULT_BAYESIAN_PARALLELISM,
        )
        # Impose a cap of 1 on max parallelism for all steps.
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(), max_parallelism_cap=1
        )
        self.assertEqual(
            self._get_max_parallelism(sobol_gpei._nodes[0]),
            1,
        )
        self.assertEqual(
            self._get_max_parallelism(sobol_gpei._nodes[1]),
            1,
        )
        # Disable enforcing max parallelism for all steps.
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(), max_parallelism_override=-1
        )
        self.assertIsNone(self._get_max_parallelism(sobol_gpei._nodes[0]))
        self.assertIsNone(self._get_max_parallelism(sobol_gpei._nodes[1]))
        # Override max parallelism for all steps.
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertEqual(self._get_max_parallelism(sobol_gpei._nodes[0]), 10)
        self.assertEqual(self._get_max_parallelism(sobol_gpei._nodes[1]), 10)

    def test_set_should_deduplicate(self) -> None:
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
        )
        self.assertListEqual(
            [s.should_deduplicate for s in sobol_gpei._nodes], [False] * 2
        )
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
            should_deduplicate=True,
        )
        self.assertListEqual(
            [s.should_deduplicate for s in sobol_gpei._nodes], [True] * 2
        )

    def test_setting_experiment_attribute(self) -> None:
        exp = get_experiment()
        gs = choose_generation_strategy_legacy(
            search_space=exp.search_space, experiment=exp
        )
        self.assertEqual(gs._experiment, exp)

    def test_setting_num_completed_initialization_trials(self) -> None:
        default_initialization_num_trials = 5
        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space()
        )

        self.assertEqual(
            assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            ).threshold,
            default_initialization_num_trials,
        )

        sobol_gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            num_completed_initialization_trials=2,
        )

        # With the new use_all_trials_in_exp=True behavior, the step will still
        # be configured for the total number of initialization trials, and the
        # transition criteria will automatically account for existing trials.
        self.assertEqual(
            assert_is_instance(
                sobol_gpei._nodes[0].transition_criteria[0], MinTrials
            ).threshold,
            default_initialization_num_trials,
        )

        # Verify that use_all_trials_in_exp is set to True for the Sobol step
        # so it accounts for existing trials automatically
        first_transition_criterion = assert_is_instance(
            sobol_gpei._nodes[0].transition_criteria[0], MinTrials
        )
        self.assertTrue(first_transition_criterion.use_all_trials_in_exp)

        # Sobol step shouldn't be created if there are enough completed trials.
        gpei = choose_generation_strategy_legacy(
            search_space=get_branin_search_space(),
            num_completed_initialization_trials=5,
        )
        self.assertEqual(len(gpei._nodes), 1)

    def test_calculate_num_initialization_trials(self) -> None:
        with self.subTest("one trial for batch trials"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=2,
                    num_trials=None,
                    use_batch_trials=True,
                ),
                1,
            )

        with self.subTest("num_trials is unset, small exp"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=2,
                    num_trials=None,
                    use_batch_trials=False,
                ),
                5,
            )

        with self.subTest("num_trials is unset, large exp"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=10,
                    num_trials=None,
                    use_batch_trials=False,
                ),
                20,
            )

        with self.subTest("many trials"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=10,
                    num_trials=200,
                    use_batch_trials=False,
                ),
                20,
            )

        with self.subTest("limited trials"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=10,
                    num_trials=50,
                    use_batch_trials=False,
                ),
                10,
            )

        with self.subTest("few trials"):
            self.assertEqual(
                calculate_num_initialization_trials(
                    num_tunable_parameters=10,
                    num_trials=10,
                    use_batch_trials=False,
                ),
                5,
            )

    @mock_botorch_optimize
    def test_saas_options(self) -> None:
        for jit_compile, use_input_warping, disable_progbar in product(
            (True, False), (True, False), (True, False)
        ):
            with self.subTest(
                f"jit_compile: {jit_compile}, use_input_warping: {use_input_warping},"
                f" disable_progbar: {disable_progbar}"
            ):
                sobol_saasbo = choose_generation_strategy_legacy(
                    search_space=get_branin_search_space(),
                    jit_compile=jit_compile,
                    disable_progbar=disable_progbar,
                    use_saasbo=True,
                    use_input_warping=use_input_warping,
                )
                self.assertEqual(
                    sobol_saasbo._nodes[0].generator_specs[0].generator_enum,
                    Generators.SOBOL,
                )
                self.assertNotIn(
                    "jit_compile",
                    none_throws(
                        sobol_saasbo._nodes[0].generator_specs[0].generator_kwargs
                    ),
                )
                self.assertNotIn(
                    "disable_progbar",
                    none_throws(
                        sobol_saasbo._nodes[0].generator_specs[0].generator_kwargs
                    ),
                )
                bo_step = sobol_saasbo._nodes[1]
                self.assertEqual(
                    bo_step.generator_specs[0].generator_enum,
                    Generators.BOTORCH_MODULAR,
                )
                model_config = (
                    bo_step.generator_specs[0]
                    .generator_kwargs["surrogate_spec"]
                    .model_configs[0]
                )
                self.assertIs(
                    model_config.botorch_model_class, SaasFullyBayesianSingleTaskGP
                )
                self.assertEqual(model_config.mll_options["jit_compile"], jit_compile)
                self.assertEqual(
                    model_config.mll_options["disable_progbar"], disable_progbar
                )
                self.assertEqual(
                    model_config.model_options["use_input_warping"], use_input_warping
                )
                run_branin_experiment_with_generation_strategy(
                    generation_strategy=sobol_saasbo,
                )

    @mock_botorch_optimize
    def test_non_saasbo_discards_irrelevant_generator_kwargs(self) -> None:
        for jit_compile, use_input_warping, disable_progbar in product(
            (True, False), (True, False), (True, False)
        ):
            with self.subTest(str(jit_compile)):
                gp_saasbo = choose_generation_strategy_legacy(
                    search_space=get_branin_search_space(),
                    jit_compile=jit_compile,
                    disable_progbar=disable_progbar,
                    use_saasbo=False,
                    use_input_warping=use_input_warping,
                )
                self.assertEqual(len(gp_saasbo._nodes), 2)
                self.assertEqual(
                    gp_saasbo._nodes[0].generator_specs[0].generator_enum,
                    Generators.SOBOL,
                )
                self.assertNotIn(
                    "jit_compile",
                    none_throws(
                        gp_saasbo._nodes[0].generator_specs[0].generator_kwargs
                    ),
                )
                bo_step = gp_saasbo._nodes[1]
                self.assertEqual(
                    bo_step.generator_specs[0].generator_enum,
                    Generators.BOTORCH_MODULAR,
                )

                generator_kwargs = none_throws(
                    bo_step.generator_specs[0].generator_kwargs
                )
                for k in ("jit_compile", "disable_progbar", "use_input_warping"):
                    self.assertNotIn(k, generator_kwargs)
                self.assertNotIn("surrogate_spec", generator_kwargs)
                run_branin_experiment_with_generation_strategy(
                    generation_strategy=gp_saasbo,
                )
