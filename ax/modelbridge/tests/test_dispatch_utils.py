#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from typing import Any, Dict
from unittest import mock

import torch
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.dispatch_utils import (
    calculate_num_initialization_trials,
    choose_generation_strategy,
    DEFAULT_BAYESIAN_PARALLELISM,
)
from ax.modelbridge.factory import Cont_X_trans, Y_trans
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.winsorize import Winsorize
from ax.models.winsorization_config import WinsorizationConfig
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_search_space,
    get_discrete_search_space,
    get_experiment,
    get_factorial_search_space,
    get_large_factorial_search_space,
    get_large_ordinal_search_space,
)


class TestDispatchUtils(TestCase):
    """Tests that dispatching utilities correctly select generation strategies."""

    def test_choose_generation_strategy(self) -> None:
        expected_transforms = [Winsorize] + Cont_X_trans + Y_trans
        expected_transform_configs = {
            "Winsorize": {"derelativize_with_raw_status_quo": False},
            "Derelativize": {"use_raw_status_quo": False},
        }
        with self.subTest("GPEI"):
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space()
            )
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
            self.assertEqual(sobol_gpei._steps[1].model, Models.GPEI)
            expected_model_kwargs: Dict[str, Any] = {
                "torch_device": None,
                "transforms": expected_transforms,
                "transform_configs": expected_transform_configs,
            }
            self.assertEqual(sobol_gpei._steps[1].model_kwargs, expected_model_kwargs)
            device = torch.device("cpu")
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                verbose=True,
                torch_device=device,
            )
            expected_model_kwargs["torch_device"] = device
            self.assertEqual(sobol_gpei._steps[1].model_kwargs, expected_model_kwargs)
        with self.subTest("max initialization trials"):
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
            )
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[0].num_trials, 2)
            self.assertEqual(sobol_gpei._steps[1].model, Models.GPEI)
        with self.subTest("num_initialization_trials > max_initialization_trials"):
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
                num_initialization_trials=3,
            )
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[0].num_trials, 3)
            self.assertEqual(sobol_gpei._steps[1].model, Models.GPEI)
        with self.subTest("num_initialization_trials > max_initialization_trials"):
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                max_initialization_trials=2,
                num_initialization_trials=3,
            )
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[0].num_trials, 3)
            self.assertEqual(sobol_gpei._steps[1].model, Models.GPEI)
        with self.subTest("MOO"):
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=[])
            )
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                optimization_config=optimization_config,
            )
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
            self.assertEqual(sobol_gpei._steps[1].model, Models.MOO)
            model_kwargs = not_none(sobol_gpei._steps[1].model_kwargs)
            self.assertEqual(
                list(model_kwargs.keys()),
                ["torch_device", "transforms", "transform_configs"],
            )
            self.assertGreater(len(model_kwargs["transforms"]), 0)
        with self.subTest("Sobol (we can try every option)"):
            sobol = choose_generation_strategy(
                search_space=get_factorial_search_space(), num_trials=1000
            )
            self.assertEqual(sobol._steps[0].model, Models.SOBOL)
            self.assertEqual(len(sobol._steps), 1)
        with self.subTest("Sobol (because of too many categories)"):
            sobol_large = choose_generation_strategy(
                search_space=get_large_factorial_search_space(), verbose=True
            )
            self.assertEqual(sobol_large._steps[0].model, Models.SOBOL)
            self.assertEqual(len(sobol_large._steps), 1)
        with self.subTest("Sobol (because of too many categories) with saasbo"):
            with self.assertLogs(
                choose_generation_strategy.__module__, logging.WARNING
            ) as logger:
                sobol_large = choose_generation_strategy(
                    search_space=get_large_factorial_search_space(),
                    verbose=True,
                    use_saasbo=True,
                )
                self.assertTrue(
                    any(
                        "SAASBO is incompatible with Sobol" in output
                        for output in logger.output
                    ),
                    logger.output,
                )
            self.assertEqual(sobol_large._steps[0].model, Models.SOBOL)
            self.assertEqual(len(sobol_large._steps), 1)
        with self.subTest("GPEI despite many unordered 2-value parameters"):
            gs = choose_generation_strategy(
                search_space=get_large_factorial_search_space(
                    num_levels=2, num_parameters=10
                ),
            )
            self.assertEqual(gs._steps[0].model, Models.SOBOL)
            self.assertEqual(gs._steps[1].model, Models.GPEI)
        with self.subTest("GPEI-Batched"):
            sobol_gpei_batched = choose_generation_strategy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
            )
            self.assertEqual(sobol_gpei_batched._steps[0].num_trials, 1)
        with self.subTest("BO_MIXED (purely categorical)"):
            bo_mixed = choose_generation_strategy(
                search_space=get_factorial_search_space()
            )
            self.assertEqual(bo_mixed._steps[0].model, Models.SOBOL)
            self.assertEqual(bo_mixed._steps[0].num_trials, 6)
            self.assertEqual(bo_mixed._steps[1].model, Models.BO_MIXED)
            expected_model_kwargs = {
                "torch_device": None,
                "transforms": expected_transforms,
                "transform_configs": expected_transform_configs,
            }
            self.assertEqual(bo_mixed._steps[1].model_kwargs, expected_model_kwargs)
        with self.subTest("BO_MIXED (mixed search space)"):
            ss = get_branin_search_space(with_choice_parameter=True)
            # pyre-fixme[16]: `Parameter` has no attribute `_is_ordered`.
            ss.parameters["x2"]._is_ordered = False
            bo_mixed_2 = choose_generation_strategy(search_space=ss)
            self.assertEqual(bo_mixed_2._steps[0].model, Models.SOBOL)
            self.assertEqual(bo_mixed_2._steps[0].num_trials, 5)
            self.assertEqual(bo_mixed_2._steps[1].model, Models.BO_MIXED)
            expected_model_kwargs = {
                "torch_device": None,
                "transforms": expected_transforms,
                "transform_configs": expected_transform_configs,
            }
            self.assertEqual(bo_mixed._steps[1].model_kwargs, expected_model_kwargs)
        with self.subTest("BO_MIXED (mixed multi-objective optimization)"):
            search_space = get_branin_search_space(with_choice_parameter=True)
            search_space.parameters["x2"]._is_ordered = False
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(objectives=[])
            )
            moo_mixed = choose_generation_strategy(
                search_space=search_space, optimization_config=optimization_config
            )
            self.assertEqual(moo_mixed._steps[0].model, Models.SOBOL)
            self.assertEqual(moo_mixed._steps[0].num_trials, 5)
            self.assertEqual(moo_mixed._steps[1].model, Models.BO_MIXED)
            model_kwargs = not_none(moo_mixed._steps[1].model_kwargs)
            self.assertEqual(
                list(model_kwargs.keys()),
                ["torch_device", "transforms", "transform_configs"],
            )
            self.assertGreater(len(model_kwargs["transforms"]), 0)
        with self.subTest("SAASBO"):
            sobol_fullybayesian = choose_generation_strategy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
                num_initialization_trials=3,
                use_saasbo=True,
            )
            self.assertEqual(sobol_fullybayesian._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_fullybayesian._steps[0].num_trials, 3)
            self.assertEqual(sobol_fullybayesian._steps[1].model, Models.FULLYBAYESIAN)
            self.assertTrue(
                not_none(sobol_fullybayesian._steps[1].model_kwargs)["verbose"]
            )
        with self.subTest("SAASBO MOO"):
            sobol_fullybayesianmoo = choose_generation_strategy(
                search_space=get_branin_search_space(),
                use_batch_trials=True,
                num_initialization_trials=3,
                use_saasbo=True,
                optimization_config=MultiObjectiveOptimizationConfig(
                    objective=MultiObjective(objectives=[])
                ),
            )
            self.assertEqual(sobol_fullybayesianmoo._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_fullybayesianmoo._steps[0].num_trials, 3)
            self.assertEqual(
                sobol_fullybayesianmoo._steps[1].model,
                Models.FULLYBAYESIANMOO,
            )
            self.assertTrue(
                not_none(sobol_fullybayesianmoo._steps[1].model_kwargs)["verbose"]
            )
        with self.subTest("SAASBO"):
            sobol_fullybayesian_large = choose_generation_strategy(
                search_space=get_large_ordinal_search_space(
                    n_ordinal_choice_parameters=5, n_continuous_range_parameters=10
                ),
                use_saasbo=True,
            )
            self.assertEqual(sobol_fullybayesian_large._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_fullybayesian_large._steps[0].num_trials, 30)
            self.assertEqual(
                sobol_fullybayesian_large._steps[1].model,
                Models.FULLYBAYESIAN,
            )
            self.assertTrue(
                not_none(sobol_fullybayesian_large._steps[1].model_kwargs)["verbose"]
            )
        with self.subTest("num_initialization_trials"):
            ss = get_large_factorial_search_space()
            for _, param in ss.parameters.items():
                param._is_ordered = True
            # 2 * len(ss.parameters) init trials are performed if num_trials is large
            gs_12_init_trials = choose_generation_strategy(
                search_space=ss, num_trials=100
            )
            self.assertEqual(gs_12_init_trials._steps[0].model, Models.SOBOL)
            self.assertEqual(gs_12_init_trials._steps[0].num_trials, 12)
            self.assertEqual(gs_12_init_trials._steps[1].model, Models.GPEI)
            # at least 5 initialization trials are performed
            gs_5_init_trials = choose_generation_strategy(search_space=ss, num_trials=0)
            self.assertEqual(gs_5_init_trials._steps[0].model, Models.SOBOL)
            self.assertEqual(gs_5_init_trials._steps[0].num_trials, 5)
            self.assertEqual(gs_5_init_trials._steps[1].model, Models.GPEI)
            # avoid spending >20% of budget on initialization trials if there are
            # more than 5 initialization trials
            gs_6_init_trials = choose_generation_strategy(
                search_space=ss, num_trials=30
            )
            self.assertEqual(gs_6_init_trials._steps[0].model, Models.SOBOL)
            self.assertEqual(gs_6_init_trials._steps[0].num_trials, 6)
            self.assertEqual(gs_6_init_trials._steps[1].model, Models.GPEI)

    def test_disable_progbar(self) -> None:
        for disable_progbar in (True, False):
            with self.subTest(str(disable_progbar)):
                sobol_saasbo = choose_generation_strategy(
                    search_space=get_branin_search_space(),
                    disable_progbar=disable_progbar,
                    use_saasbo=True,
                )
                self.assertEqual(sobol_saasbo._steps[0].model, Models.SOBOL)
                self.assertNotIn(
                    "disable_progbar",
                    not_none(sobol_saasbo._steps[0].model_kwargs),
                )
                self.assertEqual(sobol_saasbo._steps[1].model, Models.FULLYBAYESIAN)
                self.assertEqual(
                    not_none(sobol_saasbo._steps[1].model_kwargs)["disable_progbar"],
                    disable_progbar,
                )

    def test_disable_progbar_for_non_saasbo_discards_the_model_kwarg(self) -> None:
        for disable_progbar in (True, False):
            with self.subTest(str(disable_progbar)):
                gp_saasbo = choose_generation_strategy(
                    search_space=get_branin_search_space(),
                    disable_progbar=disable_progbar,
                    use_saasbo=False,
                )
                self.assertEqual(len(gp_saasbo._steps), 2)
                self.assertEqual(gp_saasbo._steps[0].model, Models.SOBOL)
                self.assertNotIn(
                    "disable_progbar",
                    not_none(gp_saasbo._steps[0].model_kwargs),
                )
                self.assertEqual(gp_saasbo._steps[1].model, Models.GPEI)
                self.assertNotIn(
                    "disable_progbar",
                    not_none(gp_saasbo._steps[1].model_kwargs),
                )

    def test_setting_random_seed(self) -> None:
        sobol = choose_generation_strategy(
            search_space=get_factorial_search_space(), random_seed=9
        )
        sobol.gen(experiment=get_experiment(), n=1)
        # First model is actually a bridge, second is the Sobol engine.
        self.assertEqual(not_none(sobol.model).model.seed, 9)

        with self.subTest("warns if use_saasbo is true"):
            with self.assertLogs(
                choose_generation_strategy.__module__, logging.WARNING
            ) as logger:
                sobol = choose_generation_strategy(
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
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space()
            )
            self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
            self.assertTrue(sobol_gpei._steps[0].enforce_num_trials)
            self.assertIsNotNone(sobol_gpei._steps[1].max_parallelism)
        with self.subTest("False"):
            sobol_gpei = choose_generation_strategy(
                search_space=get_branin_search_space(),
                enforce_sequential_optimization=False,
            )
            self.assertEqual(sobol_gpei._steps[0].num_trials, 5)
            self.assertFalse(sobol_gpei._steps[0].enforce_num_trials)
            self.assertIsNone(sobol_gpei._steps[1].max_parallelism)
        with self.subTest("False and max_parallelism_override"):
            with self.assertLogs(
                choose_generation_strategy.__module__, logging.INFO
            ) as logger:
                choose_generation_strategy(
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
                choose_generation_strategy.__module__, logging.INFO
            ) as logger:
                choose_generation_strategy(
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
                choose_generation_strategy(
                    search_space=get_branin_search_space(),
                    enforce_sequential_optimization=False,
                    max_parallelism_override=5,
                    max_parallelism_cap=5,
                )

    def test_max_parallelism_override(self) -> None:
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertTrue(all(s.max_parallelism == 10 for s in sobol_gpei._steps))

    def test_winsorization(self) -> None:
        winsorized = choose_generation_strategy(
            search_space=get_branin_search_space(),
            winsorization_config=WinsorizationConfig(upper_quantile_margin=2),
        )
        tc = not_none(winsorized._steps[1].model_kwargs).get("transform_configs")
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
        self.assertIn("Derelativize", tc)
        self.assertDictEqual(tc["Derelativize"], {"use_raw_status_quo": False})

        winsorized = choose_generation_strategy(
            search_space=get_branin_search_space(),
            derelativize_with_raw_status_quo=True,
        )
        tc = not_none(winsorized._steps[1].model_kwargs).get("transform_configs")
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

    def test_no_winzorization_wins(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            unwinsorized = choose_generation_strategy(
                search_space=get_branin_search_space(),
                winsorization_config=WinsorizationConfig(upper_quantile_margin=2),
                no_winsorization=True,
            )
            self.assertEqual(len(w), 1)
            self.assertIn("Not winsorizing", str(w[-1].message))

        self.assertNotIn(
            # transform_configs would have "Winsorize" if it existed
            "transform_configs",
            set(not_none(unwinsorized._steps[1].model_kwargs)),
        )

    def test_num_trials(self) -> None:
        ss = get_discrete_search_space()
        with self.subTest(
            "with budget that is lower than exhaustive, BayesOpt is used"
        ):
            sobol_gpei = choose_generation_strategy(search_space=ss, num_trials=23)
            self.assertEqual(sobol_gpei._steps[0].model, Models.SOBOL)
            self.assertEqual(sobol_gpei._steps[1].model, Models.BO_MIXED)
        with self.subTest("with budget that is exhaustive, Sobol is used"):
            sobol = choose_generation_strategy(search_space=ss, num_trials=36)
            self.assertEqual(sobol._steps[0].model, Models.SOBOL)
            self.assertEqual(len(sobol._steps), 1)
        with self.subTest("with budget that is exhaustive and use_saasbo, it warns"):
            with self.assertLogs(
                choose_generation_strategy.__module__, logging.WARNING
            ) as logger:
                sobol = choose_generation_strategy(
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
            self.assertEqual(sobol._steps[0].model, Models.SOBOL)
            self.assertEqual(len(sobol._steps), 1)

    def test_use_batch_trials(self) -> None:
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), use_batch_trials=True
        )
        self.assertEqual(sobol_gpei._steps[0].num_trials, 1)

    def test_fixed_num_initialization_trials(self) -> None:
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
        )
        self.assertEqual(sobol_gpei._steps[0].num_trials, 3)

    def test_max_parallelism_adjustments(self) -> None:
        # No adjustment.
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())
        self.assertIsNone(sobol_gpei._steps[0].max_parallelism)
        self.assertEqual(
            sobol_gpei._steps[1].max_parallelism, DEFAULT_BAYESIAN_PARALLELISM
        )
        # Impose a cap of 1 on max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_cap=1
        )
        self.assertEqual(
            sobol_gpei._steps[0].max_parallelism,
            1,
        )
        self.assertEqual(
            sobol_gpei._steps[1].max_parallelism,
            1,
        )
        # Disable enforcing max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=-1
        )
        self.assertIsNone(sobol_gpei._steps[0].max_parallelism)
        self.assertIsNone(sobol_gpei._steps[1].max_parallelism)
        # Override max parallelism for all steps.
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(), max_parallelism_override=10
        )
        self.assertEqual(sobol_gpei._steps[0].max_parallelism, 10)
        self.assertEqual(sobol_gpei._steps[1].max_parallelism, 10)

    def test_set_should_deduplicate(self) -> None:
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
        )
        self.assertListEqual(
            [s.should_deduplicate for s in sobol_gpei._steps], [False] * 2
        )
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            use_batch_trials=True,
            num_initialization_trials=3,
            should_deduplicate=True,
        )
        self.assertListEqual(
            [s.should_deduplicate for s in sobol_gpei._steps], [True] * 2
        )

    def test_setting_experiment_attribute(self) -> None:
        exp = get_experiment()
        gs = choose_generation_strategy(search_space=exp.search_space, experiment=exp)
        self.assertEqual(gs._experiment, exp)

    def test_setting_num_completed_initialization_trials(self) -> None:
        default_initialization_num_trials = 5
        sobol_gpei = choose_generation_strategy(search_space=get_branin_search_space())

        self.assertEqual(
            sobol_gpei._steps[0].num_trials, default_initialization_num_trials
        )

        num_completed_initialization_trials = 2
        sobol_gpei = choose_generation_strategy(
            search_space=get_branin_search_space(),
            num_completed_initialization_trials=num_completed_initialization_trials,
        )

        self.assertEqual(
            sobol_gpei._steps[0].num_trials,
            default_initialization_num_trials - num_completed_initialization_trials,
        )

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

    def test_use_update(self) -> None:
        search_space = get_branin_search_space()
        # No experiment, no SAAS, default to False.
        gs = choose_generation_strategy(search_space=search_space)
        self.assertFalse(gs._steps[1].use_update)
        # Pass in True.
        gs = choose_generation_strategy(search_space=search_space, use_update=True)
        self.assertTrue(gs._steps[1].use_update)
        # With experiment without any metrics available while running.
        experiment = get_branin_experiment()
        with mock.patch.object(
            experiment.metrics["branin"],
            "is_available_while_running",
            return_value=False,
        ):
            # No SAAS, default to False.
            gs = choose_generation_strategy(
                search_space=search_space, experiment=experiment
            )
            self.assertFalse(gs._steps[1].use_update)
            # SAAS, default to True.
            gs = choose_generation_strategy(
                search_space=search_space, experiment=experiment, use_saasbo=True
            )
            self.assertTrue(gs._steps[1].use_update)
            # SAAS and pass in False.
            gs = choose_generation_strategy(
                search_space=search_space,
                experiment=experiment,
                use_saasbo=True,
                use_update=False,
            )
            self.assertFalse(gs._steps[1].use_update)
        # SAAS with metrics available while running.
        gs = choose_generation_strategy(
            search_space=search_space, experiment=experiment, use_saasbo=True
        )
        # Default to False.
        self.assertFalse(gs._steps[1].use_update)
        # Error with True.
        with self.assertRaisesRegex(UnsupportedError, "use_update"):
            choose_generation_strategy(
                search_space=search_space, experiment=experiment, use_update=True
            )
