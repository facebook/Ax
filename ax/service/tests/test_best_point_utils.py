#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.service.utils.best_point import (
    get_best_parameters,
    get_best_raw_objective_point,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_metric


class TestBestPointUtils(TestCase):
    """Testing the best point utilities functionality that is not tested in
    main `AxClient` testing suite (`TestSErviceAPI`)."""

    def test_best_raw_objective_point(self):
        exp = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        opt_conf = exp.optimization_config.clone()
        opt_conf.objective.metric._name = "not_branin"
        with self.assertRaisesRegex(ValueError, "No data has been logged"):
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_scalarized(self):
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(metrics=[get_branin_metric()], minimize=False)
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})

    def test_best_raw_objective_point_scalarized_multi(self):
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(
                metrics=[get_branin_metric(), get_branin_metric()],
                weights=[0.1, -0.9],
                minimize=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})
