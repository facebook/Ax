#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.service.utils.best_point import (
    get_best_parameters,
    get_best_raw_objective_point,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


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
