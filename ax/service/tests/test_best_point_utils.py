#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import (
    get_best_parameters,
    get_best_raw_objective_point,
)
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import hartmann6
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

    def test_best_raw_objective_point_unsatisfiable(self):
        ax_client = AxClient()

        ax_client.create_experiment(
            name="hartmann_test_experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {
                    "name": "x2",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "x3",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "x5",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [0.0, 1.0],
                },
            ],
            objective_name="hartmann6",
            parameter_constraints=["x1 + x2 <= 2.0"],
            outcome_constraints=["l2norm <= -99999"],  # unsatisfiable
        )

        ax_client.experiment.optimization_config.outcome_constraints.append(
            OutcomeConstraint(
                metric=Hartmann6Metric(
                    name="relative",
                    param_names=ax_client.experiment.parameters.keys(),
                    lower_is_better=False,
                ),
                op=ComparisonOp.LEQ,
                bound=0,
                relative=True,
            )
        )

        def evaluate(parameters):
            x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
            return {
                "hartmann6": (hartmann6(x), 0.0),
                "l2norm": (np.sqrt((x ** 2).sum()), 0.0),
            }

        for _ in range(10):
            parameters, trial_index = ax_client.get_next_trial()

            ax_client.complete_trial(
                trial_index=trial_index, raw_data=evaluate(parameters)
            )

        with self.assertLogs(
            logger="ax.service.utils.best_point",
            level="WARN",
        ) as lg:
            self.assertIsNone(ax_client.get_best_parameters())
            self.assertTrue(
                any(
                    "Filtering out infeasible arms based on relative outcome "
                    + "constraints is not yet supported."
                    in warning
                    for warning in lg.output
                ),
                msg=lg.output,
            )

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
