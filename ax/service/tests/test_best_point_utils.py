#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.cross_validation import AssessModelFitResult
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

    def test_best_from_model_predictions(self):
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
        )

        def evaluate(parameters):
            x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
            return {
                "hartmann6": (hartmann6(x), 0.0),
                "l2norm": (np.sqrt((x ** 2).sum()), 0.0),
            }

        for _ in range(12):  # These will all be SOBOL runs
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=evaluate(parameters)
            )
        with patch.object(
            ArrayModelBridge,
            "model_best_point",
            return_value=(
                (
                    Arm(
                        name="6_0",
                        parameters={
                            "x1": 0.96247882489115,
                            "x2": 0.051644640043377876,
                            "x3": 0.6749767670407891,
                            "x4": 0.6150536900386214,
                            "x5": 0.059872522950172424,
                            "x6": 0.013009095564484596,
                        },
                    ),
                    (
                        {
                            "l2norm": 1.329161904140509,
                            "hartmann6": -0.0021285272846029435,
                        },
                        {
                            "l2norm": {
                                "l2norm": 8.021127869460997e-08,
                                "hartmann6": 0.0,
                            },
                            "hartmann6": {
                                "l2norm": 0.0,
                                "hartmann6": 1.6449112385323933e-07,
                            },
                        },
                    ),
                )
            ),
        ) as mock_model_best_point, self.assertLogs(
            logger="ax.service.utils.best_point", level="WARN"
        ) as lg:
            # Because refitting after final run is only supported for
            # ArrayModelBridge we do not expect model_best_point to be called here
            self.assertIsNotNone(ax_client.get_best_parameters())
            mock_model_best_point.assert_not_called()

            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=evaluate(parameters)
            )

            # Test bad model fit causes function to resort back to raw data
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={},
                    bad_fit_metrics_to_fisher_score={
                        "hartmann6": 0,
                        "l2norm": 0,
                    },
                ),
            ):
                self.assertIsNotNone(ax_client.get_best_parameters())
                self.assertTrue(
                    any("Model fit is poor" in warning for warning in lg.output),
                    msg=lg.output,
                )

            # Test model best point is used when fit is good
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={
                        "hartmann6": 0,
                        "l2norm": 0,
                    },
                    bad_fit_metrics_to_fisher_score={},
                ),
            ):
                self.assertIsNotNone(ax_client.get_best_parameters())
                mock_model_best_point.assert_called()

        # Assert the non-mocked method works correctly as well
        self.assertIsNotNone(ax_client.get_best_parameters())

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
