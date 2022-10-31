#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.modelbridge.cross_validation import AssessModelFitResult
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.service.utils.best_point import (
    get_best_parameters,
    get_best_raw_objective_point,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import fast_botorch_optimize


class TestBestPointUtils(TestCase):
    """Testing the best point utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    @fast_botorch_optimize
    def test_best_from_model_prediction(self) -> None:
        exp = get_branin_experiment()

        for _ in range(3):
            sobol = Models.SOBOL(search_space=exp.search_space)
            generator_run = sobol.gen(n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            exp.attach_data(exp.fetch_data())

        gpei = Models.BOTORCH(experiment=exp, data=exp.lookup_data())
        generator_run = gpei.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

        with patch.object(
            TorchModelBridge,
            "model_best_point",
            return_value=(
                (
                    Arm(
                        name="0_0",
                        parameters={"x1": -4.842811906710267, "x2": 11.887089014053345},
                    ),
                    (
                        {"branin": 34.76260622783635},
                        {"branin": {"branin": 0.00028306433439807734}},
                    ),
                )
            ),
        ) as mock_model_best_point, self.assertLogs(
            logger="ax.service.utils.best_point", level="WARN"
        ) as lg:
            # Test bad model fit causes function to resort back to raw data
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={},
                    bad_fit_metrics_to_fisher_score={
                        "branin": 0,
                    },
                ),
            ):
                self.assertIsNotNone(get_best_parameters(exp, Models))
                self.assertTrue(
                    any("Model fit is poor" in warning for warning in lg.output),
                    msg=lg.output,
                )
                mock_model_best_point.assert_not_called()

            # Test model best point is used when fit is good
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={
                        "branin": 0,
                    },
                    bad_fit_metrics_to_fisher_score={},
                ),
            ):
                self.assertIsNotNone(get_best_parameters(exp, Models))
                mock_model_best_point.assert_called()

        # Assert the non-mocked method works correctly as well
        self.assertIsNotNone(get_best_parameters(exp, Models))

    def test_best_raw_objective_point(self) -> None:
        exp = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        # pyre-fixme[16]: Optional type has no attribute `clone`.
        opt_conf = exp.optimization_config.clone()
        opt_conf.objective.metric._name = "not_branin"
        with self.assertRaisesRegex(ValueError, "No data has been logged"):
            get_best_raw_objective_point(exp, opt_conf)

        # Test constraints work as expected.
        observations = [[1.0, 2.0], [3.0, 4.0], [5.0, -6.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
            minimize=False,
        )
        _, best_prediction = not_none(get_best_parameters(exp, Models))
        best_metrics = not_none(best_prediction)[0]
        self.assertDictEqual(best_metrics, {"m1": 3.0, "m2": 4.0})

    def test_best_raw_objective_point_unsatisfiable(self) -> None:
        exp = get_branin_experiment()
        trial = exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        trial.mark_completed()
        exp.fetch_data()

        # pyre-fixme[16]: Optional type has no attribute `clone`.
        opt_conf = exp.optimization_config.clone()
        opt_conf.outcome_constraints.append(
            OutcomeConstraint(
                metric=get_branin_metric(), op=ComparisonOp.LEQ, bound=0, relative=False
            )
        )

        with self.assertRaisesRegex(ValueError, "No points satisfied"):
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_unsatisfiable_relative(self) -> None:
        exp = get_branin_experiment()

        # Optimization config with unsatisfiable constraint
        # pyre-fixme[16]: Optional type has no attribute `clone`.
        opt_conf = exp.optimization_config.clone()
        opt_conf.outcome_constraints.append(
            OutcomeConstraint(
                metric=get_branin_metric(),
                op=ComparisonOp.GEQ,
                bound=9999,
                relative=True,
            )
        )

        trial = exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        trial.mark_completed()
        exp.fetch_data()

        with self.assertLogs(logger="ax.service.utils.best_point", level="WARN") as lg:
            get_best_raw_objective_point(exp, opt_conf)
            self.assertTrue(
                any("No status quo provided" in warning for warning in lg.output),
                msg=lg.output,
            )

        exp.status_quo = Arm(parameters={"x1": 0, "x2": 0}, name="status_quo")
        sq_trial = exp.new_trial(
            # pyre-fixme[6]: For 1st param expected `List[Arm]` but got
            #  `List[Optional[Arm]]`.
            generator_run=GeneratorRun(arms=[exp.status_quo])
        ).run()
        sq_trial.mark_completed()
        exp.fetch_data()

        with self.assertRaisesRegex(ValueError, "No points satisfied"):
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_scalarized(self) -> None:
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(metrics=[get_branin_metric()], minimize=False)
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})

    def test_best_raw_objective_point_scalarized_multi(self) -> None:
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
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})
