# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product

import torch
from ax.benchmark.benchmark_metric import BenchmarkMapMetric, BenchmarkMetric
from ax.benchmark.benchmark_problem import get_continuous_search_space
from ax.benchmark.benchmark_test_functions.botorch_test import BoTorchTestFunction
from ax.benchmark.problems.synthetic.from_botorch import (
    _get_name,
    create_problem_from_botorch,
    get_augmented_branin_problem,
    get_augmented_branin_search_space,
)
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.multi_objective import BraninCurrin, ConstrainedBraninCurrin
from botorch.test_functions.synthetic import (
    Ackley,
    Branin,
    ConstrainedGramacy,
    ConstrainedHartmann,
    Cosine8,
    Hartmann,
)
from pyre_extensions import assert_is_instance, none_throws


class TestBoTorchProblems(TestCase):
    def test_get_augmented_branin_search_space(self) -> None:
        with self.subTest("fidelity"):
            search_space = get_augmented_branin_search_space(
                fidelity_or_task="fidelity"
            )
            param = assert_is_instance(search_space.parameters["x2"], RangeParameter)
            self.assertEqual(param.target_value, 1.0)
            self.assertTrue(param.is_fidelity)

        with self.subTest("task"):
            problem = get_augmented_branin_problem(fidelity_or_task="task")
            param = assert_is_instance(
                problem.search_space.parameters["x2"], ChoiceParameter
            )
            self.assertEqual(param.target_value, 1.0)
            self.assertTrue(param.is_task)
            self.assertFalse(param.is_fidelity)

    def test_get_augmented_branin_problem(self) -> None:
        with self.subTest("inference value as trace"):
            problem = get_augmented_branin_problem(
                fidelity_or_task="fidelity", report_inference_value_as_trace=True
            )
            self.assertTrue(problem.report_inference_value_as_trace)
            self.assertEqual(problem.name, "AugmentedBranin")

        with self.subTest("Do not report inference value as trace"):
            problem = get_augmented_branin_problem(
                fidelity_or_task="fidelity", report_inference_value_as_trace=False
            )
            self.assertFalse(problem.report_inference_value_as_trace)


class TestFromBoTorch(TestCase):
    def test_single_objective_from_botorch(self) -> None:
        for botorch_test_problem in [Ackley(), ConstrainedHartmann(dim=6)]:
            test_problem = create_problem_from_botorch(
                test_problem_class=botorch_test_problem.__class__,
                test_problem_kwargs={},
                num_trials=1,
                baseline_value=100.0,
            )

            # Test search space
            self.assertEqual(
                len(test_problem.search_space.parameters), botorch_test_problem.dim
            )
            self.assertEqual(
                len(test_problem.search_space.parameters),
                len(test_problem.search_space.range_parameters),
            )
            self.assertTrue(
                all(
                    test_problem.search_space.range_parameters[f"x{i}"].lower
                    == botorch_test_problem._bounds[i][0]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' lower bounds must all match Botorch problem's bounds.",
            )
            self.assertTrue(
                all(
                    test_problem.search_space.range_parameters[f"x{i}"].upper
                    == botorch_test_problem._bounds[i][1]
                    for i in range(botorch_test_problem.dim)
                ),
                "Parameters' upper bounds must all match Botorch problem's bounds.",
            )

            # Test optimum
            self.assertEqual(
                test_problem.optimal_value, botorch_test_problem._optimal_value
            )
            # test optimization config
            metric_name = test_problem.optimization_config.objective.metric.name
            self.assertEqual(metric_name, test_problem.name)
            self.assertTrue(test_problem.optimization_config.objective.minimize)
            # test repr method
            if isinstance(botorch_test_problem, Ackley):
                self.assertEqual(
                    test_problem.optimization_config.outcome_constraints, []
                )
            else:
                outcome_constraint = (
                    test_problem.optimization_config.outcome_constraints[0]
                )
                self.assertEqual(outcome_constraint.metric.name, "constraint_slack_0")
                self.assertEqual(outcome_constraint.op, ComparisonOp.GEQ)
                self.assertFalse(outcome_constraint.relative)
                self.assertEqual(outcome_constraint.bound, 0.0)

    def _test_constrained_from_botorch(
        self,
        observe_noise_sd: bool,
        noise_std: float | dict[str, float],
        test_problem_class: type[ConstrainedBaseTestProblem],
    ) -> None:
        ax_problem = create_problem_from_botorch(
            test_problem_class=test_problem_class,
            test_problem_kwargs={},
            lower_is_better=True,
            num_trials=1,
            observe_noise_sd=observe_noise_sd,
            noise_std=noise_std,
        )
        test_problem = assert_is_instance(ax_problem.test_function, BoTorchTestFunction)
        botorch_problem = assert_is_instance(
            test_problem.botorch_problem, ConstrainedBaseTestProblem
        )
        self.assertEqual(ax_problem.noise_std, noise_std)
        opt_config = ax_problem.optimization_config
        outcome_constraints = opt_config.outcome_constraints
        self.assertEqual(
            [constraint.metric.name for constraint in outcome_constraints],
            [f"constraint_slack_{i}" for i in range(botorch_problem.num_constraints)],
        )
        objective = opt_config.objective
        metric = (
            objective.metrics[0]
            if isinstance(objective, MultiObjective)
            else objective.metric
        )

        self.assertEqual(
            assert_is_instance(metric, BenchmarkMetric).observe_noise_sd,
            observe_noise_sd,
        )

        # TODO: Support observing noise variance only for some outputs
        for constraint in outcome_constraints:
            self.assertEqual(
                assert_is_instance(constraint.metric, BenchmarkMetric).observe_noise_sd,
                observe_noise_sd,
            )

    def test_constrained_soo_from_botorch(self) -> None:
        for observe_noise_sd, noise_std in product(
            [False, True],
            [
                0.0,
                0.1,
                {
                    "ConstrainedGramacy": 0.1,
                    "constraint_slack_0": 0.3,
                    "constraint_slack_1": 0.4,
                },
            ],
        ):
            with self.subTest(observe_noise_sd=observe_noise_sd, noise_std=noise_std):
                self._test_constrained_from_botorch(
                    observe_noise_sd=observe_noise_sd,
                    noise_std=noise_std,
                    test_problem_class=ConstrainedGramacy,
                )

    def test_constrained_moo_from_botorch(self) -> None:
        self._test_constrained_from_botorch(
            observe_noise_sd=False,
            noise_std=0.0,
            test_problem_class=ConstrainedBraninCurrin,
        )

    def _test_moo_from_botorch(self, lower_is_better: bool) -> None:
        test_problem = BraninCurrin()
        branin_currin_problem = create_problem_from_botorch(
            test_problem_class=test_problem.__class__,
            test_problem_kwargs={},
            num_trials=1,
            lower_is_better=lower_is_better,
        )

        # Test search space
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters), test_problem.dim
        )
        self.assertEqual(
            len(branin_currin_problem.search_space.parameters),
            len(branin_currin_problem.search_space.range_parameters),
        )
        self.assertTrue(
            all(
                branin_currin_problem.search_space.range_parameters[f"x{i}"].lower
                == test_problem._bounds[i][0]
                for i in range(test_problem.dim)
            ),
            "Parameters' lower bounds must all match Botorch problem's bounds.",
        )
        self.assertTrue(
            all(
                branin_currin_problem.search_space.range_parameters[f"x{i}"].upper
                == test_problem._bounds[i][1]
                for i in range(test_problem.dim)
            ),
            "Parameters' upper bounds must all match Botorch problem's bounds.",
        )

        # Test hypervolume
        self.assertEqual(branin_currin_problem.optimal_value, test_problem._max_hv)
        opt_config = assert_is_instance(
            branin_currin_problem.optimization_config, MultiObjectiveOptimizationConfig
        )
        reference_point = [
            threshold.bound for threshold in opt_config.objective_thresholds
        ]
        self.assertEqual(reference_point, test_problem._ref_point)

        self.assertTrue(
            all(
                t.op is (ComparisonOp.LEQ if lower_is_better else ComparisonOp.GEQ)
                for t in opt_config.objective_thresholds
            )
        )
        self.assertTrue(
            all(
                metric.lower_is_better is lower_is_better
                for metric in opt_config.metrics.values()
            )
        )

    def test_moo_from_botorch(self) -> None:
        self._test_moo_from_botorch(lower_is_better=True)
        self._test_moo_from_botorch(lower_is_better=False)

    def test_create_problem_from_botorch_with_shifted_function(self) -> None:
        ax_problem = create_problem_from_botorch(
            test_problem_class=BraninCurrin,
            test_problem_kwargs={},
            num_trials=10,
            use_shifted_function=True,
        )
        dim = 2
        # Check that search space is expanded by default.
        self.assertEqual(
            ax_problem.search_space,
            get_continuous_search_space(bounds=[(0.0, 2.0)] * dim),
        )
        # Check that the arguments are passed down to the BoTorchTestFunction.
        test_problem = assert_is_instance(ax_problem.test_function, BoTorchTestFunction)
        self.assertTrue(test_problem.use_shifted_function)
        self.assertEqual(none_throws(test_problem._offset).shape, torch.Size([dim]))
        # Check that the offset is applied.
        self.assertAllClose(
            test_problem.tensorize_params({f"x{i}": 0 for i in range(dim)}),
            -none_throws(test_problem._offset),
        )

    def test_maximization_problem(self) -> None:
        test_problem = create_problem_from_botorch(
            test_problem_class=Cosine8,
            lower_is_better=False,
            num_trials=1,
            test_problem_kwargs={},
            baseline_value=-8,
        )
        self.assertFalse(test_problem.optimization_config.objective.minimize)

    def test_sq_out_of_search_space(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Status quo parameters are not in the search space."
        ):
            create_problem_from_botorch(
                test_problem_class=Branin,
                lower_is_better=True,
                num_trials=1,
                test_problem_kwargs={},
                status_quo_params={"x0": 20.0, "x1": 20.0},
            )

    def test_get_embedded_from_botorch(self) -> None:
        problem = create_problem_from_botorch(
            test_problem_class=Hartmann,
            test_problem_kwargs={"dim": 6},
            n_dummy_dimensions=24,
            num_trials=1,
        )
        self.assertEqual(problem.name, "Hartmann_30d")
        self.assertEqual(len(problem.search_space.parameters), 30)

    def test_get_name(self) -> None:
        with self.subTest("Basic case"):
            name = _get_name(test_problem=Branin(), observe_noise_sd=False)
            self.assertEqual(name, "Branin")

        with self.subTest("Observe noise sd"):
            name = _get_name(test_problem=Branin(), observe_noise_sd=True)
            self.assertEqual(name, "Branin_observed_noise")

        with self.subTest("dim specified"):
            name = _get_name(
                test_problem=Hartmann(dim=6), dim=6, observe_noise_sd=False
            )
            self.assertEqual(name, "Hartmann_6d")

        with self.subTest("dim specified and embedded dims"):
            name = _get_name(
                test_problem=Hartmann(dim=6),
                dim=6,
                n_dummy_dimensions=24,
                observe_noise_sd=False,
            )
            self.assertEqual(name, "Hartmann_30d")

        with self.subTest("dim not specified and embedded dims"):
            name = _get_name(
                test_problem=Branin(), n_dummy_dimensions=24, observe_noise_sd=False
            )
            self.assertEqual(name, "Branin_26d")

        with self.subTest("embedded dims and observed noise"):
            name = _get_name(
                test_problem=Branin(), n_dummy_dimensions=24, observe_noise_sd=True
            )
            self.assertEqual(name, "Branin_observed_noise_26d")

    def test_with_map_metric(self) -> None:
        with self.subTest("With default n_steps"):
            problem = create_problem_from_botorch(
                test_problem_class=Branin,
                test_problem_kwargs={},
                num_trials=1,
                use_map_metric=True,
            )
            self.assertIsInstance(
                problem.optimization_config.objective.metric, BenchmarkMapMetric
            )
            self.assertEqual(problem.test_function.n_steps, 1)

        with self.subTest("With non-default n_steps"):
            n_steps = 4
            problem = create_problem_from_botorch(
                test_problem_class=Branin,
                test_problem_kwargs={},
                num_trials=1,
                use_map_metric=True,
                n_steps=4,
            )
            self.assertIsInstance(
                problem.optimization_config.objective.metric, BenchmarkMapMetric
            )
            self.assertEqual(problem.test_function.n_steps, n_steps)

        with self.subTest("MOO"):
            problem = create_problem_from_botorch(
                test_problem_class=BraninCurrin,
                test_problem_kwargs={},
                num_trials=1,
                use_map_metric=True,
            )
            metric = next(iter(problem.optimization_config.metrics.values()))
            self.assertIsInstance(metric, BenchmarkMapMetric)

    def test_string_test_problem_class(self) -> None:
        """Test that test_problem_class can be provided as a string."""

        problem_from_string = create_problem_from_botorch(
            test_problem_class="Branin",
            test_problem_kwargs={"negate": True},
            num_trials=1,
            baseline_value=10.0,
        )
        problem_from_class = create_problem_from_botorch(
            test_problem_class=Branin,
            test_problem_kwargs={"negate": True},
            num_trials=1,
            baseline_value=10.0,
        )

        string_test_function = assert_is_instance(
            problem_from_string.test_function, BoTorchTestFunction
        )
        class_test_function = assert_is_instance(
            problem_from_class.test_function, BoTorchTestFunction
        )
        self.assertEqual(problem_from_string.name, "Branin")
        string_botorch_problem = string_test_function.botorch_problem
        class_botorch_problem = class_test_function.botorch_problem
        self.assertIsInstance(string_botorch_problem, Branin)
        self.assertIsInstance(class_botorch_problem, Branin)
        self.assertTrue(string_botorch_problem.negate)
        self.assertTrue(class_botorch_problem.negate)

        self.assertEqual(
            problem_from_string.search_space,
            problem_from_class.search_space,
        )
        self.assertEqual(
            problem_from_string.optimal_value, problem_from_class.optimal_value
        )
        self.assertEqual(
            problem_from_string.optimization_config,
            problem_from_class.optimization_config,
        )
