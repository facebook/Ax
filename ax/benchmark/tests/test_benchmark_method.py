# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    get_continuous_search_space,
    get_moo_opt_config,
    get_soo_opt_config,
)
from ax.benchmark.methods.sobol import get_sobol_generation_strategy
from ax.core.experiment import Experiment
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class TestBenchmarkMethod(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.gs = get_sobol_generation_strategy()

    def test_benchmark_method(self) -> None:
        method = BenchmarkMethod(name="Sobol10", generation_strategy=self.gs)
        self.assertEqual(method.name, "Sobol10")

        # test that `fit_tracking_metrics` has been correctly set to False
        for step in method.generation_strategy._steps:
            self.assertFalse(none_throws(step.model_kwargs).get("fit_tracking_metrics"))
        self.assertEqual(method.timeout_hours, 4)

        method = BenchmarkMethod(generation_strategy=self.gs, timeout_hours=10)
        self.assertEqual(method.name, method.generation_strategy.name)
        self.assertEqual(method.timeout_hours, 10)

        # test that instantiation works with node-based strategies
        method = BenchmarkMethod(name="Sobol10", generation_strategy=self.gs)
        for node in method.generation_strategy._nodes:
            self.assertFalse(
                none_throws(node.model_spec_to_gen_from.model_kwargs).get(
                    "fit_tracking_metrics"
                )
            )

    def test_get_best_parameters(self) -> None:
        """
        This is tested more thoroughly in `test_benchmark` -- setting up an
        experiment with data and trials without just running a benchmark is a
        pain, so in that file, we just run a benchmark.
        """
        search_space = get_continuous_search_space(bounds=[(0, 1)])
        experiment = Experiment(name="test", is_test=True, search_space=search_space)
        moo_config = get_moo_opt_config(outcome_names=["a", "b"], ref_point=[0, 0])

        method = BenchmarkMethod(generation_strategy=self.gs)

        with self.subTest("MOO not supported"), self.assertRaisesRegex(
            NotImplementedError, "not currently supported for multi-objective"
        ):
            method.get_best_parameters(
                experiment=experiment, optimization_config=moo_config, n_points=1
            )

        soo_config = get_soo_opt_config(outcome_names=["a"])
        with self.subTest("Multiple points not supported"), self.assertRaisesRegex(
            NotImplementedError, "only n_points=1"
        ):
            method.get_best_parameters(
                experiment=experiment, optimization_config=soo_config, n_points=2
            )

        with self.subTest("Empty experiment"):
            result = method.get_best_parameters(
                experiment=experiment, optimization_config=soo_config, n_points=1
            )
            self.assertEqual(result, [])
