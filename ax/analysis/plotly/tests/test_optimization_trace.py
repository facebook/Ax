# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

from ax.analysis.plotly.arm_effects.optimization_trace import OptimizationTrace
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from ax.core.arm import Arm

from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Generators
from ax.utils.testing.core_stubs import get_branin_experiment
from pyre_extensions import assert_is_instance, none_throws


class TestOptimizationTrace(TestCase):
    """Tests for the OptimizationTrace analysis."""

    def setUp(self) -> None:
        self.experiment = get_branin_experiment(with_relative_constraint=True)
        self.metric = none_throws(self.experiment.optimization_config).objective.metric
        sobol = Generators.SOBOL(search_space=self.experiment.search_space)
        for parameters in [
            {"x1": 0.0, "x2": 0.0},
            {"x1": 1.0, "x2": 1.0},
        ]:
            t = self.experiment.new_trial(
                GeneratorRun(
                    arms=[Arm(parameters=parameters)], generation_node_name="Manual"
                )
            ).run()
            t.mark_completed()
            if self.experiment.status_quo is None:
                self.experiment.status_quo = t.arm
        for _ in range(2):
            t = self.experiment.new_trial(sobol.gen(1)).run()
            t.mark_completed()
        model = Generators.BOTORCH_MODULAR(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
        )
        for _ in range(2):
            t = self.experiment.new_trial(model.gen(1)).run()
            t.mark_completed()
        self.experiment.fetch_data()

    def test_optimization_trace_minimize(self) -> None:
        """Test optimization trace visualization for minimization problem."""
        trace = OptimizationTrace().compute(experiment=self.experiment)[0]
        self.assertIsInstance(trace, PlotlyAnalysisCard)
        fig = trace.get_figure()
        self.assertTrue(fig is not None)
        self.assertTrue(len(fig.data) > 0)

    def test_optimization_trace_tracks_running_optimum(self) -> None:
        """Test optimization trace visualization for maximization problem."""
        for minimize in [True, False]:
            self.experiment.optimization_config.objective.metric.lower_is_better = (
                minimize
            )
            self.experiment.optimization_config.objective.minimize = minimize
            trace = OptimizationTrace().compute(
                experiment=self.experiment,
            )[0]
            self.assertIsInstance(trace, PlotlyAnalysisCard)
            fig = trace.get_figure()
            direction_str = "minimizing" if minimize else "maximizing"
            self.assertIn(direction_str, trace.title.lower())

            # Verify data points match trial values
            for data in fig.data:
                if len(data.y) == 1 and data.y[0] is None:
                    # Empty scatters alter the legend without altering the plot
                    continue
                elif data.name in ["Unknown", "Sobol", "BoTorch"]:
                    self.assertEqual(len(data.y), 2)
                elif data.name == "Running optimum":
                    # Verify running optimum is continually improving
                    optimization_mult = -1 if minimize else 1
                    self.assertListEqual(
                        sorted(optimization_mult * assert_is_instance(data.y, tuple)),
                        list(optimization_mult * assert_is_instance(data.y, tuple)),
                    )
                else:
                    self.fail(msg=f"Found unaccounted for data object {data} in test")

    def test_optimization_trace_errors(self) -> None:
        """Test error handling in optimization trace visualization."""
        # Experiment not provided
        with self.assertRaisesRegex(
            UserInputError, "Experiment cannot be None for OptimizationTrace analysis."
        ):
            OptimizationTrace().compute()
        # Empty experiment
        exp = get_branin_experiment()
        with self.assertRaisesRegex(UserInputError, "Experiment contains no data"):
            OptimizationTrace().compute(experiment=exp)

        exp = self.experiment.clone_with()
        # No optimization config
        exp._optimization_config = None
        with self.assertRaisesRegex(
            UserInputError,
            "Experiment must have an optimization config for OptimizationTrace "
            "analysis.",
        ):
            OptimizationTrace().compute(experiment=exp)

        # Multiobjective
        exp._optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=[Objective(self.metric)] * 2),
        )
        with self.assertRaisesRegex(UserInputError, "multi-objective optimization"):
            OptimizationTrace().compute(experiment=exp)

        # Metric values not found in experiment data
        exp._optimization_config = OptimizationConfig(
            objective=Objective(Metric(name="foo"), minimize=True)
        )
        exp._tracking_metrics = {self.metric.name: self.metric}
        with self.assertRaisesRegex(
            UserInputError, "Optimization metric .* not found in experiment data"
        ):
            OptimizationTrace().compute(experiment=exp)
