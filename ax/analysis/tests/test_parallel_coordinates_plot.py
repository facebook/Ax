# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.analysis.parallel_coordinates_plot import ParallelCoordinatesPlot
from ax.core.batch_trial import BatchTrial

from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_data_batch, get_branin_experiment
from plotly import graph_objs as go


class TestParallelCoordinatesPlot(TestCase):
    def test_parallel_coordinates_plot(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)

        parallel_coordinates_plot = ParallelCoordinatesPlot(
            experiment=exp, objective_name="branin"
        )

        # Assert that plot can be constructed successfully
        plot = parallel_coordinates_plot.get_fig()
        self.assertIsInstance(plot, go.Figure)

    def test_parallel_coordinates_plot_batch_trial(self) -> None:
        exp = get_branin_experiment(with_batch=True)

        batch_trial = exp.trials[0]
        if not isinstance(batch_trial, BatchTrial):
            raise TypeError("was not a batch trial")
        batch_trial.mark_running(no_runner_required=True)

        exp.attach_data(
            get_branin_data_batch(batch=batch_trial)
        )  # Add data for batch trial
        batch_trial.mark_completed()

        parallel_coordinates_plot = ParallelCoordinatesPlot(
            experiment=exp, objective_name="branin"
        )

        # Assert that plot can be constructed successfully
        plot = parallel_coordinates_plot.get_fig()
        self.assertIsInstance(plot, go.Figure)

    def test_parallel_coordinates_plot_objective_from_config(self) -> None:
        exp = get_branin_experiment(
            has_optimization_config=True, with_completed_trial=True
        )

        parallel_coordinates_plot = ParallelCoordinatesPlot(experiment=exp)

        # Assert that plot can be constructed successfully
        plot = parallel_coordinates_plot.get_fig()
        self.assertIsInstance(plot, go.Figure)

    def test_metric_not_found(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)

        parallel_coordinates_plot = ParallelCoordinatesPlot(
            experiment=exp, objective_name="invalid_metric"
        )

        # Assert that plot can be constructed successfully

        with self.assertRaises(ValueError):
            _ = parallel_coordinates_plot.get_df()

    def test_parallel_coordinates_get_df(self) -> None:
        exp = get_branin_experiment(with_completed_trial=True)

        parallel_coordinates_plot = ParallelCoordinatesPlot(
            experiment=exp, objective_name="branin"
        )

        # Assert that dataframe can be constructed successfully
        df = parallel_coordinates_plot.get_df()
        # check objective in dataframe
        self.assertIn("branin", df.columns)
        # check parameter names in dataframe
        self.assertIn("x1", df.columns)
        self.assertIn("x2", df.columns)
