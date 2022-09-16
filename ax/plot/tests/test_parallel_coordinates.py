# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.plot.base import AxPlotConfig, AxPlotTypes
from ax.plot.parallel_coordinates import plot_parallel_coordinates
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ParallelCoordinatesTest(TestCase):
    def testParallelCoordinates(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()

        # Assert that each type of plot can be constructed successfully
        plot = plot_parallel_coordinates(experiment=exp)

        self.assertIsInstance(plot, AxPlotConfig)
        self.assertEqual(plot.plot_type, AxPlotTypes.GENERIC)
