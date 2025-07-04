# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.dispatch import choose_analyses
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


class TestUtils(TestCase):
    def test_choose_analyses(self) -> None:
        analyses = choose_analyses(experiment=get_branin_experiment())
        self.assertEqual(
            {analysis.__class__.__name__ for analysis in analyses},
            {
                "ParallelCoordinatesPlot",
                "TopSurfacesAnalysis",
                "Summary",
                "CrossValidationPlot",
            },
        )

        # Multi-objective case
        analyses = choose_analyses(
            experiment=get_branin_experiment_with_multi_objective()
        )
        self.assertEqual(
            {analysis.__class__.__name__ for analysis in analyses},
            {"TopSurfacesAnalysis", "ScatterPlot", "Summary", "CrossValidationPlot"},
        )
