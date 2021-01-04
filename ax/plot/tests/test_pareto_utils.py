#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.exceptions.core import UnsupportedError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.registry import Models
from ax.plot.pareto_utils import compute_pareto_frontier
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ParetoUtilsTest(TestCase):
    def setUp(self):
        experiment = get_branin_experiment()
        experiment.add_tracking_metric(
            BraninMetric(name="m2", param_names=["x1", "x2"])
        )
        sobol = Models.SOBOL(experiment.search_space)
        a = sobol.gen(5)
        experiment.new_batch_trial(generator_run=a).run()
        self.experiment = experiment
        self.metrics = list(experiment.metrics.values())

    def testComputeParetoFrontierByTrial(self):
        # Experiments with batch trials must specify trial_index or data
        with self.assertRaises(UnsupportedError):
            compute_pareto_frontier(
                self.experiment,
                self.metrics[0],
                self.metrics[1],
                absolute_metrics=[m.name for m in self.metrics],
            )
        compute_pareto_frontier(
            self.experiment,
            self.metrics[0],
            self.metrics[1],
            trial_index=0,
            absolute_metrics=[m.name for m in self.metrics],
            num_points=2,
        )

    def testComputeParetoFrontierByData(self):
        # Experiments with batch trials must specify trial_index or data
        compute_pareto_frontier(
            self.experiment,
            self.metrics[0],
            self.metrics[1],
            data=self.experiment.fetch_data(),
            absolute_metrics=[m.name for m in self.metrics],
            num_points=2,
        )
