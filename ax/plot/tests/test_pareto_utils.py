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
    def testOODStatusQuo(self):
        # An OOD status quo arm without a trial index will raise an error
        experiment = get_branin_experiment()
        experiment.add_tracking_metric(
            BraninMetric(name="m2", param_names=["x1", "x2"])
        )
        metrics = list(experiment.metrics.values())
        sobol = Models.SOBOL(experiment.search_space)
        a = sobol.gen(5)
        experiment.new_batch_trial(generator_run=a).run()
        # Experiments with batch trials must specify a trial index
        with self.assertRaises(UnsupportedError):
            compute_pareto_frontier(
                experiment,
                metrics[0],
                metrics[1],
                absolute_metrics=[m.name for m in metrics],
            )
        compute_pareto_frontier(
            experiment,
            metrics[0],
            metrics[1],
            trial_index=0,
            absolute_metrics=[m.name for m in metrics],
        )
