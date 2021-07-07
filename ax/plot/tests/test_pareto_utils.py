#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
from ax.core.data import Data
from ax.core.objective import Objective, MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.exceptions.core import UnsupportedError
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.modelbridge.registry import Models
from ax.plot.pareto_frontier import interact_multiple_pareto_frontier
from ax.plot.pareto_utils import (
    compute_posterior_pareto_frontier,
    get_observed_pareto_frontiers,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


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

    def testComputePosteriorParetoFrontierByTrial(self):
        # Experiments with batch trials must specify trial_index or data
        with self.assertRaises(UnsupportedError):
            compute_posterior_pareto_frontier(
                self.experiment,
                self.metrics[0],
                self.metrics[1],
                absolute_metrics=[m.name for m in self.metrics],
            )
        pfr = compute_posterior_pareto_frontier(
            self.experiment,
            self.metrics[0],
            self.metrics[1],
            trial_index=0,
            absolute_metrics=[m.name for m in self.metrics],
            num_points=2,
        )
        self.assertIsNone(pfr.arm_names)

    def testComputePosteriorParetoFrontierByData(self):
        # Experiments with batch trials must specify trial_index or data
        compute_posterior_pareto_frontier(
            self.experiment,
            self.metrics[0],
            self.metrics[1],
            data=self.experiment.fetch_data(),
            absolute_metrics=[m.name for m in self.metrics],
            num_points=2,
        )

    def testObservedParetoFrontiers(self):
        experiment = get_branin_experiment(
            with_batch=True, has_optimization_config=False, with_status_quo=True
        )

        # Optimization config is not optional
        with self.assertRaises(ValueError):
            get_observed_pareto_frontiers(experiment=experiment, data=Data())

        objectives = [
            Objective(
                metric=BraninMetric(
                    name="m1", param_names=["x1", "x2"], lower_is_better=True
                ),
                minimize=True,
            ),
            Objective(
                metric=NegativeBraninMetric(
                    name="m2", param_names=["x1", "x2"], lower_is_better=True
                ),
                minimize=True,
            ),
            Objective(
                metric=BraninMetric(
                    name="m3", param_names=["x1", "x2"], lower_is_better=True
                ),
                minimize=True,
            ),
        ]
        bounds = [0, -100, 0]
        objective_thresholds = [
            ObjectiveThreshold(
                metric=objective.metric,
                bound=bounds[i],
                relative=True,
                op=ComparisonOp.LEQ,
            )
            for i, objective in enumerate(objectives)
        ]
        objective = MultiObjective(objectives=objectives)
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=objective,
            objective_thresholds=objective_thresholds,
        )
        experiment.optimization_config = optimization_config
        experiment.trials[0].run()

        # For the check below, compute which arms are better than SQ
        df = experiment.fetch_data().df
        df["sem"] = np.nan
        data = Data(df)
        sq_val = df[(df["arm_name"] == "status_quo") & (df["metric_name"] == "m1")][
            "mean"
        ].values[0]
        pareto_arms = sorted(
            df[(df["mean"] <= sq_val) & (df["metric_name"] == "m1")]["arm_name"]
            .unique()
            .tolist()
        )

        pfrs = get_observed_pareto_frontiers(experiment=experiment, data=data)
        # We have all pairs of metrics
        self.assertEqual(len(pfrs), 3)
        true_pairs = [("m1", "m2"), ("m1", "m3"), ("m2", "m3")]
        for i, pfr in enumerate(pfrs):
            self.assertEqual(pfr.primary_metric, true_pairs[i][0])
            self.assertEqual(pfr.secondary_metric, true_pairs[i][1])
            self.assertEqual(pfr.absolute_metrics, [])
            self.assertEqual(list(pfr.means.keys()), ["m1", "m2", "m3"])
            self.assertEqual(len(pfr.means["m1"]), len(pareto_arms))
            self.assertTrue(np.isnan(pfr.sems["m1"]).all())
            self.assertEqual(len(pfr.arm_names), len(pareto_arms))
            arm_idx = np.argsort(pfr.arm_names)
            for i, idx in enumerate(arm_idx):
                name = pareto_arms[i]
                self.assertEqual(pfr.arm_names[idx], name)
                self.assertEqual(
                    pfr.param_dicts[idx], experiment.arms_by_name[name].parameters
                )

    def testPlotMultipleParetoFrontiers(self):
        experiment = get_branin_experiment_with_multi_objective(
            has_objective_thresholds=True,
        )
        sobol = Models.SOBOL(experiment.search_space)
        a = sobol.gen(5)
        experiment.new_batch_trial(generator_run=a).run()
        pfrs = get_observed_pareto_frontiers(experiment=experiment)
        pfrs2 = copy.deepcopy(pfrs)
        pfr_lists = {"pfrs 1": pfrs, "pfrs 2": pfrs2}
        self.assertIsNotNone(interact_multiple_pareto_frontier(pfr_lists))
