#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.adapter.registry import Generators
from ax.core.data import Data
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.plot.pareto_frontier import interact_pareto_frontier
from ax.plot.pareto_utils import (
    _extract_observed_pareto_2d,
    _relativize_values,
    get_observed_pareto_frontiers,
)
from ax.utils.common.testutils import TestCase
from ax.utils.stats.math_utils import relativize
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)


class ParetoUtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        experiment = get_branin_experiment()
        experiment.add_tracking_metric(
            BraninMetric(name="m2", param_names=["x1", "x2"])
        )
        sobol = Generators.SOBOL(experiment)
        a = sobol.gen(5)
        experiment.new_batch_trial(generator_run=a).run()
        self.experiment = experiment
        self.metrics = list(experiment.metrics.values())

    def test_get_observed_pareto_frontiers(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
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
        bounds = [0, 100, 1000]
        rels = [True, True, False]
        objective_thresholds = [
            ObjectiveThreshold(
                metric=objective.metric,
                bound=bounds[i],
                relative=rels[i],
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
        data = Data(df=df)
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
            self.assertEqual(pfr.absolute_metrics, ["m3"])
            self.assertEqual(list(pfr.means.keys()), ["m1", "m2", "m3"])
            self.assertEqual(len(pfr.means["m1"]), len(pareto_arms))
            self.assertTrue(np.isnan(pfr.sems["m1"]).all())
            self.assertEqual(len(pfr.arm_names), len(pareto_arms))  # pyre-ignore
            self.assertEqual(pfr.objective_thresholds, {"m1": 0, "m2": 100, "m3": 1000})
            # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[ty...
            arm_idx = np.argsort(pfr.arm_names)
            for i, idx in enumerate(arm_idx):
                name = pareto_arms[i]
                self.assertEqual(pfr.arm_names[idx], name)  # pyre-ignore
                self.assertEqual(
                    pfr.param_dicts[idx], experiment.arms_by_name[name].parameters
                )
        pfrs = get_observed_pareto_frontiers(experiment=experiment, data=data, rel=True)
        pfr = pfrs[0]
        self.assertEqual(pfr.absolute_metrics, [])
        self.assertEqual(
            pfr.objective_thresholds,
            {"m1": 0, "m2": 100, "m3": (1000 / sq_val - 1) * 100},
        )
        pfrs = get_observed_pareto_frontiers(
            experiment=experiment, data=data, rel=False
        )
        pfr = pfrs[0]
        self.assertEqual(pfr.absolute_metrics, ["m1", "m2", "m3"])
        self.assertEqual(pfr.objective_thresholds, {"m1": sq_val, "m2": 0, "m3": 1000})
        pfrs = get_observed_pareto_frontiers(
            experiment=experiment, data=data, arm_names=["0_1"]
        )
        for pfr in pfrs:
            self.assertTrue("status_quo" in pfr.arm_names)  # pyre-ignore

        # Test with missing objective thresholds.
        optimization_config._objective_thresholds = []
        with self.assertRaisesRegex(UserInputError, "`rel` must be"):
            get_observed_pareto_frontiers(experiment=experiment, data=data)
        pfr = get_observed_pareto_frontiers(
            experiment=experiment, data=data, rel=False
        )[0]
        self.assertEqual(pfr.absolute_metrics, ["m1", "m2", "m3"])
        self.assertEqual(pfr.primary_metric, "m1")
        self.assertEqual(pfr.secondary_metric, "m2")
        self.assertEqual(len(pfr.means["m1"]), sum(df["metric_name"] == "m1"))
        self.assertEqual(pfr.objective_thresholds, {})
        pfr = get_observed_pareto_frontiers(experiment=experiment, data=data, rel=True)[
            0
        ]
        self.assertEqual(pfr.absolute_metrics, [])

    def test_PlotParetoFrontiers(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            has_objective_thresholds=True,
        )
        sobol = Generators.SOBOL(experiment)
        a = sobol.gen(5)
        experiment.new_batch_trial(generator_run=a).run()
        experiment.fetch_data()
        pfrs = get_observed_pareto_frontiers(experiment=experiment)
        label_dict = {"branin_a": "a_new_metric"}
        b = interact_pareto_frontier(pfrs, label_dict=label_dict)
        self.assertEqual(
            b.data["layout"]["updatemenus"][0]["buttons"][0]["label"],
            "a_new_metric<br>vs branin_b",
        )
        self.assertEqual(b.data["layout"]["xaxis"]["title"]["text"], "branin_b")
        self.assertEqual(b.data["layout"]["yaxis"]["title"]["text"], "a_new_metric")

    def test_extract_observed_pareto_2d(self) -> None:
        Y = np.array([[1.0, 2.0], [2.1, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 0.0]])
        reference_point = (1.5, 0.5)
        minimize = False
        pareto = _extract_observed_pareto_2d(
            Y=Y, reference_point=reference_point, minimize=minimize
        )

        # first and last points are beyond reference point, third point is dominated
        self.assertTrue(np.array_equal(pareto, np.array([[2.1, 1.0], [2.0, 2.0]])))

        # different `minimize` in each direction
        minimize = (True, False)
        pareto = _extract_observed_pareto_2d(
            Y=Y, reference_point=reference_point, minimize=minimize
        )
        self.assertTrue(np.array_equal(pareto, np.array([[1.0, 2.0]])))

        # no `reference_point` provided
        minimize = False
        pareto = _extract_observed_pareto_2d(
            Y=Y, reference_point=None, minimize=minimize
        )
        self.assertTrue(
            np.array_equal(pareto, np.array([[3.0, 0.0], [2.1, 1.0], [2.0, 2.0]]))
        )

    def test__relativize_values(self) -> None:
        # With NaN sem.
        means = [1.0, 2.0, 3.0]
        sems = [float("nan"), 1.0, float("nan")]
        tf_mean, tf_sem = _relativize_values(
            means=means,
            sems=sems,
            sq_mean=2.0,
            sq_sem=1.0,
        )
        self.assertIs(sems, tf_sem)
        self.assertEqual(tf_mean, [-50.0, 0.0, 50.0])

        # With non-NaN sem.
        sems = [0.5, 1.0, 0.3]
        tf_mean, tf_sem = _relativize_values(
            means=means,
            sems=sems,
            sq_mean=2.0,
            sq_sem=1.0,
        )
        rel_mean, rel_sem = relativize(
            means_t=np.array(means),
            sems_t=np.array(sems),
            mean_c=2.0,
            sem_c=1.0,
            as_percent=True,
        )
        self.assertEqual(tf_mean, rel_mean.tolist())
        self.assertEqual(tf_sem, rel_sem.tolist())
