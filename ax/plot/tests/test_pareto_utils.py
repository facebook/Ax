#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from unittest.mock import patch

import numpy as np
import torch
from ax.core.data import Data
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.modelbridge.registry import Models
from ax.plot.pareto_frontier import (
    interact_multiple_pareto_frontier,
    interact_pareto_frontier,
)
from ax.plot.pareto_utils import (
    _extract_observed_pareto_2d,
    get_observed_pareto_frontiers,
    infer_reference_point_from_experiment,
    to_nonrobust_search_space,
)

from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_experiment_with_observations,
    get_robust_search_space_environmental,
    get_search_space,
)


class ParetoUtilsTest(TestCase):
    def setUp(self) -> None:
        experiment = get_branin_experiment()
        experiment.add_tracking_metric(
            BraninMetric(name="m2", param_names=["x1", "x2"])
        )
        sobol = Models.SOBOL(experiment.search_space)
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
        bounds = [0, -100, 1000]
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
            self.assertEqual(pfr.absolute_metrics, ["m3"])
            self.assertEqual(list(pfr.means.keys()), ["m1", "m2", "m3"])
            self.assertEqual(len(pfr.means["m1"]), len(pareto_arms))
            self.assertTrue(np.isnan(pfr.sems["m1"]).all())
            self.assertEqual(len(pfr.arm_names), len(pareto_arms))  # pyre-ignore
            self.assertEqual(
                pfr.objective_thresholds, {"m1": 0, "m2": -100, "m3": 1000}
            )
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
            {"m1": 0, "m2": -100, "m3": (1000 / sq_val - 1) * 100},
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

    def testPlotParetoFrontiers(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            has_objective_thresholds=True,
        )
        sobol = Models.SOBOL(experiment.search_space)
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
        pfrs2 = copy.deepcopy(pfrs)
        pfr_lists = {"pfrs 1": pfrs, "pfrs 2": pfrs2}
        self.assertIsNotNone(interact_multiple_pareto_frontier(pfr_lists))

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

    def test_to_nonrobust_search_space(self) -> None:
        # Return non-robust search space as is.
        search_space = get_search_space()
        self.assertIs(to_nonrobust_search_space(search_space), search_space)
        # Prune environmental variables and distributions from RSS.
        rss = get_robust_search_space_environmental()
        transformed_ss = to_nonrobust_search_space(rss)
        # Can't use isinstance here since RSS is also a SearchSpace.
        self.assertEqual(transformed_ss.__class__, SearchSpace)
        self.assertEqual(transformed_ss.parameters, rss._parameters)
        self.assertEqual(
            transformed_ss.parameter_constraints, rss.parameter_constraints
        )


class TestInfereReferencePointFromExperiment(TestCase):
    def test_infer_reference_point_from_experiment(self) -> None:
        for constrained in (True, False):
            observations = [[-1.0, 1.0], [-0.5, 2.0], [-2.0, 0.5], [-0.1, 0.1]]
            if constrained:
                observations = [
                    o + [c] for o, c in zip(observations, [1.0, 0.5, 1.0, 1.0])
                ]
            # Getting an experiment with 2 objectives by the above observations.
            experiment = get_experiment_with_observations(
                observations=observations,
                minimize=True,
                scalarized=False,
                constrained=constrained,
            )

            inferred_reference_point = infer_reference_point_from_experiment(experiment)

            # The nadir point for this experiment is [-0.5, 0.5]. The function actually
            # deducts 0.1*Y_range from each of the objectives. Since the range for each
            # of the objectives is +/-1.5, the inferred reference point would
            # be [-0.35, 0.35].
            self.assertEqual(inferred_reference_point[0].op, ComparisonOp.LEQ)
            self.assertEqual(inferred_reference_point[0].bound, -0.35)
            self.assertEqual(inferred_reference_point[0].metric.name, "m1")
            self.assertEqual(inferred_reference_point[1].op, ComparisonOp.GEQ)
            self.assertEqual(inferred_reference_point[1].bound, 0.35)
            self.assertEqual(inferred_reference_point[1].metric.name, "m2")

    def test_infer_reference_point_from_experiment_shuffled_metrics(self) -> None:
        # Generating an experiment with given data.
        observations = [
            [-1.0, 1.0, 0.1],
            [-0.5, 2.0, 0.2],
            [-2.0, 0.5, 0.3],
            [-0.1, 0.1, 0.4],
        ]
        experiment = get_experiment_with_observations(
            observations=observations,
            minimize=True,
            scalarized=False,
            constrained=True,
        )

        # Constructing fake outputs for `get_pareto_frontier_and_configs` so that
        # the order of metrics `m1`, `m2` and `m3` are reversed.
        frontier_observations_shuffled = [
            Observation(
                features=ObservationFeatures(parameters={"x": 0.0, "y": 0.0}),
                data=ObservationData(
                    metric_names=["m3", "m2", "m1"],
                    means=np.array([0.1, 1.0, -1.0]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 0.1, "y": 0.1}),
                data=ObservationData(
                    metric_names=["m3", "m2", "m1"],
                    means=np.array([0.2, 2.0, -0.5]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 0.2, "y": 0.2}),
                data=ObservationData(
                    metric_names=["m3", "m2", "m1"],
                    means=np.array([0.3, 0.5, -2.0]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
        ]
        f_shuffled = torch.tensor(
            [
                [0.1000, 1.0000, -1.0000],
                [0.2000, 2.0000, -0.5000],
                [0.3000, 0.5000, -2.0000],
            ],
            dtype=torch.float64,
        )
        obj_w_shuffled = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
        obj_t_shuffled = torch.tensor(
            [-torch.inf, -torch.inf, torch.inf], dtype=torch.float64
        )

        # Test the function with these shuffled output for
        # `get_pareto_frontier_and_configs`.
        with patch(
            "ax.plot.pareto_utils.get_pareto_frontier_and_configs",
            return_value=(
                frontier_observations_shuffled,
                f_shuffled,
                obj_w_shuffled,
                obj_t_shuffled,
            ),
        ):
            inferred_reference_point = infer_reference_point_from_experiment(experiment)

            self.assertEqual(inferred_reference_point[0].op, ComparisonOp.LEQ)
            self.assertEqual(inferred_reference_point[0].bound, -0.35)
            self.assertEqual(inferred_reference_point[0].metric.name, "m1")
            self.assertEqual(inferred_reference_point[1].op, ComparisonOp.GEQ)
            self.assertEqual(inferred_reference_point[1].bound, 0.35)
            self.assertEqual(inferred_reference_point[1].metric.name, "m2")
