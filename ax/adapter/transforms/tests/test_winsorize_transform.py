#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from copy import deepcopy
from math import sqrt
from typing import Any, SupportsIndex
from unittest import mock

import numpy as np
from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.winsorize import (
    _get_auto_winsorization_cutoffs_outcome_constraint,
    _get_auto_winsorization_cutoffs_single_objective,
    _get_tukey_cutoffs,
    AUTO_WINS_QUANTILE,
    Winsorize,
)
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.generators.base import Generator
from ax.generators.winsorization_config import WinsorizationConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_observations_with_invalid_value,
    get_optimization_config,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal

INF = float("inf")
OBSERVATION_DATA = [
    Observation(
        features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}),
        data=ObservationData(
            means=np.array([1.0, 2.0, 6.0]),
            covariance=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]]),
            metric_names=["a", "b", "b"],
        ),
        arm_name="1_1",
    )
]


class WinsorizeTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.obsd1 = ObservationData(
            metric_names=["m1", "m2", "m2"],
            means=np.array([0.0, 0.0, 1.0]),
            covariance=np.array([[1.0, 0.2, 0.4], [0.2, 2.0, 0.8], [0.4, 0.8, 3.0]]),
        )
        self.obsd2 = ObservationData(
            metric_names=["m1", "m1", "m2", "m2"],
            means=np.array([1.0, 2.0, 2.0, 1.0]),
            covariance=np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.2, 0.4],
                    [0.0, 0.2, 2.0, 0.8],
                    [0.0, 0.4, 0.8, 3.0],
                ]
            ),
        )
        self.observations = [
            Observation(features=ObservationFeatures({}), data=obsd)
            for obsd in [self.obsd1, self.obsd2]
        ]
        self.experiment_data = extract_experiment_data(
            experiment=get_experiment_with_observations(
                observations=[  # Same means as above.
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 2.0],
                    [2.0, 1.0],
                ],
                sems=[
                    [1.0, sqrt(2.0)],
                    [1.0, sqrt(3.0)],
                    [1.0, sqrt(2.0)],
                    [1.0, sqrt(3.0)],
                ],
            ),
            data_loader_config=DataLoaderConfig(),
        )

        self.t = Winsorize(
            search_space=None,
            experiment_data=self.experiment_data,
            config={
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.2)
            },
        )
        self.t1 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.8)
            },
        )
        self.t2 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={
                "winsorization_config": WinsorizationConfig(lower_quantile_margin=0.2)
            },
        )
        self.t3 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(
                        upper_quantile_margin=0.6, upper_boundary=1.9
                    ),
                }
            },
        )
        self.t4 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(lower_quantile_margin=0.8),
                    "m2": WinsorizationConfig(
                        lower_quantile_margin=0.8, lower_boundary=0.3
                    ),
                }
            },
        )

        self.obsd3 = ObservationData(
            metric_names=["m3", "m3", "m3", "m3"],
            means=np.array([0.0, 1.0, 5.0, 3.0]),
            covariance=np.eye(4),
        )
        self.obs3 = Observation(features=ObservationFeatures({}), data=self.obsd3)
        self.t5 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations) + deepcopy([self.obs3]),
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(lower_quantile_margin=0.4),
                }
            },
        )
        self.t6 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(
                        lower_quantile_margin=0.4, lower_boundary=0.0
                    ),
                }
            },
        )

    def test_Init(self) -> None:
        self.assertEqual(self.t.cutoffs["m1"], (-INF, 2.0))
        self.assertEqual(self.t.cutoffs["m2"], (-INF, 2.0))
        self.assertEqual(self.t1.cutoffs["m1"], (-INF, 1.0))
        self.assertEqual(self.t1.cutoffs["m2"], (-INF, 1.0))
        self.assertEqual(self.t2.cutoffs["m1"], (0.0, INF))
        self.assertEqual(self.t2.cutoffs["m2"], (0.0, INF))
        with self.assertRaisesRegex(
            DataRequiredError,
            "`Winsorize` transform requires non-empty data.",
        ):
            Winsorize(search_space=None, observations=[])
        with self.assertRaisesRegex(
            ValueError,
            "Transform config for `Winsorize` transform must be specified and "
            "non-empty when using winsorization.",
        ):
            Winsorize(
                search_space=None,
                observations=deepcopy(self.observations[:1]),
            )
        with self.assertRaisesRegex(
            UserInputError,
            "`derelativize_with_raw_status_quo` must be a boolean. Got 1234.",
        ):
            Winsorize(
                search_space=None,
                observations=deepcopy(self.observations[:1]),
                config={"derelativize_with_raw_status_quo": 1234},
            )

    def test_TransformObservations(self) -> None:
        observation_data = self.t1._transform_observation_data([deepcopy(self.obsd1)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t1._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 1.0, 1.0])
        observation_data = self.t2._transform_observation_data([deepcopy(self.obsd1)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t2._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [1.0, 2.0, 2.0, 1.0])

    def test_InitPercentileBounds(self) -> None:
        self.assertEqual(self.t3.cutoffs["m1"], (-INF, 1.0))
        self.assertEqual(self.t3.cutoffs["m2"], (-INF, 1.9))
        self.assertEqual(self.t4.cutoffs["m1"], (1.0, INF))
        self.assertEqual(self.t4.cutoffs["m2"], (0.3, INF))

    def test_TransformObservationsPercentileBounds(self) -> None:
        observation_data = self.t3._transform_observation_data([deepcopy(self.obsd1)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [0.0, 0.0, 1.0])
        observation_data = self.t3._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 1.9, 1.0])
        observation_data = self.t4._transform_observation_data([deepcopy(self.obsd1)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [1.0, 0.3, 1.0])
        observation_data = self.t4._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [1.0, 2.0, 2.0, 1.0])

    def test_TransformObservationsDifferentLowerUpper(self) -> None:
        observation_data = self.t5._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertEqual(self.t5.cutoffs["m1"], (-INF, 1.0))
        self.assertEqual(self.t5.cutoffs["m2"], (1.0, INF))
        self.assertEqual(self.t5.cutoffs["m3"], (-INF, INF))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])
        # Nothing should happen to m3
        observation_data = self.t5._transform_observation_data([deepcopy(self.obsd3)])[
            0
        ]
        self.assertListEqual(list(observation_data.means), [0.0, 1.0, 5.0, 3.0])
        # With winsorization boundaries
        observation_data = self.t6._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertEqual(self.t6.cutoffs["m1"], (-INF, 1.0))
        self.assertEqual(self.t6.cutoffs["m2"], (0.0, INF))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])

    def test_optimization_config_default(self) -> None:
        # Specify the winsorization
        optimization_config = get_optimization_config()
        percentiles = get_default_transform_cutoffs(
            optimization_config=optimization_config,
            winsorization_config={"m1": WinsorizationConfig(0.2, 0.0)},
        )
        self.assertDictEqual(percentiles, {"m1": (1, INF)})

    def test_tukey_cutoffs(self) -> None:
        Y = np.array([-100, 0, 1, 2, 50])
        self.assertEqual(_get_tukey_cutoffs(Y=Y, lower=True), -3.0)
        self.assertEqual(_get_tukey_cutoffs(Y=Y, lower=False), 5.0)

    def test_winsorize_outcome_constraints(self) -> None:
        metric_values = [-100, 0, 1, 2, 3, 4, 5, 6, 7, 50]
        ma, mb = Metric(name="a"), Metric(name="b")
        outcome_constraint_leq = OutcomeConstraint(
            metric=ma, op=ComparisonOp.LEQ, bound=10, relative=False
        )
        outcome_constraint_geq = OutcomeConstraint(
            metric=mb, op=ComparisonOp.GEQ, bound=-9, relative=False
        )
        # From above with a loose bound
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_leq],
        )
        self.assertEqual(cutoffs, (-INF, 23.5))
        # From above with a tight bound
        outcome_constraint_leq.bound = 2
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_leq],
        )
        self.assertEqual(cutoffs, (-INF, 13.5))
        # From below with a loose bound
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-31.5, INF))
        # From below with a tight bound
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-6.5, INF))
        # Both with the tight bounds
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_leq, outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-6.5, 13.5))

    def test_winsorization_single_objective(self) -> None:
        metric_values = [-100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 50.0]
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values, minimize=True
        )
        self.assertEqual(cutoffs, (-INF, 13.5))
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            metric_values=metric_values, minimize=False
        )
        self.assertEqual(cutoffs, (-6.5, INF))

    def test_winsorization_without_optimization_config(self) -> None:
        means = np.array([-100, 0, 1, 2, 3, 4, 5, 6, 7, 50])
        obsd = ObservationData(
            metric_names=["m1"] * 10,
            means=means,
            covariance=np.eye(10),
        )
        config = {
            "winsorization_config": {
                # pyre-fixme[6]: For 1st param expected `float` but got `None`.
                # pyre-fixme[6]: For 2nd param expected `float` but got `None`.
                "m1": WinsorizationConfig(None, None),
            }
        }
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        # None and 0.0 should be treated the same way
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.0, 0.0)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        # From above
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.0, 0.2)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-INF, 7))
        # From below
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.2, 0.0)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (0, INF))
        # Do both automatically
        config["winsorization_config"]["m1"] = WinsorizationConfig(
            AUTO_WINS_QUANTILE, AUTO_WINS_QUANTILE
        )
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-6.5, 13.5))
        # Add a second metric that shouldn't be winsorized
        config["winsorization_config"]["m1"] = WinsorizationConfig(
            0.0, AUTO_WINS_QUANTILE
        )
        obsd2 = ObservationData(
            metric_names=["m2"] * 10,
            means=means,
            covariance=np.eye(10),
        )
        transform = get_transform(
            observation_data=[deepcopy(obsd), deepcopy(obsd2)], config=config
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))
        self.assertEqual(transform.cutoffs["m2"], (-INF, INF))
        # Winsorize both
        config["winsorization_config"]["m2"] = WinsorizationConfig(0.2, 0.0)
        transform = get_transform(
            observation_data=[deepcopy(obsd), deepcopy(obsd2)], config=config
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))
        self.assertEqual(transform.cutoffs["m2"], (0.0, INF))

    def test_winsorization_with_optimization_config(self) -> None:
        obsd_1 = ObservationData(
            metric_names=["m1"] * 10,
            means=np.array([-100, 0, 1, 2, 3, 4, 5, 6, 7, 50]),
            covariance=np.eye(10),
        )
        obsd_2 = ObservationData(
            metric_names=["m2"] * 7,
            means=np.array([-10, 0, 1, 2, 3, 4, 47]),
            covariance=np.eye(7),
        )
        obsd_3 = ObservationData(
            metric_names=["m3"] * 6,
            means=np.array([-456, 1, 2, 3, 4, 9]),
            covariance=np.eye(6),
        )
        all_obsd = [obsd_1, obsd_2, obsd_3]
        m1 = Metric(name="m1", lower_is_better=False)
        m2 = Metric(name="m2", lower_is_better=True)
        m3 = Metric(name="m3")
        # Scalarized objective
        for minimize in [True, False]:
            optimization_config = OptimizationConfig(
                objective=ScalarizedObjective(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[1, -1],
                    minimize=minimize,
                )
            )
            transform = get_transform(
                observation_data=deepcopy(all_obsd),
                optimization_config=optimization_config,
            )
            if minimize:
                self.assertEqual(
                    transform.cutoffs,
                    {"m1": (-INF, 13.5), "m2": (-6.0, INF), "m3": (-INF, INF)},
                )
            else:
                self.assertEqual(
                    transform.cutoffs,
                    {"m1": (-6.5, INF), "m2": (-INF, 10.0), "m3": (-INF, INF)},
                )
        # Simple single-objective problem
        optimization_config = OptimizationConfig(
            objective=Objective(metric=m2, minimize=True)
        )
        transform = get_transform(
            observation_data=deepcopy(all_obsd), optimization_config=optimization_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add a relative constraint, which should raise an error
        outcome_constraint = OutcomeConstraint(
            metric=m1, op=ComparisonOp.LEQ, bound=3, relative=True
        )
        optimization_config = OptimizationConfig(
            objective=Objective(metric=m2, minimize=True),
            outcome_constraints=[outcome_constraint],
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "Automatic winsorization doesn't support relative outcome constraints "
            "or objective thresholds when `derelativize_with_raw_status_quo` is not "
            "set to `True`.",
        ):
            get_transform(
                observation_data=deepcopy(all_obsd),
                optimization_config=optimization_config,
            )
        # Make the constraint absolute, which should trigger winsorization
        optimization_config.outcome_constraints[0].relative = False
        transform = get_transform(
            observation_data=deepcopy(all_obsd), optimization_config=optimization_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))  # 6 + 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Change to a GEQ constraint
        optimization_config.outcome_constraints[0].op = ComparisonOp.GEQ
        transform = get_transform(
            observation_data=deepcopy(all_obsd), optimization_config=optimization_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add a scalarized outcome constraint which should print a warning
        optimization_config.outcome_constraints = [
            ScalarizedOutcomeConstraint(
                metrics=[m1, m3], op=ComparisonOp.GEQ, bound=3, relative=False
            )
        ]
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd),
                optimization_config=optimization_config,
            )
        for i in range(2):
            self.assertTrue(
                "Automatic winsorization isn't supported for a "
                "`ScalarizedOutcomeConstraint`. Specify the winsorization settings "
                f"manually if you want to winsorize metric m{['1', '3'][i]}."
                in [str(w.message) for w in ws]
            )
        # Multi-objective without objective thresholds should warn and winsorize
        moo_objective = MultiObjective(
            [Objective(m1, minimize=False), Objective(m2, minimize=True)]
        )
        optimization_config = MultiObjectiveOptimizationConfig(objective=moo_objective)
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd),
                optimization_config=optimization_config,
            )
        for _ in range(2):
            self.assertTrue(
                "Encountered a `MultiObjective` without objective thresholds. We "
                "will winsorize each objective separately. We strongly recommend "
                "specifying the objective thresholds when using multi-objective "
                "optimization." in [str(w.message) for w in ws]
            )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add relative objective thresholds (should raise an error)
        objective_thresholds = [
            ObjectiveThreshold(m1, 3, relative=True),
            ObjectiveThreshold(m2, 4, relative=True),
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=moo_objective,
            objective_thresholds=objective_thresholds,
            outcome_constraints=[],
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "Automatic winsorization doesn't support relative outcome constraints or "
            "objective thresholds when `derelativize_with_raw_status_quo` is not set "
            "to `True`.",
        ):
            get_transform(
                observation_data=deepcopy(all_obsd),
                optimization_config=optimization_config,
            )
        # Make the objective thresholds absolute (should trigger winsorization)
        optimization_config.objective_thresholds[0].relative = False
        optimization_config.objective_thresholds[1].relative = False
        transform = get_transform(
            observation_data=deepcopy(all_obsd), optimization_config=optimization_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add an absolute outcome constraint
        optimization_config.outcome_constraints = [
            OutcomeConstraint(metric=m3, op=ComparisonOp.GEQ, bound=3, relative=False)
        ]
        transform = get_transform(
            observation_data=deepcopy(all_obsd), optimization_config=optimization_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-3.5, INF))  # 1 - 1.5 * 3

    @mock.patch(
        "ax.adapter.base.observations_from_data",
        autospec=True,
        return_value=(OBSERVATION_DATA),
    )
    def test_relative_constraints(self, _: mock.Mock) -> None:
        # Adapter with in-design status quo
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0, 20),
                RangeParameter("y", ParameterType.FLOAT, 0, 20),
            ]
        )
        objective = Objective(Metric("c"), minimize=False)

        # Test with relative constraint, in-design status quo
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=-10, relative=True
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=-10,
                    relative=True,
                ),
            ],
        )
        adapter = Adapter(
            experiment=Experiment(search_space=search_space),
            generator=Generator(),
            transforms=[],
            data=Data(),
            optimization_config=oc,
        )
        with self.assertRaisesRegex(
            DataRequiredError, "model was not fit with status quo"
        ):
            Winsorize(
                search_space=search_space,
                observations=OBSERVATION_DATA,
                adapter=adapter,
                config={"derelativize_with_raw_status_quo": True},
            )

        adapter = Adapter(
            experiment=Experiment(
                search_space=search_space,
                status_quo=Arm(parameters={"x": 2.0, "y": 10.0}, name="1_1"),
            ),
            generator=Generator(),
            transforms=[],
            data=Data(),
            optimization_config=oc,
        )
        with self.assertRaisesRegex(
            UnsupportedError, "`derelativize_with_raw_status_quo` is not set to `True`"
        ):
            Winsorize(
                search_space=search_space,
                observations=OBSERVATION_DATA,
                adapter=adapter,
            )
        t = Winsorize(
            search_space=search_space,
            observations=OBSERVATION_DATA,
            adapter=adapter,
            config={"derelativize_with_raw_status_quo": True},
        )
        self.assertDictEqual(t.cutoffs, {"a": (-INF, 3.5), "b": (-INF, 12.0)})

    def test_non_finite_data_raises(self) -> None:
        for invalid_value in [float("nan"), float("inf")]:
            observations = get_observations_with_invalid_value(
                invalid_value=invalid_value
            )
            config: dict[str, Any] = {
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.2)
            }
            with self.assertRaisesRegex(
                ValueError, f"Non-finite data found for metric m1: {invalid_value}"
            ):
                Winsorize(search_space=None, observations=observations, config=config)

    def test_transform_experiment_data(self) -> None:
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Data is within winsorization bounds. No change.
        self.assertEqual(self.experiment_data, transformed_data)

        # Modify the cutoffs to check for winsorization.
        self.t.cutoffs["m1"] = (0.5, 1.0)
        self.t.cutoffs["m2"] = (-INF, 1.5)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Check that the data is winsorized correctly.
        expected_mean = DataFrame(
            index=transformed_data.observation_data.index,
            columns=transformed_data.observation_data["mean"].columns,
            data=[
                [0.5, 0.0],
                [0.5, 1.0],
                [1.0, 1.5],
                [1.0, 1.0],
            ],
        )
        assert_frame_equal(transformed_data.observation_data["mean"], expected_mean)
        # Sem and arm data are not modified.
        assert_frame_equal(
            transformed_data.observation_data["sem"],
            self.experiment_data.observation_data["sem"],
        )
        assert_frame_equal(transformed_data.arm_data, self.experiment_data.arm_data)


def get_transform(
    observation_data: list[ObservationData],
    config: dict[str, Any] | None = None,
    optimization_config: OptimizationConfig | None = None,
) -> Winsorize:
    observations = [
        Observation(features=ObservationFeatures({}), data=obsd)
        for obsd in observation_data
    ]
    if optimization_config is not None:
        adapter = _wrap_optimization_config_in_adapter(
            optimization_config=optimization_config
        )
        return Winsorize(
            search_space=None,
            observations=observations,
            config=config,
            adapter=adapter,
        )
    return Winsorize(
        search_space=None,
        observations=observations,
        config=config,
    )


def get_default_transform_cutoffs(
    optimization_config: OptimizationConfig,
    winsorization_config: dict[str, WinsorizationConfig] | None = None,
    obs_data_len: SupportsIndex = 6,
) -> dict[str, tuple[float, float]]:
    obsd = ObservationData(
        metric_names=["m1"] * obs_data_len,
        means=np.array(range(obs_data_len)),
        # pyre-fixme[6]: For 1st argument expected `int` but got `SupportsIndex`.
        covariance=np.eye(obs_data_len),
    )
    obs = Observation(features=ObservationFeatures({}), data=obsd)
    adapter = _wrap_optimization_config_in_adapter(
        optimization_config=optimization_config
    )
    transform = Winsorize(
        search_space=None,
        observations=[deepcopy(obs)],
        adapter=adapter,
        config={
            "winsorization_config": winsorization_config,
        },
    )
    return transform.cutoffs


def _wrap_optimization_config_in_adapter(
    optimization_config: OptimizationConfig,
) -> Adapter:
    return Adapter(
        experiment=Experiment(search_space=SearchSpace(parameters=[])),
        generator=Generator(),
        optimization_config=optimization_config,
    )
