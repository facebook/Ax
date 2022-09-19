#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from copy import deepcopy

import numpy as np
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
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.modelbridge.transforms.winsorize import (
    _get_auto_winsorization_cutoffs_outcome_constraint,
    _get_auto_winsorization_cutoffs_single_objective,
    _get_tukey_cutoffs,
    AUTO_WINS_QUANTILE,
    WinsorizationConfig,
    Winsorize,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_optimization_config
from typing_extensions import SupportsIndex


class WinsorizeTransformTest(TestCase):
    def setUp(self) -> None:
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
        self.t = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={  # pyre-ignore
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.2)
            },
        )
        self.t1 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={  # pyre-ignore
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.8)
            },
        )
        self.t2 = Winsorize(
            search_space=None,
            observations=deepcopy(self.observations),
            config={  # pyre-ignore
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

    def testInit(self) -> None:
        self.assertEqual(self.t.cutoffs["m1"], (-float("inf"), 2.0))
        self.assertEqual(self.t.cutoffs["m2"], (-float("inf"), 2.0))
        self.assertEqual(self.t1.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t1.cutoffs["m2"], (-float("inf"), 1.0))
        self.assertEqual(self.t2.cutoffs["m1"], (0.0, float("inf")))
        self.assertEqual(self.t2.cutoffs["m2"], (0.0, float("inf")))
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
            "Expected `optimization_config` of type `OptimizationConfig` "
            "but got type `<class 'int'>.",
        ):
            Winsorize(
                search_space=None,
                observations=deepcopy(self.observations[:1]),
                config={"optimization_config": 1234},
            )

    def testTransformObservations(self) -> None:
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

    def testInitPercentileBounds(self) -> None:
        self.assertEqual(self.t3.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t3.cutoffs["m2"], (-float("inf"), 1.9))
        self.assertEqual(self.t4.cutoffs["m1"], (1.0, float("inf")))
        self.assertEqual(self.t4.cutoffs["m2"], (0.3, float("inf")))

    def testTransformObservationsPercentileBounds(self) -> None:
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

    def testTransformObservationsDifferentLowerUpper(self) -> None:
        observation_data = self.t5._transform_observation_data([deepcopy(self.obsd2)])[
            0
        ]
        self.assertEqual(self.t5.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t5.cutoffs["m2"], (1.0, float("inf")))
        self.assertEqual(self.t5.cutoffs["m3"], (-float("inf"), float("inf")))
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
        self.assertEqual(self.t6.cutoffs["m1"], (-float("inf"), 1.0))
        self.assertEqual(self.t6.cutoffs["m2"], (0.0, float("inf")))
        self.assertListEqual(list(observation_data.means), [1.0, 1.0, 2.0, 1.0])

    def test_optimization_config_default(self) -> None:
        # Specify the winsorization
        optimization_config = get_optimization_config()
        percentiles = get_default_transform_cutoffs(
            optimization_config=optimization_config,
            winsorization_config={"m1": WinsorizationConfig(0.2, 0.0)},
        )
        self.assertDictEqual(percentiles, {"m1": (1, float("inf"))})

        # Don't winsorize if optimization_config is mistyped
        optimization_config = "not an optimization config"
        with self.assertRaisesRegex(
            UserInputError,
            "Expected `optimization_config` of type `OptimizationConfig`",
        ):
            get_default_transform_cutoffs(optimization_config=optimization_config)

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
        self.assertEqual(cutoffs, (-float("inf"), 23.5))
        # From above with a tight bound
        outcome_constraint_leq.bound = 2
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_leq],
        )
        self.assertEqual(cutoffs, (-float("inf"), 13.5))
        # From below with a loose bound
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-31.5, float("inf")))
        # From below with a tight bound
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-6.5, float("inf")))
        # Both with the tight bounds
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            outcome_constraints=[outcome_constraint_leq, outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-6.5, 13.5))

    def test_winsorization_single_objective(self) -> None:
        metric_values = [-100, 0, 1, 2, 3, 4, 5, 6, 7, 50]
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            minimize=True,
        )
        self.assertEqual(cutoffs, (-float("inf"), 13.5))
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            # pyre-fixme[6]: For 1st param expected `List[float]` but got `List[int]`.
            metric_values=metric_values,
            minimize=False,
        )
        self.assertEqual(cutoffs, (-6.5, float("inf")))

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
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), float("inf")))
        # None and 0.0 should be treated the same way
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.0, 0.0)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), float("inf")))
        # From above
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.0, 0.2)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), 7))
        # From below
        config["winsorization_config"]["m1"] = WinsorizationConfig(0.2, 0.0)
        transform = get_transform(observation_data=[deepcopy(obsd)], config=config)
        self.assertEqual(transform.cutoffs["m1"], (0, float("inf")))
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
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), 13.5))
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), float("inf")))
        # Winsorize both
        config["winsorization_config"]["m2"] = WinsorizationConfig(0.2, 0.0)
        transform = get_transform(
            observation_data=[deepcopy(obsd), deepcopy(obsd2)], config=config
        )
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), 13.5))
        self.assertEqual(transform.cutoffs["m2"], (0.0, float("inf")))

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
        # Scalarized objective shouldn't be winsorized but should print a warning
        config = {
            "optimization_config": OptimizationConfig(
                objective=ScalarizedObjective(metrics=[m1, m2])
            )
        }
        warnings.simplefilter("always", append=True)
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd), config=config
            )
            for i in range(2):
                print([w.message for w in ws])
                self.assertTrue(
                    "Automatic winsorization isn't supported for ScalarizedObjective. "
                    "Specify the winsorization settings manually if you want to "
                    f"winsorize metric m{i + 1}." in [str(w.message) for w in ws]
                )
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), float("inf")))
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), float("inf")))
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Simple single-objective problem
        config["optimization_config"].objective = Objective(metric=m2, minimize=True)
        transform = get_transform(observation_data=deepcopy(all_obsd), config=config)
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), float("inf")))
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Add a relative constraint, which should raise an error
        outcome_constraint = OutcomeConstraint(
            metric=m1, op=ComparisonOp.LEQ, bound=3, relative=True
        )
        config = {
            "optimization_config": OptimizationConfig(
                objective=Objective(metric=m2, minimize=True),
                outcome_constraints=[outcome_constraint],
            )
        }
        with self.assertRaisesRegex(
            UnsupportedError,
            "Automatic winsorization doesn't support relative outcome constraints. "
            "Make sure a `Derelativize` transform is applied first.",
        ):
            get_transform(observation_data=deepcopy(all_obsd), config=config)
        # Make the constraint absolute, which should trigger winsorization
        config["optimization_config"].outcome_constraints[0].relative = False
        transform = get_transform(observation_data=deepcopy(all_obsd), config=config)
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), 13.5))  # 6 + 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Change to a GEQ constraint
        config["optimization_config"].outcome_constraints[0].op = ComparisonOp.GEQ
        transform = get_transform(observation_data=deepcopy(all_obsd), config=config)
        self.assertEqual(transform.cutoffs["m1"], (-6.5, float("inf")))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Add a scalarized outcome constraint which should print a warning
        config["optimization_config"].outcome_constraints = [
            ScalarizedOutcomeConstraint(
                metrics=[m1, m3], op=ComparisonOp.GEQ, bound=3, relative=False
            )
        ]
        warnings.simplefilter("always", append=True)
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd), config=config
            )
            for i in range(2):
                self.assertTrue(
                    "Automatic winsorization isn't supported for a "
                    "`ScalarizedOutcomeConstraint`. Specify the winsorization settings "
                    f"manually if you want to winsorize metric m{['1', '3'][i]}."
                    in [str(w.message) for w in ws]
                )
        # Making the constraint relative should print the same warning
        config["optimization_config"].outcome_constraints[0].relative = True
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd), config=config
            )
            for i in range(2):
                self.assertTrue(
                    "Automatic winsorization isn't supported for a "
                    "`ScalarizedOutcomeConstraint`. Specify the winsorization settings "
                    f"manually if you want to winsorize metric m{['1', '3'][i]}."
                    in [str(w.message) for w in ws]
                )
        # Multi-objective without objective thresholds (should print a warning)
        moo_objective = MultiObjective(
            [Objective(m1, minimize=False), Objective(m2, minimize=True)]
        )
        moo_config = {
            "optimization_config": MultiObjectiveOptimizationConfig(
                objective=moo_objective
            )
        }
        warnings.simplefilter("always", append=True)
        with warnings.catch_warnings(record=True) as ws:
            transform = get_transform(
                observation_data=deepcopy(all_obsd), config=moo_config
            )
            for i in range(2):
                self.assertTrue(
                    "Automatic winsorization isn't supported for an objective in "
                    "`MultiObjective` without objective thresholds. Specify the "
                    "winsorization settings manually if you want to winsorize "
                    f"metric m{i + 1}." in [str(w.message) for w in ws]
                )
        self.assertEqual(transform.cutoffs["m1"], (-float("inf"), float("inf")))
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), float("inf")))
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Add relative objective thresholds (should raise an error)
        objective_thresholds = [
            ObjectiveThreshold(m1, 3, relative=True),
            ObjectiveThreshold(m2, 4, relative=True),
        ]
        moo_config = {
            "optimization_config": MultiObjectiveOptimizationConfig(
                objective=moo_objective,
                objective_thresholds=objective_thresholds,
                outcome_constraints=[],
            )
        }
        with self.assertRaisesRegex(
            UnsupportedError,
            "Automatic winsorization doesn't support relative objective thresholds. "
            "Make sure a `Derelevatize` transform is applied first.",
        ):
            get_transform(observation_data=deepcopy(all_obsd), config=moo_config)
        # Make the objective thresholds absolute (should trigger winsorization)
        moo_config["optimization_config"].objective_thresholds[0].relative = False
        moo_config["optimization_config"].objective_thresholds[1].relative = False
        transform = get_transform(
            observation_data=deepcopy(all_obsd), config=moo_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, float("inf")))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-float("inf"), float("inf")))
        # Add an absolute outcome constraint
        moo_config["optimization_config"].outcome_constraints = [
            OutcomeConstraint(metric=m3, op=ComparisonOp.GEQ, bound=3, relative=False)
        ]
        transform = get_transform(
            observation_data=deepcopy(all_obsd), config=moo_config
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, float("inf")))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-float("inf"), 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-3.5, float("inf")))  # 1 - 1.5 * 3


# pyre-fixme[2]: Parameter must be annotated.
def get_transform(observation_data, config) -> Winsorize:
    observations = [
        Observation(features=ObservationFeatures({}), data=obsd)
        for obsd in observation_data
    ]
    return Winsorize(
        search_space=None,
        observations=observations,
        config=config,
    )


# pyre-fixme[3]: Return type must be annotated.
def get_default_transform_cutoffs(
    # pyre-fixme[2]: Parameter must be annotated.
    optimization_config,
    # pyre-fixme[2]: Parameter must be annotated.
    winsorization_config=None,
    obs_data_len: SupportsIndex = 6,
):
    obsd = ObservationData(
        metric_names=["m1"] * obs_data_len,
        means=np.array(range(obs_data_len)),
        covariance=np.eye(obs_data_len),
    )
    obs = Observation(features=ObservationFeatures({}), data=obsd)
    transform = Winsorize(
        search_space=None,
        observations=[deepcopy(obs)],
        config={
            "optimization_config": optimization_config,
            "winsorization_config": winsorization_config,
        },
    )
    return transform.cutoffs
