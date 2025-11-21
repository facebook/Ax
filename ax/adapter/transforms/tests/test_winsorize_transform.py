#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from copy import deepcopy
from functools import partial
from math import sqrt
from typing import Any

import numpy as np
from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import ExperimentData, extract_experiment_data
from ax.adapter.transforms.winsorize import (
    _get_auto_winsorization_cutoffs_outcome_constraint,
    _get_auto_winsorization_cutoffs_single_objective,
    _get_tukey_cutoffs,
    AUTO_WINS_QUANTILE,
    Winsorize,
)
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
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
from ax.exceptions.core import AxOptimizationWarning, DataRequiredError, UserInputError
from ax.generators.base import Generator
from ax.generators.winsorization_config import WinsorizationConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_optimization_config,
)
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pyre_extensions import none_throws

INF = float("inf")


class WinsorizeTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        observations = [[0.0, 0.0], [float("nan"), 1.0], [1.0, 2.0], [2.0, 1.0]]
        sems = [
            [1.0, sqrt(2.0)],
            [float("nan"), sqrt(3.0)],
            [1.0, sqrt(2.0)],
            [1.0, sqrt(3.0)],
        ]
        self.experiment_data = extract_experiment_data(
            experiment=get_experiment_with_observations(
                observations=observations, sems=sems
            ),
            data_loader_config=DataLoaderConfig(),
        )
        self.observations = self.experiment_data.convert_to_list_of_observations()
        self.t = self._get_transform(
            config={
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.2)
            },
        )
        self.t1 = self._get_transform(
            config={
                "winsorization_config": WinsorizationConfig(upper_quantile_margin=0.8)
            },
        )
        self.t2 = self._get_transform(
            config={
                "winsorization_config": WinsorizationConfig(lower_quantile_margin=0.2)
            },
        )
        self.t3 = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(
                        upper_quantile_margin=0.6, upper_boundary=1.9
                    ),
                }
            },
        )
        self.t4 = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(lower_quantile_margin=0.8),
                    "m2": WinsorizationConfig(
                        lower_quantile_margin=0.8, lower_boundary=0.3
                    ),
                }
            },
        )
        self.t5 = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(lower_quantile_margin=0.4),
                }
            },
        )
        self.t6 = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(upper_quantile_margin=0.6),
                    "m2": WinsorizationConfig(
                        lower_quantile_margin=0.4, lower_boundary=0.0
                    ),
                }
            },
        )
        self.values = [-100.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 50.0]

    def _get_transform(
        self, config: dict[str, Any], experiment_data: ExperimentData | None = None
    ) -> Winsorize:
        return Winsorize(
            search_space=None,
            experiment_data=experiment_data or self.experiment_data,
            config=config,
        )

    def test_Init(self) -> None:
        for t, expected_cutoffs in [
            (self.t, {"m1": (-INF, 2.0), "m2": (-INF, 2.0)}),
            (self.t1, {"m1": (-INF, 1.0), "m2": (-INF, 1.0)}),
            (self.t2, {"m1": (0.0, INF), "m2": (0.0, INF)}),
            (self.t3, {"m1": (-INF, 1.0), "m2": (-INF, 1.9)}),
            (self.t4, {"m1": (1.0, INF), "m2": (0.3, INF)}),
            (self.t5, {"m1": (-INF, 1.0), "m2": (1.0, INF)}),
            (self.t6, {"m1": (-INF, 1.0), "m2": (0.0, INF)}),
        ]:
            self.assertEqual(t.cutoffs, expected_cutoffs)
        with self.assertRaisesRegex(
            DataRequiredError,
            "`Winsorize` transform requires non-empty data.",
        ):
            Winsorize(search_space=None)
        # Initialize with no opt config.
        t = Winsorize(
            search_space=None,
            experiment_data=self.experiment_data,
        )
        self.assertEqual(t.cutoffs, {})
        with self.assertRaisesRegex(
            UserInputError,
            "`derelativize_with_raw_status_quo` must be a boolean. Got 1234.",
        ):
            Winsorize(
                search_space=None,
                experiment_data=self.experiment_data,
                config={"derelativize_with_raw_status_quo": 1234},
            )

    def test_TransformObservations(self) -> None:
        obsd_list = [obs.data for obs in self.observations]
        for t, expected in [
            (self.t1, [[0.0, 0.0], [1.0], [1.0, 1.0], [1.0, 1.0]]),
            (self.t2, [[0.0, 0.0], [1.0], [1.0, 2.0], [2.0, 1.0]]),
            (self.t3, [[0.0, 0.0], [1.0], [1.0, 1.9], [1.0, 1.0]]),
            (self.t4, [[1.0, 0.3], [1.0], [1.0, 2.0], [2.0, 1.0]]),
            (self.t5, [[0.0, 1.0], [1.0], [1.0, 2.0], [1.0, 1.0]]),
            (self.t6, [[0.0, 0.0], [1.0], [1.0, 2.0], [1.0, 1.0]]),
        ]:
            observation_data = t._transform_observation_data(deepcopy(obsd_list))
            tf_means = [obsd.means.tolist() for obsd in observation_data]
            self.assertListEqual(tf_means, expected)

    def test_optimization_config_default(self) -> None:
        # Specify the winsorization
        experiment = get_experiment_with_observations(
            observations=[[m, 0.2] for m in range(6)],
            optimization_config=get_optimization_config(),
        )
        adapter = Adapter(experiment=experiment, generator=Generator())
        percentiles = Winsorize(
            search_space=None,
            adapter=adapter,
            experiment_data=adapter.get_training_data(),
            config={
                "winsorization_config": {"m1": WinsorizationConfig(0.2, 0.0)},
            },
        ).cutoffs
        self.assertDictEqual(percentiles, {"m1": (1, INF), "m2": (-INF, INF)})

    def test_tukey_cutoffs(self) -> None:
        Y = np.array([-100, 0, 1, 2, 50])
        self.assertEqual(_get_tukey_cutoffs(Y=Y, lower=True), -3.0)
        self.assertEqual(_get_tukey_cutoffs(Y=Y, lower=False), 5.0)

    def test_winsorize_outcome_constraints(self) -> None:
        ma, mb = Metric(name="a"), Metric(name="b")
        outcome_constraint_leq = OutcomeConstraint(
            metric=ma, op=ComparisonOp.LEQ, bound=10, relative=False
        )
        outcome_constraint_geq = OutcomeConstraint(
            metric=mb, op=ComparisonOp.GEQ, bound=-9, relative=False
        )
        # From above with a loose bound
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=self.values, outcome_constraints=[outcome_constraint_leq]
        )
        self.assertEqual(cutoffs, (-INF, 23.5))
        # From above with a tight bound
        outcome_constraint_leq.bound = 2
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=self.values, outcome_constraints=[outcome_constraint_leq]
        )
        self.assertEqual(cutoffs, (-INF, 13.5))
        # From below with a loose bound
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=self.values, outcome_constraints=[outcome_constraint_geq]
        )
        self.assertEqual(cutoffs, (-31.5, INF))
        # From below with a tight bound
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=self.values, outcome_constraints=[outcome_constraint_geq]
        )
        self.assertEqual(cutoffs, (-6.5, INF))
        # Both with the tight bounds
        outcome_constraint_geq.bound = 5
        cutoffs = _get_auto_winsorization_cutoffs_outcome_constraint(
            metric_values=self.values,
            outcome_constraints=[outcome_constraint_leq, outcome_constraint_geq],
        )
        self.assertEqual(cutoffs, (-6.5, 13.5))

    def test_winsorization_single_objective(self) -> None:
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            metric_values=self.values, minimize=True
        )
        self.assertEqual(cutoffs, (-INF, 13.5))
        cutoffs = _get_auto_winsorization_cutoffs_single_objective(
            metric_values=self.values, minimize=False
        )
        self.assertEqual(cutoffs, (-6.5, INF))

    def test_winsorization_without_optimization_config(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[[o] for o in self.values]
        )
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )
        get_transform = partial(self._get_transform, experiment_data=experiment_data)
        transform = get_transform(
            config={"winsorization_config": {"m1": WinsorizationConfig(None, None)}}
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        # None and 0.0 should be treated the same way
        transform = get_transform(
            config={"winsorization_config": {"m1": WinsorizationConfig(0.0, 0.0)}}
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        # From above
        transform = get_transform(
            config={"winsorization_config": {"m1": WinsorizationConfig(0.0, 0.2)}}
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 7))
        # From below
        transform = get_transform(
            config={"winsorization_config": {"m1": WinsorizationConfig(0.2, 0.0)}}
        )
        self.assertEqual(transform.cutoffs["m1"], (0, INF))
        # Do both automatically
        transform = get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(AUTO_WINS_QUANTILE, AUTO_WINS_QUANTILE)
                }
            }
        )
        self.assertEqual(transform.cutoffs["m1"], (-6.5, 13.5))
        # Add a second metric that shouldn't be winsorized
        experiment = get_experiment_with_observations(
            observations=[[o, o] for o in self.values]
        )
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )
        transform = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(0.0, AUTO_WINS_QUANTILE)
                }
            },
            experiment_data=experiment_data,
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))
        self.assertEqual(transform.cutoffs["m2"], (-INF, INF))
        # Winsorize both
        transform = self._get_transform(
            config={
                "winsorization_config": {
                    "m1": WinsorizationConfig(0.0, AUTO_WINS_QUANTILE),
                    "m2": WinsorizationConfig(0.2, 0.0),
                }
            },
            experiment_data=experiment_data,
        )
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))
        self.assertEqual(transform.cutoffs["m2"], (0.0, INF))

    def test_winsorization_with_optimization_config(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[
                [-100.0, -10.0, -456.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 3.0],
                [3.0, 3.0, 4.0],
                [4.0, 4.0, 9.0],
                [5.0, 47.0, float("nan")],
                [6.0, float("nan"), float("nan")],
                [7.0, float("nan"), float("nan")],
                [50.0, float("nan"), float("nan")],
            ],
            constrained=True,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=DataLoaderConfig(),
        )
        # Scalarized objective
        for minimize in [True, False]:
            experiment.optimization_config = OptimizationConfig(
                objective=ScalarizedObjective(
                    metrics=[Metric(name="m1"), Metric(name="m2")],
                    weights=[1, -1],
                    minimize=minimize,
                )
            )
            adapter = Adapter(experiment=experiment, generator=Generator())
            transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
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
        m1 = Metric(name="m1", lower_is_better=False)
        m2 = Metric(name="m2", lower_is_better=True)
        m3 = Metric(name="m3")
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=m2, minimize=True)
        )
        adapter = Adapter(experiment=experiment, generator=Generator())
        transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add a relative constraint, which should warn and skip relative metrics.
        outcome_constraint = OutcomeConstraint(
            metric=m1, op=ComparisonOp.LEQ, bound=3, relative=True
        )
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=m2, minimize=True),
            outcome_constraints=[outcome_constraint],
        )
        adapter = Adapter(experiment=experiment, generator=Generator())
        with self.assertWarnsRegex(
            AxOptimizationWarning,
            "Automatic winsorization doesn't support relative outcome constraints "
            "or objective thresholds when `derelativize_with_raw_status_quo` is not "
            "set to `True`.",
        ):
            transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-INF, INF))
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Make the constraint absolute, which should trigger winsorization
        outcome_constraint.relative = False
        transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-INF, 13.5))  # 6 + 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Change to a GEQ constraint
        outcome_constraint.op = ComparisonOp.GEQ
        transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add a scalarized outcome constraint which should print a warning
        none_throws(experiment.optimization_config).outcome_constraints = [
            ScalarizedOutcomeConstraint(
                metrics=[m1, m3], op=ComparisonOp.GEQ, bound=3, relative=False
            )
        ]
        with warnings.catch_warnings(record=True) as ws:
            transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
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
        experiment._optimization_config = optimization_config
        adapter = Adapter(experiment=experiment, generator=Generator())
        with warnings.catch_warnings(record=True) as ws:
            transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
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
        # Add relative objective thresholds. Should warn and skip.
        objective_thresholds = [
            ObjectiveThreshold(m1, 3, relative=True),
            ObjectiveThreshold(m2, 4, relative=True),
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=moo_objective,
            objective_thresholds=objective_thresholds,
            outcome_constraints=[],
        )
        experiment.optimization_config = optimization_config
        adapter = Adapter(experiment=experiment, generator=Generator())
        with self.assertWarnsRegex(
            AxOptimizationWarning,
            "Automatic winsorization doesn't support relative outcome constraints or "
            "objective thresholds when `derelativize_with_raw_status_quo` is not set "
            "to `True`.",
        ):
            transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        for i in range(1, 4):
            self.assertEqual(transform.cutoffs[f"m{i}"], (-INF, INF))
        # Make the objective thresholds absolute (should trigger winsorization)
        optimization_config.objective_thresholds[0].relative = False
        optimization_config.objective_thresholds[1].relative = False
        transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-INF, INF))
        # Add an absolute outcome constraint
        optimization_config.outcome_constraints = [
            OutcomeConstraint(metric=m3, op=ComparisonOp.GEQ, bound=3, relative=False)
        ]
        transform = Winsorize(experiment_data=experiment_data, adapter=adapter)
        self.assertEqual(transform.cutoffs["m1"], (-6.5, INF))  # 1 - 1.5 * 5
        self.assertEqual(transform.cutoffs["m2"], (-INF, 10.0))  # 4 + 1.5 * 4
        self.assertEqual(transform.cutoffs["m3"], (-3.5, INF))  # 1 - 1.5 * 3

    def test_relative_constraints(self) -> None:
        # Adapter with in-design status quo
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0, 20),
                RangeParameter("y", ParameterType.FLOAT, 0, 20),
            ]
        )
        # Test with relative constraint, in-design status quo
        oc = OptimizationConfig(
            objective=Objective(Metric("c"), minimize=False),
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
        experiment = get_experiment_with_observations(
            observations=[[0.5, 0.5, 0.5]],
            optimization_config=oc,
            search_space=search_space,
        )
        adapter = Adapter(experiment=experiment, generator=Generator())
        with self.assertRaisesRegex(
            DataRequiredError, "model was not fit with status quo"
        ):
            Winsorize(
                search_space=search_space,
                experiment_data=adapter.get_training_data(),
                adapter=adapter,
                config={"derelativize_with_raw_status_quo": True},
            )

        sq_arm = Arm(parameters={"x": 2.0, "y": 10.0}, name="1_1")
        experiment.status_quo = sq_arm
        t = (
            experiment.new_trial()
            .add_arm(sq_arm)
            .mark_running(no_runner_required=True)
            .mark_completed()
        )
        data = Data(
            df=DataFrame.from_records(
                [
                    {
                        "arm_name": sq_arm.name,
                        "metric_name": metric_name,
                        "mean": mean,
                        "sem": sem,
                        "trial_index": t.index,
                        "metric_signature": metric_name,
                    }
                    for metric_name, mean, sem in (("a", 1.0, 2.0), ("b", 2.0, 4.0))
                ]
            )
        )
        experiment.attach_data(data)

        adapter = Adapter(experiment=experiment, generator=Generator())
        # Warns and skips without `derelativize_with_raw_status_quo`.
        with self.assertWarnsRegex(
            AxOptimizationWarning,
            "`derelativize_with_raw_status_quo` is not set to `True`",
        ):
            t = Winsorize(
                search_space=search_space,
                experiment_data=adapter.get_training_data(),
                adapter=adapter,
            )
        self.assertDictEqual(
            t.cutoffs, {"a": (-INF, INF), "b": (-INF, INF), "c": (0.5, INF)}
        )
        # Winsorizes with `derelativize_with_raw_status_quo`.
        t = Winsorize(
            search_space=search_space,
            experiment_data=adapter.get_training_data(),
            adapter=adapter,
            config={"derelativize_with_raw_status_quo": True},
        )
        self.assertDictEqual(
            t.cutoffs, {"a": (-INF, 4.25), "b": (-INF, 4.25), "c": (0.5, INF)}
        )

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
                [float("nan"), 1.0],
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
