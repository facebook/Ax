#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from unittest import mock

import numpy as np
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.utils.common.testutils import TestCase


class DerelativizeTransformTest(TestCase):
    def setUp(self) -> None:
        m = mock.patch.object(ModelBridge, "__abstractmethods__", frozenset())
        self.addCleanup(m.stop)
        m.start()

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=(
            [
                Observation(
                    features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}),
                    data=ObservationData(
                        means=np.array([1.0, 2.0, 6.0]),
                        covariance=np.array(
                            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
                        ),
                        metric_names=["a", "b", "b"],
                    ),
                    arm_name="1_1",
                ),
                Observation(
                    features=ObservationFeatures(parameters={"x": None, "y": None}),
                    data=ObservationData(
                        means=np.array([1.0, 2.0, 6.0]),
                        covariance=np.array(
                            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
                        ),
                        metric_names=["a", "b", "b"],
                    ),
                    arm_name="1_2",
                ),
            ]
        ),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    @mock.patch(
        "ax.modelbridge.base.ModelBridge._predict",
        autospec=True,
        return_value=(
            [
                ObservationData(
                    means=np.array([3.0, 5.0]),
                    covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
                    metric_names=["a", "b"],
                )
            ]
        ),
    )
    # pyre-fixme[3]: Return type must be annotated.
    def testDerelativizeTransform(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_predict,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_fit,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_observations_from_data,
    ):
        t = Derelativize(
            search_space=None,
            observations=[],
        )

        # ModelBridge with in-design status quo
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0, 20),
                RangeParameter("y", ParameterType.FLOAT, 0, 20),
            ]
        )
        g = ModelBridge(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment(search_space, "test"),
            data=Data(),
            status_quo_name="1_1",
        )

        # Test with no relative constraints
        objective = Objective(Metric("c"))
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    op=ComparisonOp.LEQ,
                    bound=2,
                    weights=[0.5, 0.5],
                    relative=False,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        self.assertTrue(oc == oc2)

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
        oc = t.transform_optimization_config(oc, g, None)
        self.assertTrue(
            oc.outcome_constraints
            == [
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=4.5, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=4.5,
                    relative=False,
                ),
            ]
        )
        obsf = mock_predict.mock_calls[0][1][1][0]
        obsf2 = ObservationFeatures(parameters={"x": 2.0, "y": 10.0})
        self.assertTrue(obsf == obsf2)

        # Test with relative constraint, out-of-design status quo
        mock_predict.side_effect = RuntimeError()
        g = ModelBridge(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment(search_space, "test"),
            data=Data(),
            status_quo_name="1_2",
        )
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
        oc = t.transform_optimization_config(oc, g, None)
        self.assertTrue(
            oc.outcome_constraints
            == [
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=3.6, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=3.6,
                    relative=False,
                ),
            ]
        )
        self.assertEqual(mock_predict.call_count, 2)

        # Raises error if predict fails with in-design status quo
        g = ModelBridge(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment(search_space, "test"),
            data=Data(),
            status_quo_name="1_1",
        )
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=-10, relative=True
                ),
            ],
        )
        with self.assertRaises(RuntimeError):
            #  `None`.
            oc = t.transform_optimization_config(oc, g, None)

        # Bypasses error if use_raw_sq
        t2 = Derelativize(
            search_space=None,
            observations=[],
            config={"use_raw_status_quo": True},
        )
        oc2 = t2.transform_optimization_config(deepcopy(oc), g, None)

        # Raises error with relative constraint, no status quo
        g = ModelBridge(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment(search_space, "test"),
            data=Data(),
        )
        with self.assertRaises(ValueError):
            #  `None`.
            oc = t.transform_optimization_config(oc, g, None)

        # Raises error with relative constraint, no modelbridge
        with self.assertRaises(ValueError):
            #  `None`.
            oc = t.transform_optimization_config(oc, None, None)

    def testErrors(self) -> None:
        t = Derelativize(
            search_space=None,
            observations=[],
        )
        oc = OptimizationConfig(
            objective=Objective(Metric("c")),
            outcome_constraints=[
                OutcomeConstraint(Metric("a"), ComparisonOp.LEQ, bound=2, relative=True)
            ],
        )
        search_space = SearchSpace(
            parameters=[RangeParameter("x", ParameterType.FLOAT, 0, 20)]
        )
        g = ModelBridge(search_space, None, [])
        with self.assertRaises(ValueError):
            #  `None`.
            t.transform_optimization_config(oc, None, None)
        with self.assertRaises(ValueError):
            #  `None`.
            t.transform_optimization_config(oc, g, None)
