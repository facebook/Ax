#!/usr/bin/env python3

from unittest import mock

import numpy as np
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
)
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import ComparisonOp
from ae.lazarus.ae.generator.base import Generator
from ae.lazarus.ae.generator.transforms.derelativize import Derelativize
from ae.lazarus.ae.utils.common.testutils import TestCase


class DerelativizeTransformTest(TestCase):
    def setUp(self):
        m = mock.patch.object(Generator, "__abstractmethods__", frozenset())
        self.addCleanup(m.stop)
        m.start()

    @mock.patch(
        "ae.lazarus.ae.generator.base.observations_from_data",
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
                    condition_name="1_1",
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
                    condition_name="1_2",
                ),
            ]
        ),
    )
    @mock.patch("ae.lazarus.ae.generator.base.Generator._fit", autospec=True)
    @mock.patch(
        "ae.lazarus.ae.generator.base.Generator._predict",
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
    def testDerelativizeTransform(
        self, mock_predict, mock_fit, mock_observations_from_data
    ):
        t = Derelativize(
            search_space=None, observation_features=None, observation_data=None
        )

        # Generator with in-design status quo
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0, 20),
                RangeParameter("y", ParameterType.FLOAT, 0, 20),
            ]
        )
        g = Generator(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment("test", search_space),
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
                )
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
            ]
        )
        obsf = mock_predict.mock_calls[0][1][1][0]
        obsf2 = ObservationFeatures(parameters={"x": 2.0, "y": 10.0})
        self.assertTrue(obsf == obsf2)

        # Test with relative constraint, out-of-design status quo
        mock_predict.side_effect = Exception()
        g = Generator(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment("test", search_space),
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
            ]
        )
        self.assertEqual(mock_predict.call_count, 2)

        # Raises error if predict fails with in-design status quo
        g = Generator(search_space, None, [], status_quo_name="1_1")
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
        with self.assertRaises(Exception):
            oc = t.transform_optimization_config(oc, g, None)

        # Raises error with relative constraint, no status quo
        exp = Experiment("name", search_space)
        g = Generator(search_space, None, [], exp)
        with self.assertRaises(ValueError):
            oc = t.transform_optimization_config(oc, g, None)

        # Raises error with relative constraint, no generator
        with self.assertRaises(ValueError):
            oc = t.transform_optimization_config(oc, None, None)

    def testErrors(self):
        t = Derelativize(
            search_space=None, observation_features=None, observation_data=None
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
        g = Generator(search_space, None, [])
        with self.assertRaises(ValueError):
            t.transform_optimization_config(oc, None, None)
        with self.assertRaises(ValueError):
            t.transform_optimization_config(oc, g, None)
