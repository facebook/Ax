#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import ExitStack
from copy import deepcopy
from unittest import mock
from unittest.mock import Mock, patch

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
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.utils.common.testutils import TestCase


class DerelativizeTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        m = mock.patch.object(ModelBridge, "__abstractmethods__", frozenset())
        self.addCleanup(m.stop)
        m.start()

    def test_DerelativizeTransform(self) -> None:
        for negative_metrics in [False, True]:
            sq_sign = -1.0 if negative_metrics else 1.0
            observed_means = sq_sign * np.array([1.0, 2.0, 6.0])
            sq_b_observed = np.mean(observed_means[1:])
            predicted_means = sq_sign * np.array([3.0, 5.0])
            sq_b_predicted = predicted_means[-1]
            predict_return_value = [
                ObservationData(
                    means=predicted_means,
                    covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
                    metric_names=["a", "b"],
                )
            ]
            observations_from_data_return_value = [
                Observation(
                    features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}),
                    data=ObservationData(
                        means=observed_means,
                        covariance=np.array(
                            [
                                [1.0, 2.0, 0.0],
                                [3.0, 4.0, 0.0],
                                [0.0, 0.0, 4.0],
                            ]
                        ),
                        metric_names=["a", "b", "b"],
                    ),
                    arm_name="1_1",
                ),
                Observation(
                    features=ObservationFeatures(parameters={"x": None, "y": None}),
                    data=ObservationData(
                        means=observed_means,
                        covariance=np.array(
                            [
                                [1.0, 2.0, 0.0],
                                [3.0, 4.0, 0.0],
                                [0.0, 0.0, 4.0],
                            ]
                        ),
                        metric_names=["a", "b", "b"],
                    ),
                    arm_name="1_2",
                ),
            ]
            with ExitStack() as es:
                mock_predict = es.enter_context(
                    mock.patch(
                        "ax.modelbridge.base.ModelBridge._predict",
                        autospec=True,
                        return_value=predict_return_value,
                    )
                )
                mock_fit = es.enter_context(
                    mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
                )
                mock_observations_from_data = es.enter_context(
                    mock.patch(
                        "ax.modelbridge.base.observations_from_data",
                        autospec=True,
                        return_value=observations_from_data_return_value,
                    )
                )
                self._test_DerelativizeTransform(
                    mock_predict=mock_predict,
                    mock_fit=mock_fit,
                    mock_observations_from_data=mock_observations_from_data,
                    sq_b_observed=sq_b_observed,
                    sq_b_predicted=sq_b_predicted,
                )

    def _test_DerelativizeTransform(
        self,
        mock_predict: Mock,
        mock_fit: Mock,
        mock_observations_from_data: Mock,
        sq_b_observed: float,
        sq_b_predicted: float,
    ) -> None:
        t = Derelativize(search_space=None, observations=[])

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
        objective = Objective(Metric("c"), minimize=True)
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
        relative_bound = -10
        oc = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=relative_bound, relative=True
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        sq_val = sq_b_predicted
        absolute_bound = (1 + np.sign(sq_val) * relative_bound / 100.0) * sq_val
        self.assertTrue(
            oc2.outcome_constraints
            == [
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=absolute_bound, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
            ]
        )
        obsf = mock_predict.mock_calls[0][1][1][0]
        obsf2 = ObservationFeatures(parameters={"x": 2.0, "y": 10.0})
        self.assertTrue(obsf == obsf2)
        self.assertEqual(mock_predict.call_count, 1)

        # The model should not be used when `use_raw_status_quo` is True
        t2 = deepcopy(t)
        t2.config["use_raw_status_quo"] = True
        t2.transform_optimization_config(deepcopy(oc), g, None)
        self.assertEqual(mock_predict.call_count, 1)

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
                    Metric("b"), ComparisonOp.LEQ, bound=relative_bound, relative=True
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=relative_bound,
                    relative=True,
                ),
            ],
        )
        oc2 = t.transform_optimization_config(oc, g, None)
        sq_val = sq_b_observed
        absolute_bound = (1 + np.sign(sq_val) * relative_bound / 100.0) * sq_val
        self.assertTrue(
            oc2.outcome_constraints
            == [
                OutcomeConstraint(
                    Metric("a"), ComparisonOp.LEQ, bound=2, relative=False
                ),
                OutcomeConstraint(
                    Metric("b"), ComparisonOp.LEQ, bound=absolute_bound, relative=False
                ),
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=absolute_bound,
                    relative=False,
                ),
            ]
        )
        self.assertEqual(mock_predict.call_count, 1)

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
            t.transform_optimization_config(oc, g, None)

        # Bypasses error if use_raw_sq
        t2 = Derelativize(
            search_space=None,
            observations=[],
            config={"use_raw_status_quo": True},
        )
        t2.transform_optimization_config(deepcopy(oc), g, None)

        # But not if sq arm is not available.
        with patch(
            f"{Derelativize.__module__}.unwrap_observation_data", return_value=({}, {})
        ), self.assertRaisesRegex(
            DataRequiredError, "Status-quo metric value not yet available for metric "
        ):
            t2.transform_optimization_config(deepcopy(oc), g, None)

        # Same for scalarized constraint only.
        oc_scalarized_only = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                ScalarizedOutcomeConstraint(
                    metrics=[Metric("a"), Metric("b")],
                    weights=[0.0, 1.0],
                    op=ComparisonOp.LEQ,
                    bound=-10,
                    relative=True,
                ),
            ],
        )
        with patch(
            f"{Derelativize.__module__}.unwrap_observation_data", return_value=({}, {})
        ), self.assertRaisesRegex(
            DataRequiredError,
            "Status-quo metric value not yet available for metric\\(s\\) ",
        ):
            t2.transform_optimization_config(deepcopy(oc_scalarized_only), g, None)

        # Raises error with relative constraint, no status quo.
        g = ModelBridge(
            search_space=search_space,
            model=None,
            transforms=[],
            experiment=Experiment(search_space, "test"),
            data=Data(),
        )
        with self.assertRaises(DataRequiredError):
            t.transform_optimization_config(deepcopy(oc), g, None)

        # Raises error with relative constraint, no modelbridge.
        with self.assertRaises(ValueError):
            t.transform_optimization_config(deepcopy(oc), None, None)

    def test_Errors(self) -> None:
        t = Derelativize(
            search_space=None,
            observations=[],
        )
        oc = OptimizationConfig(
            objective=Objective(Metric("c"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(Metric("a"), ComparisonOp.LEQ, bound=2, relative=True)
            ],
        )
        search_space = SearchSpace(
            parameters=[RangeParameter("x", ParameterType.FLOAT, 0, 20)]
        )
        g = ModelBridge(search_space, None, [])
        with self.assertRaises(ValueError):
            t.transform_optimization_config(oc, None, None)
        with self.assertRaises(DataRequiredError):
            t.transform_optimization_config(oc, g, None)
