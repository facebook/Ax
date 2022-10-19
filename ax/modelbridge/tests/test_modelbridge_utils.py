#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Union

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.risk_measures import RiskMeasure
from ax.core.search_space import RobustSearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.modelbridge.modelbridge_utils import (
    _array_to_tensor,
    extract_risk_measure,
    extract_robust_digest,
    feasible_hypervolume,
    RISK_MEASURE_NAME_TO_CLASS,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import get_robust_search_space, get_search_space
from botorch.acquisition.risk_measures import VaR


class TestModelBridgeUtils(TestCase):
    def test__array_to_tensor(self) -> None:
        from ax.modelbridge import ModelBridge

        @dataclass
        class MockModelbridge(ModelBridge):
            def _array_to_tensor(self, array: Union[np.ndarray, List[float]]):
                return _array_to_tensor(array=array)

        mock_modelbridge = MockModelbridge()
        arr = [0.0]
        res = _array_to_tensor(array=arr)
        self.assertEqual(len(res.size()), 1)
        self.assertEqual(res.size()[0], 1)

        res = _array_to_tensor(array=arr, modelbridge=mock_modelbridge)
        self.assertEqual(len(res.size()), 1)
        self.assertEqual(res.size()[0], 1)

    def test_extract_risk_measure(self) -> None:
        rm = RiskMeasure(
            risk_measure="VaR",
            options={"alpha": 0.8, "n_w": 5},
        )
        rm_module = extract_risk_measure(risk_measure=rm)
        self.assertIsInstance(rm_module, VaR)
        self.assertEqual(rm_module.alpha, 0.8)
        self.assertEqual(rm_module.n_w, 5)

        # Test unknown risk measure.
        with self.assertRaisesRegex(UserInputError, "constructing"):
            extract_risk_measure(
                risk_measure=RiskMeasure(
                    risk_measure="VVar",
                    options={},
                )
            )
        # Test invalid options.
        with self.assertRaisesRegex(UserInputError, "constructing"):
            extract_risk_measure(
                risk_measure=RiskMeasure(
                    risk_measure="VaR",
                    options={"alpha": 5, "n_w": 5},
                )
            )

        # Test using user-defined risk measures.

        class CustomRM(VaR):
            pass

        RISK_MEASURE_NAME_TO_CLASS["custom"] = CustomRM

        rm = RiskMeasure(
            risk_measure="custom",
            options={"alpha": 0.8, "n_w": 5},
        )
        self.assertEqual(rm.risk_measure, "custom")
        self.assertIsInstance(extract_risk_measure(risk_measure=rm), CustomRM)

    def test_extract_robust_digest(self) -> None:
        # Test with non-robust search space.
        ss = get_search_space()
        self.assertIsNone(extract_robust_digest(ss, list(ss.parameters)))
        # Test with non-environmental search space.
        for multiplicative in (True, False):
            rss = get_robust_search_space(num_samples=8)
            if multiplicative:
                for p in rss.parameter_distributions:
                    p.multiplicative = True
                rss.multiplicative = True
            robust_digest = not_none(extract_robust_digest(rss, list(rss.parameters)))
            self.assertEqual(robust_digest.multiplicative, multiplicative)
            self.assertEqual(robust_digest.environmental_variables, [])
            self.assertIsNone(robust_digest.sample_environmental)
            samples = not_none(robust_digest.sample_param_perturbations)()
            self.assertEqual(samples.shape, (8, 4))
            constructor = np.ones if multiplicative else np.zeros
            self.assertTrue(np.equal(samples[:, 2:], constructor((8, 2))).all())
            # Exponential distribution is non-negative, so we can check for that.
            self.assertTrue(np.all(samples[:, 1] > 0))
            # Check that it works as expected if param_names is missing some
            # non-distributional parameters.
            robust_digest = not_none(
                extract_robust_digest(rss, list(rss.parameters)[:-1])
            )
            samples = not_none(robust_digest.sample_param_perturbations)()
            self.assertEqual(samples.shape, (8, 3))
            self.assertTrue(np.equal(samples[:, 2:], constructor((8, 1))).all())
            self.assertTrue(np.all(samples[:, 1] > 0))
            # Check that it errors if we're missing distributional parameters.
            with self.assertRaisesRegex(RuntimeError, "All distributional"):
                extract_robust_digest(rss, list(rss.parameters)[1:])
        # Test with environmental search space.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=8,
            environmental_variables=all_params[:2],
        )
        robust_digest = not_none(extract_robust_digest(rss, list(rss.parameters)))
        self.assertFalse(robust_digest.multiplicative)
        self.assertIsNone(robust_digest.sample_param_perturbations)
        self.assertEqual(robust_digest.environmental_variables, ["x", "y"])
        samples = not_none(robust_digest.sample_environmental)()
        self.assertEqual(samples.shape, (8, 2))
        # Both are continuous distributions, should be non-zero.
        self.assertTrue(np.all(samples != 0))
        # Check for error if environmental variables are not at the end.
        with self.assertRaisesRegex(RuntimeError, "last entries"):
            extract_robust_digest(rss, list(rss.parameters)[::-1])
        # Test with mixed search space.
        rss = RobustSearchSpace(
            parameters=all_params[1:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=8,
            environmental_variables=all_params[:1],
        )
        robust_digest = not_none(extract_robust_digest(rss, list(rss.parameters)))
        self.assertFalse(robust_digest.multiplicative)
        self.assertEqual(
            not_none(robust_digest.sample_param_perturbations)().shape, (8, 3)
        )
        self.assertEqual(not_none(robust_digest.sample_environmental)().shape, (8, 1))
        self.assertEqual(robust_digest.environmental_variables, ["x"])

    def test_feasible_hypervolume(self) -> None:
        ma = Metric(name="a", lower_is_better=False)
        mb = Metric(name="b", lower_is_better=True)
        mc = Metric(name="c", lower_is_better=False)
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(metrics=[ma, mb]),
            outcome_constraints=[
                OutcomeConstraint(
                    mc,
                    op=ComparisonOp.GEQ,
                    bound=0,
                    relative=False,
                )
            ],
            objective_thresholds=[
                ObjectiveThreshold(
                    ma,
                    bound=1.0,
                ),
                ObjectiveThreshold(
                    mb,
                    bound=1.0,
                ),
            ],
        )
        feas_hv = feasible_hypervolume(
            optimization_config,
            values={
                "a": np.array(
                    [
                        1.0,
                        3.0,
                        2.0,
                        2.0,
                    ]
                ),
                "b": np.array(
                    [
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                    ]
                ),
                "c": np.array(
                    [
                        0.0,
                        -0.0,
                        1.0,
                        -2.0,
                    ]
                ),
            },
        )
        self.assertEqual(list(feas_hv), [0.0, 0.0, 1.0, 1.0])
