#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from ax.core.metric import Metric
from ax.core.objective import MultiObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.risk_measures import RiskMeasure
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.modelbridge.modelbridge_utils import (
    _array_to_tensor,
    extract_risk_measure,
    extract_robust_digest,
    extract_search_space_digest,
    feasible_hypervolume,
    process_contextual_datasets,
    RISK_MEASURE_NAME_TO_CLASS,
    transform_search_space,
)
from ax.modelbridge.registry import Cont_X_trans, Y_trans
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space, get_search_space
from botorch.acquisition.risk_measures import VaR
from botorch.utils.datasets import ContextualDataset, SupervisedDataset
from pyre_extensions import none_throws


class TestModelBridgeUtils(TestCase):
    def test__array_to_tensor(self) -> None:
        from ax.modelbridge import ModelBridge

        @dataclass
        class MockModelbridge(ModelBridge):
            def _array_to_tensor(self, array: npt.NDArray | list[float]):
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
            robust_digest = none_throws(
                extract_robust_digest(rss, list(rss.parameters))
            )
            self.assertEqual(robust_digest.multiplicative, multiplicative)
            self.assertEqual(robust_digest.environmental_variables, [])
            self.assertIsNone(robust_digest.sample_environmental)
            samples = none_throws(robust_digest.sample_param_perturbations)()
            self.assertEqual(samples.shape, (8, 4))
            constructor = np.ones if multiplicative else np.zeros
            self.assertTrue(np.equal(samples[:, 2:], constructor((8, 2))).all())
            # Exponential distribution is non-negative, so we can check for that.
            self.assertTrue(np.all(samples[:, 1] > 0))
            # Check that it works as expected if param_names is missing some
            # non-distributional parameters.
            robust_digest = none_throws(
                extract_robust_digest(rss, list(rss.parameters)[:-1])
            )
            samples = none_throws(robust_digest.sample_param_perturbations)()
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
        robust_digest = none_throws(extract_robust_digest(rss, list(rss.parameters)))
        self.assertFalse(robust_digest.multiplicative)
        self.assertIsNone(robust_digest.sample_param_perturbations)
        self.assertEqual(robust_digest.environmental_variables, ["x", "y"])
        samples = none_throws(robust_digest.sample_environmental)()
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
        robust_digest = none_throws(extract_robust_digest(rss, list(rss.parameters)))
        self.assertFalse(robust_digest.multiplicative)
        self.assertEqual(
            none_throws(robust_digest.sample_param_perturbations)().shape, (8, 3)
        )
        self.assertEqual(
            none_throws(robust_digest.sample_environmental)().shape, (8, 1)
        )
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

    def test_get_transformed_dimensionality(self) -> None:
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="range",
                    parameter_type=ParameterType.FLOAT,
                    lower=1,
                    upper=8,
                ),
                ChoiceParameter(
                    name="choice",
                    parameter_type=ParameterType.INT,
                    values=[11, 18, 1998],
                    is_ordered=False,
                ),
            ]
        )

        transformed_search_space = transform_search_space(
            search_space=search_space,
            transforms=Cont_X_trans + Y_trans,
            transform_configs={},
        )

        expected = SearchSpace(
            parameters=[
                RangeParameter(
                    name="range", parameter_type=ParameterType.FLOAT, lower=0, upper=1
                ),
                RangeParameter(
                    name="choice_OH_PARAM__0",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                ),
                RangeParameter(
                    name="choice_OH_PARAM__1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                ),
                RangeParameter(
                    name="choice_OH_PARAM__2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                ),
            ],
            parameter_constraints=[],
        )

        self.assertEqual(transformed_search_space, expected)

    def test_process_contextual_datasets(self) -> None:
        num_samples = 5
        num_contexts = 3
        feature_names = [f"x_c{i}" for i in range(num_contexts)]
        parameter_decomposition = {
            f"context_{i}": [f"x_c{i}"] for i in range(num_contexts)
        }
        context_buckets = list(parameter_decomposition.keys())
        context_outcome_list = [f"y:context_{i}" for i in range(num_contexts)]
        metric_decomposition = {f"{c}": [f"y:{c}"] for c in context_buckets}

        X = torch.rand(num_samples, num_contexts)

        dataset_list = [
            SupervisedDataset(
                X=X,
                Y=torch.rand(num_samples, 1),
                Yvar=torch.rand(num_samples, 1),
                feature_names=feature_names,
                outcome_names=["m1_overall"],
            ),
            SupervisedDataset(
                X=X,
                Y=torch.rand(num_samples, 1),
                Yvar=torch.rand(num_samples, 1),
                feature_names=feature_names,
                outcome_names=["m2_overall"],
            ),
        ]
        # process dataset list with overall outcome only
        contextual_datasets = process_contextual_datasets(
            datasets=dataset_list,
            outcomes=["m1_overall", "m2_overall"],
            parameter_decomposition=parameter_decomposition,
        )
        self.assertEqual(len(contextual_datasets), 2)
        for d in contextual_datasets:
            self.assertIsInstance(d, ContextualDataset)
            self.assertEqual(len(d.outcome_names), 1)

        for m in context_outcome_list:
            dataset_list.append(
                SupervisedDataset(
                    X=X,
                    Y=torch.rand(num_samples, 1),
                    Yvar=torch.rand(num_samples, 1),
                    feature_names=feature_names,
                    outcome_names=[m],
                )
            )
        # # process dataset list with context-level outcomes
        contextual_datasets = process_contextual_datasets(
            datasets=dataset_list[2:],
            outcomes=context_outcome_list,
            parameter_decomposition=parameter_decomposition,
            metric_decomposition=metric_decomposition,
        )
        self.assertEqual(len(contextual_datasets), 1)
        self.assertIsInstance(contextual_datasets[0], ContextualDataset)
        self.assertListEqual(contextual_datasets[0].outcome_names, context_outcome_list)

        # process dataset list with overall outcome and context-level outcomes
        contextual_datasets = process_contextual_datasets(
            datasets=dataset_list,
            outcomes=["m1_overall", "m2_overall"] + context_outcome_list,
            parameter_decomposition=parameter_decomposition,
            metric_decomposition=metric_decomposition,
        )
        self.assertEqual(len(contextual_datasets), 3)
        for d in contextual_datasets:
            self.assertIsInstance(d, ContextualDataset)

    def test_extract_search_space_digest(self) -> None:
        # This is also tested as part of broader TorchModelBridge tests.
        # Test log & logit scale parameters.
        for log_scale, logit_scale in [(True, False), (False, True)]:
            ss = SearchSpace(
                parameters=[
                    RangeParameter(
                        name="x1",
                        parameter_type=ParameterType.FLOAT,
                        lower=0.1,
                        upper=0.9,
                        log_scale=log_scale,
                        logit_scale=logit_scale,
                    )
                ]
            )
            with self.assertRaisesRegex(UserInputError, "Log and Logit"):
                extract_search_space_digest(ss, list(ss.parameters))
