#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import torch
from ax.adapter.adapter_utils import (
    _get_adapter_training_data,
    arm_to_np_array,
    can_map_to_binary,
    extract_objective_weight_matrix,
    extract_search_space_digest,
    feasible_hypervolume,
    is_unordered_choice,
    process_contextual_datasets,
    transform_search_space,
    validate_and_apply_final_transform,
)
from ax.adapter.registry import Cont_X_trans, Y_trans
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.choice_encode import ChoiceToNumericChoice
from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_observations,
    get_hierarchical_search_space,
    get_search_space_for_range_values,
)
from botorch.utils.datasets import ContextualDataset, SupervisedDataset


class TestAdapterUtils(TestCase):
    def test_feasible_hypervolume(self) -> None:
        ma = Metric(name="a", lower_is_better=False)
        mb = Metric(name="b", lower_is_better=True)
        mc = Metric(name="c", lower_is_better=False)
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[Objective(metric=ma), Objective(metric=mb)]
            ),
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
                    name="choice_OH_PARAM_0",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                ),
                RangeParameter(
                    name="choice_OH_PARAM_1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0,
                    upper=1,
                ),
                RangeParameter(
                    name="choice_OH_PARAM_2",
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
        # This is also tested as part of broader TorchAdapter tests.
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

        hss = get_hierarchical_search_space()
        # We need to at least convert non-numerical values to numerical values.
        # Otherwise, the search space digest is not sensible.
        t_instance = ChoiceToNumericChoice(search_space=hss.clone())
        hss = t_instance.transform_search_space(hss)

        ssd = extract_search_space_digest(hss, list(hss.parameters))
        self.assertEqual(ssd.hierarchical_dependencies, {0: {0: [1, 2], 1: [3]}})

    def test_get_in_desing_adapter_training_data(self) -> None:
        def get_adapter(min: float, max: float) -> TorchAdapter:
            return TorchAdapter(
                generator=BoTorchGenerator(),
                experiment=get_experiment_with_observations(
                    observations=[[0.5, 1.0], [1.5, 2.0]],
                    parameterizations=[
                        {"x1": 20.0, "x2": 1.0},
                        {"x1": 2.0, "x2": 10.0},
                    ],
                    search_space=get_search_space_for_range_values(
                        min=min, max=max, parameter_names=["x1", "x2"]
                    ),
                ),
                expand_model_space=False,  # To control in / out of design.
                fit_on_init=False,
            )

        # Test . _get_adapter_training_data w/ and w/o in_design_only return the
        # same observations when all are in-design.
        adapter = get_adapter(min=0.0, max=25.0)
        obs_feats, obs_data, arm_names = _get_adapter_training_data(adapter=adapter)
        in_design_obs_feats, in_design_obs_data, in_design_arm_names = (
            _get_adapter_training_data(adapter=adapter, in_design_only=True)
        )
        self.assertEqual(obs_feats, in_design_obs_feats)
        self.assertEqual(obs_data, in_design_obs_data)
        self.assertEqual(arm_names, in_design_arm_names)

        # Test results are different when some data is out-of-design.
        adapter = get_adapter(min=0.0, max=10.0)
        obs_feats, obs_data, arm_names = _get_adapter_training_data(adapter=adapter)
        in_design_obs_feats, in_design_obs_data, in_design_arm_names = (
            _get_adapter_training_data(adapter=adapter, in_design_only=True)
        )
        self.assertEqual(len(obs_feats), 2)
        self.assertEqual(len(obs_data), 2)
        self.assertEqual(len(arm_names), 2)

        self.assertEqual(len(in_design_obs_feats), 1)
        self.assertEqual(len(in_design_obs_data), 1)
        self.assertEqual(len(in_design_arm_names), 1)

        self.assertEqual(obs_feats[1], in_design_obs_feats[0])
        self.assertEqual(obs_data[1], in_design_obs_data[0])
        self.assertEqual(arm_names[1], in_design_arm_names[0])

        # Test nothing is returned when all are out-of-design.
        adapter = get_adapter(min=0.0, max=1.0)
        in_design_obs_feats, in_design_obs_data, in_design_arm_names = (
            _get_adapter_training_data(adapter=adapter, in_design_only=True)
        )
        self.assertEqual(len(in_design_obs_feats), 0)
        self.assertEqual(len(in_design_obs_data), 0)
        self.assertEqual(len(in_design_arm_names), 0)

    def test_arm_to_np_array(self) -> None:
        # Test extracting target point from arm with valid parameters

        # Setup: create arm with target parameter values
        target_arm = Arm(parameters={"x1": 0.5, "x2": 1.5, "x3": 2.5})
        parameters = ["x1", "x2", "x3"]

        # Execute: extract target point
        actual = arm_to_np_array(arm=target_arm, parameters=parameters)

        # Assert: confirm extracted values match expected order
        expected = np.array([0.5, 1.5, 2.5])
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual, expected)

    def test_extract_arm_to_np_array_different_parameter_order(self) -> None:
        # Test extracting target point with different parameter ordering

        # Setup: create arm and specify parameters in different order
        target_arm = Arm(parameters={"x1": 0.5, "x2": 1.5, "x3": 2.5})
        parameters = ["x3", "x1", "x2"]

        # Execute: extract target point
        actual = arm_to_np_array(arm=target_arm, parameters=parameters)

        # Assert: confirm values are extracted in specified parameter order
        expected = np.array([2.5, 0.5, 1.5])
        self.assertIsNotNone(actual)
        np.testing.assert_array_equal(actual, expected)

    def test_arm_to_np_array_none(self) -> None:
        # Test that None is returned when target_arm is None
        parameters = ["x1", "x2"]

        # Execute: extract target point from None arm
        actual = arm_to_np_array(arm=None, parameters=parameters)

        # Assert: confirm None is returned
        self.assertIsNone(actual)

    def test_validate_and_apply_final_transform_with_target_point(self) -> None:
        # Test validate_and_apply_final_transform includes target_point in output

        # Setup: create input data with target point
        objective_weights = np.array([1.0, -1.0])
        outcome_constraints = (np.array([[1.0, 0.0]]), np.array([2.0]))
        linear_constraints = (np.array([[1.0, 1.0]]), np.array([1.0]))
        pending_observations = [np.array([[0.5, 0.5]])]
        objective_thresholds = np.array([1.0, 2.0])
        target_point = np.array([0.2, 0.8])

        # Execute: apply final transform with target point
        (
            _,
            _,
            _,
            _,
            _,
            target_p,
        ) = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            objective_thresholds=objective_thresholds,
            pruning_target_point=target_point,
            final_transform=torch.tensor,
        )

        # Assert: confirm target point is correctly transformed
        self.assertIsInstance(target_p, torch.Tensor)
        torch.testing.assert_close(
            target_p, torch.tensor([0.2, 0.8], dtype=torch.double)
        )

    def test_validate_and_apply_final_transform_none_target_point(self) -> None:
        # Test validate_and_apply_final_transform with None target_point

        # Setup: create input data without target point
        objective_weights = np.array([1.0, -1.0])
        outcome_constraints = (np.array([[1.0, 0.0]]), np.array([2.0]))
        linear_constraints = (np.array([[1.0, 1.0]]), np.array([1.0]))
        pending_observations = [np.array([[0.5, 0.5]])]
        objective_thresholds = np.array([1.0, 2.0])
        target_point = None

        # Execute: apply final transform without target point
        (
            _,
            _,
            _,
            _,
            _,
            target_p,
        ) = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            objective_thresholds=objective_thresholds,
            pruning_target_point=target_point,
            final_transform=torch.tensor,
        )

        # Assert: confirm target point remains None
        self.assertIsNone(target_p)

    def test_is_unordered_choice(self) -> None:
        # Test cases where is_unordered_choice should return True
        # (with min_choices=3, max_choices=5)
        for p in [
            # Unordered choice (INT), with 3 choices
            ChoiceParameter("p", ParameterType.INT, values=[0, 1, 2], is_ordered=False),
            # Unordered choice (STRING), with 5 choices
            ChoiceParameter(
                "p",
                ParameterType.STRING,
                values=["0", "1", "2", "4", "5"],
                is_ordered=False,
            ),
            # Unordered choice (STRING), with 4 choices
            ChoiceParameter(
                "p", ParameterType.STRING, values=["a", "b", "c", "d"], is_ordered=False
            ),
        ]:
            with self.subTest(p=p):
                self.assertTrue(is_unordered_choice(p, min_choices=3, max_choices=5))

        # Test cases where is_unordered_choice should return False
        # (with min_choices=3, max_choices=5)
        for p in [
            # Too few choices
            ChoiceParameter("p", ParameterType.INT, values=[0, 1], is_ordered=False),
            # Ordered choice (INT)
            ChoiceParameter(
                "p", ParameterType.INT, values=[0, 1, 2, 4], is_ordered=True
            ),
            # Range parameter (non-choice)
            RangeParameter("p", parameter_type=ParameterType.INT, lower=0, upper=3),
            # Ordered choice (STRING)
            ChoiceParameter(
                "p", ParameterType.STRING, values=["0", "1", "2"], is_ordered=True
            ),
        ]:
            with self.subTest(p=p):
                self.assertFalse(is_unordered_choice(p, min_choices=3, max_choices=5))

        # Test error cases
        p = ChoiceParameter("p", ParameterType.INT, values=[0, 1, 2], is_ordered=False)
        with self.assertRaisesRegex(
            UserInputError, "`min_choices` must be a non-negative integer."
        ):
            is_unordered_choice(p, min_choices=-3)
        with self.assertRaisesRegex(
            UserInputError, "`max_choices` must be a non-negative integer."
        ):
            is_unordered_choice(p, max_choices=-1)
        with self.assertRaisesRegex(
            UserInputError, "`min_choices` cannot be larger than `max_choices`."
        ):
            is_unordered_choice(p, min_choices=3, max_choices=2)

    def test_can_map_to_binary(self) -> None:
        # Test cases where can_map_to_binary should return True
        for p in [
            # Int range with exactly 2 values
            RangeParameter(
                name="p", parameter_type=ParameterType.INT, lower=0, upper=1
            ),
            RangeParameter(
                name="p", parameter_type=ParameterType.INT, lower=3, upper=4
            ),
            # Choice with exactly 2 values
            ChoiceParameter(
                name="p",
                parameter_type=ParameterType.INT,
                values=[0, 1],
                is_ordered=False,
            ),
            ChoiceParameter(
                name="p",
                parameter_type=ParameterType.STRING,
                values=["a", "b"],
                is_ordered=False,
            ),
        ]:
            with self.subTest(p=p):
                self.assertTrue(can_map_to_binary(p))

        # Test cases where can_map_to_binary should return False
        for p in [
            # Float range (continuous, not binary)
            RangeParameter(
                name="p", parameter_type=ParameterType.FLOAT, lower=0, upper=1
            ),
            # Int range with more than 2 values
            RangeParameter(
                name="p", parameter_type=ParameterType.INT, lower=0, upper=3
            ),
            # Choice with more than 2 values
            ChoiceParameter(
                name="p",
                parameter_type=ParameterType.INT,
                values=[0, 1, 2],
                is_ordered=False,
            ),
            ChoiceParameter(
                name="p",
                parameter_type=ParameterType.STRING,
                values=["a", "b", "c"],
                is_ordered=False,
            ),
        ]:
            with self.subTest(p=p):
                self.assertFalse(can_map_to_binary(p))

    def test_extract_objective_weight_matrix(self) -> None:
        m1, m2, m3 = Metric(name="m1"), Metric(name="m2"), Metric(name="m3")
        outcomes = ["m1", "m2", "m3"]

        # Single Objective: one row, nonzero only in matching column.
        obj = Objective(metric=m1, minimize=False)
        result = extract_objective_weight_matrix(obj, outcomes)
        np.testing.assert_array_equal(result, [[1.0, 0.0, 0.0]])

        # Minimization flips the sign.
        obj_min = Objective(metric=m2, minimize=True)
        result = extract_objective_weight_matrix(obj_min, outcomes)
        np.testing.assert_array_equal(result, [[0.0, -1.0, 0.0]])

        # ScalarizedObjective: single row with multiple nonzero entries.
        scal = ScalarizedObjective(metrics=[m1, m3], weights=[0.3, 0.7], minimize=False)
        result = extract_objective_weight_matrix(scal, outcomes)
        np.testing.assert_array_almost_equal(result, [[0.3, 0.0, 0.7]])

        # MultiObjective: one row per sub-objective.
        multi = MultiObjective(
            objectives=[
                Objective(metric=m1, minimize=False),
                Objective(metric=m3, minimize=True),
            ]
        )
        result = extract_objective_weight_matrix(multi, outcomes)
        np.testing.assert_array_equal(result, [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
