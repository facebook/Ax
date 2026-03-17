#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import pandas as pd
import torch
from ax.adapter.adapter_utils import (
    _get_adapter_training_data,
    _get_fresh_pairwise_trial_indices,
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
from ax.core.data import Data
from ax.core.derived_metric import DerivedMetric
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.constants import Keys
from ax.utils.common.hash_utils import compute_lilo_input_hash
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
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
                    metric=mc,
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
        experiment = Experiment(
            search_space=SearchSpace(parameters=[]),
            optimization_config=optimization_config,
            tracking_metrics=[mc],
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
            experiment=experiment,
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
        target_arm = Arm(parameters={"x1": 0.5, "x2": 1.5, "x3": 2.5})
        cases = [
            # Values extracted in natural parameter order
            (["x1", "x2", "x3"], np.array([0.5, 1.5, 2.5])),
            # Values extracted in a different parameter order
            (["x3", "x1", "x2"], np.array([2.5, 0.5, 1.5])),
        ]
        for parameters, expected in cases:
            with self.subTest(parameters=parameters):
                actual = arm_to_np_array(arm=target_arm, parameters=parameters)
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
        experiment = get_branin_experiment()
        experiment.add_metric(m1)
        experiment.add_metric(m2)
        experiment.add_metric(m3)

        # Single Objective: one row, nonzero only in matching column.
        obj = Objective(metric=m1, minimize=False)
        result = extract_objective_weight_matrix(obj, outcomes, experiment)
        np.testing.assert_array_equal(result, [[1.0, 0.0, 0.0]])

        # Minimization flips the sign.
        obj_min = Objective(metric=m2, minimize=True)
        result = extract_objective_weight_matrix(obj_min, outcomes, experiment)
        np.testing.assert_array_equal(result, [[0.0, -1.0, 0.0]])

        # ScalarizedObjective: single row with multiple nonzero entries.
        scal = ScalarizedObjective(metrics=[m1, m3], weights=[0.3, 0.7], minimize=False)
        result = extract_objective_weight_matrix(scal, outcomes, experiment)
        np.testing.assert_array_almost_equal(result, [[0.3, 0.0, 0.7]])

        # MultiObjective: one row per sub-objective.
        multi = MultiObjective(
            objectives=[
                Objective(metric=m1, minimize=False),
                Objective(metric=m3, minimize=True),
            ]
        )
        result = extract_objective_weight_matrix(multi, outcomes, experiment)
        np.testing.assert_array_equal(result, [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    def test_get_fresh_pairwise_trial_indices(self) -> None:
        """Verify _get_fresh_pairwise_trial_indices hash-based filtering."""
        search_space = get_search_space_for_range_values()
        exp = Experiment(name="test", search_space=search_space)

        # Register a DerivedMetric with pairwise name so the function can
        # look up input_metric_names.
        pairwise_metric = DerivedMetric(
            name=Keys.PAIRWISE_PREFERENCE_QUERY.value,
            input_metric_names=["latency"],
        )
        exp.add_tracking_metric(pairwise_metric)

        # Helper to create trial data.
        def _attach(
            trial_index: int, arms: dict[str, float], exp: Experiment = exp
        ) -> None:
            rows = [
                {
                    "trial_index": trial_index,
                    "arm_name": name,
                    "metric_name": "latency",
                    "metric_signature": "latency",
                    "mean": val,
                    "sem": 0.1,
                }
                for name, val in arms.items()
            ]
            exp.attach_data(Data(df=pd.DataFrame(rows)))

        # Create two trials with data.
        for i in range(2):
            trial = exp.new_batch_trial()
            trial.add_arm(Arm(name=f"{i}_0", parameters={"x": float(i)}))
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            _attach(i, {f"{i}_0": float(i + 1)})

        with self.subTest("no_hashes_returns_none"):
            # No trials have LILO_INPUT_HASH -- not a LILO experiment.
            result = _get_fresh_pairwise_trial_indices(exp)
            self.assertIsNone(result)

        # Stamp trial 0 with the current hash.
        current_hash = compute_lilo_input_hash(exp, ["latency"])
        exp.trials[0]._properties[Keys.LILO_INPUT_HASH] = current_hash

        with self.subTest("fresh_hash_included"):
            result = _get_fresh_pairwise_trial_indices(exp)
            assert result is not None
            self.assertIn(0, result)
            # Trial 1 has no hash -- always included.
            self.assertIn(1, result)

        # Stamp trial 1 with a stale hash.
        exp.trials[1]._properties[Keys.LILO_INPUT_HASH] = "stale_hash_value"

        with self.subTest("stale_hash_excluded"):
            result = _get_fresh_pairwise_trial_indices(exp)
            assert result is not None
            self.assertIn(0, result)
            self.assertNotIn(1, result)

        with self.subTest("all_stale"):
            # Make both hashes stale by adding new data.
            trial2 = exp.new_batch_trial()
            trial2.add_arm(Arm(name="2_0", parameters={"x": 10.0}))
            trial2.mark_running(no_runner_required=True)
            trial2.mark_completed()
            _attach(2, {"2_0": 999.0})
            # Now both trial 0 and trial 1 have stale hashes.
            result = _get_fresh_pairwise_trial_indices(exp)
            assert result is not None
            # Trial 0 and 1 are stale, trial 2 has no hash -- included.
            self.assertNotIn(0, result)
            self.assertNotIn(1, result)
            self.assertIn(2, result)
