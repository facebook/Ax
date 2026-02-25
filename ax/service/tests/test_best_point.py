# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from unittest import mock
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from ax.adapter.registry import Generators
from ax.adapter.torch import TorchAdapter
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.service.utils.best_point import (
    get_tensor_converter_adapter,
    get_trace,
    infer_reference_point_from_experiment,
    logger,
)
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_batch_trial,
    get_experiment_with_observations,
    get_experiment_with_trial,
)
from pyre_extensions import assert_is_instance, none_throws


class TestBestPointMixin(TestCase):
    def test_get_trace(self) -> None:
        # Single objective, minimize.
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]], minimize=True
        )
        self.assertEqual(get_trace(exp), [11, 10, 9, 9, 5])

        # Same experiment with maximize via new optimization config.
        opt_conf = none_throws(exp.optimization_config).clone()
        opt_conf.objective.minimize = False
        self.assertEqual(get_trace(exp, opt_conf), [11, 11, 11, 15, 15])

        with self.subTest("Single objective with constraints"):
            # The second metric is the constraint and needs to be >= 0
            exp = get_experiment_with_observations(
                observations=[[11, -1], [10, 1], [9, 1], [15, -1], [11, 1]],
                minimize=False,
                constrained=True,
            )
            self.assertEqual(get_trace(exp), [float("-inf"), 10, 10, 10, 11])

            exp = get_experiment_with_observations(
                observations=[[11, -1], [10, 1], [9, 1], [15, -1], [11, 1]],
                minimize=True,
                constrained=True,
            )
            self.assertEqual(get_trace(exp), [float("inf"), 10, 9, 9, 9])

        # Scalarized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [2, 2], [3, 3]],
            scalarized=True,
        )
        self.assertEqual(get_trace(exp), [2, 4, 6])

        # Multi objective.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 100], [1, 2], [3, 3], [2, 4], [2, 1]],
        )
        self.assertEqual(get_trace(exp), [1, 1, 2, 9, 11, 11])

        # W/o ObjectiveThresholds (infering ObjectiveThresholds from nadir point)
        assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        ).objective_thresholds = []
        self.assertEqual(get_trace(exp), [0.0, 0.0, 2.0, 8.0, 11.0, 11.0])

        # Multi-objective w/ constraints.
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 1, 1]],
            constrained=True,
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ relative constraints & status quo.
        exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")
        exp.optimization_config.outcome_constraints[0].bound = 1.0
        exp.optimization_config.outcome_constraints[0].relative = True
        # Fails if there's no data for status quo.
        with self.assertRaisesRegex(DataRequiredError, "relative constraint"):
            get_trace(exp)
        # Add data for status quo.
        trial = Trial(experiment=exp).add_arm(arm=exp.status_quo).run().mark_completed()
        df_dict = [
            {
                "trial_index": trial.index,
                "metric_name": m,
                "arm_name": "status_quo",
                "mean": 0.0,
                "sem": 0.0,
                "metric_signature": m,
            }
            for m in ["m1", "m2", "m3"]
        ]
        status_quo_data = Data(df=pd.DataFrame.from_records(df_dict))
        exp.attach_data(data=status_quo_data)
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ first objective being minimized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [-1, 2], [3, 3], [-2, 4], [2, 1]], minimize=True
        )
        self.assertEqual(get_trace(exp), [0, 2, 2, 8, 8])

        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(get_trace(exp), [])

        # test batch trial
        exp = get_experiment_with_batch_trial(with_status_quo=False)
        trial = exp.trials[0]
        exp.optimization_config.outcome_constraints[0].relative = False
        trial.mark_running(no_runner_required=True).mark_completed()
        df_dict = []
        for i, arm in enumerate(trial.arms):
            df_dict.extend(
                [
                    {
                        "trial_index": 0,
                        "metric_name": m,
                        "arm_name": arm.name,
                        "mean": float(i),
                        "sem": 0.0,
                        "metric_signature": m,
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict)))
        self.assertEqual(get_trace(exp), [2.0])
        # test that there is performance metric in the trace for each
        # completed/early-stopped trial
        trial1 = assert_is_instance(trial, BatchTrial).clone_to(include_sq=False)
        trial1.mark_abandoned(unsafe=True)
        trial2 = exp.new_batch_trial(Generators.SOBOL(experiment=exp).gen(n=3))
        trial2.mark_running(no_runner_required=True).mark_completed()
        df_dict2 = []
        for i, arm in enumerate(trial2.arms):
            df_dict2.extend(
                [
                    {
                        "trial_index": 2,
                        "metric_name": m,
                        "arm_name": arm.name,
                        "mean": 10 * float(i),
                        "sem": 0.0,
                        "metric_signature": m,
                    }
                    for m in exp.metrics.keys()
                ]
            )
        exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict2)))
        self.assertEqual(get_trace(exp), [2.0, 20.0])

    def test_get_trace_with_include_status_quo(self) -> None:
        with self.subTest("Multi-objective: status quo dominates in some trials"):
            # Create experiment with multi-objective optimization where status quo
            # is deliberately the best arm in some trials to test include_status_quo.
            exp = get_experiment_with_observations(
                observations=[[1, 1], [-1, 2], [3, 3]], minimize=True
            )

            # Set up status quo
            exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")

            # Create batch trial where status quo DOMINATES other arms
            batch_trial1 = exp.new_batch_trial(should_add_status_quo_arm=True)
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.1, "y": 0.1}, name="poor_arm_1")
            )
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.2, "y": 0.2}, name="poor_arm_2")
            )

            # Data: Status quo has excellent values, other arms are poor
            df_dict1 = [
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "poor_arm_1",
                    "mean": 10.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "poor_arm_1",
                    "mean": -5.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "poor_arm_2",
                    "mean": 12.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "poor_arm_2",
                    "mean": -3.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
                # Status quo: excellent in both objectives
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m1",
                    "arm_name": "status_quo",
                    "mean": -10.0,
                    "sem": 0.0,
                    "metric_signature": "m1",
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": "m2",
                    "arm_name": "status_quo",
                    "mean": 10.0,
                    "sem": 0.0,
                    "metric_signature": "m2",
                },
            ]
            exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict1)))
            batch_trial1.mark_running(no_runner_required=True).mark_completed()

            # Get trace without status quo
            trace_without_sq = get_trace(exp, include_status_quo=False)

            # Get trace with status quo
            trace_with_sq = get_trace(exp, include_status_quo=True)

            # Both should have 4 trace values (3 initial + 1 batch trial)
            self.assertEqual(len(trace_without_sq), 4)
            self.assertEqual(len(trace_with_sq), 4)

            # The last value MUST differ because status quo dominates
            # Without status quo, only poor arms contribute (low hypervolume)
            # With status quo, excellent values contribute (high hypervolume)
            self.assertGreater(
                trace_with_sq[-1],
                trace_without_sq[-1],
                f"Status quo dominates in trial 3, so trace with SQ should be higher. "
                f"Without SQ: {trace_without_sq}, With SQ: {trace_with_sq}",
            )

        with self.subTest("Single-objective: status quo is best in some trials"):
            # Create single-objective experiment where status quo is deliberately
            # the best arm in some trials.
            exp = get_experiment_with_observations(
                observations=[[11], [10], [9]], minimize=True
            )

            # Get the actual metric name from the experiment
            metric_name = list(exp.metrics.keys())[0]

            exp.status_quo = Arm(parameters={"x": 0.5, "y": 0.5}, name="status_quo")

            batch_trial1 = exp.new_batch_trial(should_add_status_quo_arm=True)
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.1, "y": 0.1}, name="mediocre_arm_1")
            )
            batch_trial1.add_arm(
                Arm(parameters={"x": 0.2, "y": 0.2}, name="mediocre_arm_2")
            )

            df_dict1 = [
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "mediocre_arm_1",
                    "mean": 15.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "mediocre_arm_2",
                    "mean": 20.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
                # Status quo: best value (lowest)
                {
                    "trial_index": batch_trial1.index,
                    "metric_name": metric_name,
                    "arm_name": "status_quo",
                    "mean": 5.0,
                    "sem": 0.0,
                    "metric_signature": metric_name,
                },
            ]
            exp.attach_data(Data(df=pd.DataFrame.from_records(df_dict1)))
            batch_trial1.mark_running(no_runner_required=True).mark_completed()

            # Get trace without status quo
            trace_without_sq = get_trace(exp, include_status_quo=False)

        with self.subTest("Include status quo: status quo included in trace"):
            # Get trace with status quo
            trace_with_sq = get_trace(exp, include_status_quo=True)

            # Both should have 4 values (3 initial + 1 batch trial)
            self.assertEqual(len(trace_without_sq), 4)
            self.assertEqual(len(trace_with_sq), 4)

            # The last value MUST differ because status quo is best
            # Without status quo: best in trial 3 is 15.0, cumulative min is 9
            # With status quo: best in trial 3 is 5.0, cumulative min is 5
            self.assertLess(
                trace_with_sq[-1],
                trace_without_sq[-1],
                f"Status quo is best in trial 3, so trace with SQ should be "
                f"lower (minimize). Without SQ: {trace_without_sq}, "
                f"With SQ: {trace_with_sq}",
            )

    def test_get_hypervolume(self) -> None:
        # W/ empty data.
        exp = get_experiment_with_trial()
        self.assertEqual(BestPointMixin._get_hypervolume(exp, Mock()), 0.0)

    def test_get_best_observed_value(self) -> None:
        # Alias for easier access.
        get_best = BestPointMixin._get_best_observed_value

        # Single objective, minimize.
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]], minimize=True
        )
        self.assertEqual(get_best(exp), 5)
        # Same experiment with maximize via new optimization config.
        opt_conf = none_throws(exp.optimization_config).clone()
        opt_conf.objective.minimize = False
        self.assertEqual(get_best(exp, opt_conf), 15)

        # Scalarized.
        exp = get_experiment_with_observations(
            observations=[[1, 1], [2, 2], [3, 3]],
            scalarized=True,
        )
        self.assertEqual(get_best(exp), 6)

        # Exclude out of design arms
        exp = get_experiment_with_observations(
            observations=[[11], [10], [9], [15], [5]],
            parameterizations=[
                {"x": 0.0, "y": 0.0},
                {"x": 0.1, "y": 0.0},
                {"x": 10.0, "y": 10.0},  # out of design
                {"x": 0.2, "y": 0.0},
                {"x": 10.1, "y": 10.0},  # out of design
            ],
            minimize=True,
        )
        self.assertEqual(get_best(exp), 10)  # 5 and 9 are out of design

    def _get_pe_search_space(self) -> SearchSpace:
        """Create a standard PE_EXPERIMENT search space with m1 and m2 parameters."""
        return SearchSpace(
            parameters=[
                RangeParameter(
                    name="m1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=10.0,
                ),
                RangeParameter(
                    name="m2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=10.0,
                ),
            ]
        )

    def _make_pref_opt_config(self, profile_name: str) -> PreferenceOptimizationConfig:
        """Create a PreferenceOptimizationConfig with m1 and m2 objectives."""
        return PreferenceOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric(name="m1"), minimize=False),
                    Objective(metric=Metric(name="m2"), minimize=False),
                ]
            ),
            preference_profile_name=profile_name,
        )

    def _assert_valid_trace(self, trace: list[float], expected_len: int) -> None:
        """Assert trace has expected length, contains floats, is non-decreasing and has
        more than one unique value."""
        self.assertEqual(len(trace), expected_len)
        for value in trace:
            self.assertIsInstance(value, float)
        for i in range(1, len(trace)):
            self.assertGreaterEqual(
                trace[i],
                trace[i - 1],
                msg=f"Trace not monotonically increasing at index {i}: {trace}",
            )
        unique_values = set(trace)
        self.assertGreater(
            len(unique_values),
            1,
            msg=f"Trace has only trivial values (all same): {trace}",
        )

    def test_get_trace_preference_learning_config(self) -> None:
        """Test that get_trace works correctly with PreferenceOptimizationConfig.

        This test verifies various scenarios for BOPE experiments,
        including cases with and without PE_EXPERIMENT data.
        """
        with self.subTest("without_pe_experiment_raises_error"):
            # Setup: Create a multi-objective experiment WITHOUT PE_EXPERIMENT
            exp = get_experiment_with_observations(
                observations=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            )
            exp.name = "main_experiment"
            pref_opt_config = self._make_pref_opt_config(
                profile_name="nonexistent_profile"
            )

            # Execute & Assert: Should raise UserInputError without PE_EXPERIMENT
            with self.assertRaisesRegex(
                UserInputError,
                "Preference profile 'nonexistent_profile' not found",
            ):
                get_trace(exp, pref_opt_config)

        with self.subTest("with_pe_experiment_empty_data_raises_error"):
            # Setup: Create main experiment
            exp = get_experiment_with_observations(
                observations=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            )
            exp.name = "main_experiment_empty"

            # Setup: Create PE_EXPERIMENT with no preference comparisons
            pe_experiment = Experiment(
                name="test_profile_empty",
                search_space=self._get_pe_search_space(),
            )

            # Setup: Attach PE_EXPERIMENT without any data
            aux_exp = AuxiliaryExperiment(experiment=pe_experiment, data=None)
            exp.add_auxiliary_experiment(
                auxiliary_experiment=aux_exp,
                purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            )
            pref_opt_config = self._make_pref_opt_config(
                profile_name="test_profile_empty"
            )

            # Execute & Assert: Should raise DataRequiredError due to empty data
            with self.assertRaisesRegex(
                DataRequiredError,
                "No preference data found in preference profile",
            ):
                get_trace(exp, pref_opt_config)

        with self.subTest("with_pe_experiment_valid_data_computes_utility"):
            # This subtest verifies that when PE_EXPERIMENT exists with valid data,
            # the code uses the preference model to compute utility-based traces.

            # Setup: Create main experiment with tracking data
            exp = get_experiment_with_observations(
                observations=[[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]],
            )
            exp.name = "main_experiment_with_pe"

            # Setup: Create PE_EXPERIMENT with minimal but well-separated preference
            # data
            pe_experiment = Experiment(
                name="test_profile_with_minimal_data",
                search_space=self._get_pe_search_space(),
            )

            # Setup: Add one pairwise preference comparison (minimal data)
            trial1 = pe_experiment.new_batch_trial()
            trial1.add_arm(Arm(name="0_0", parameters={"m1": 1.0, "m2": 1.0}))
            trial1.add_arm(Arm(name="0_1", parameters={"m1": 9.0, "m2": 9.0}))
            trial1.mark_running(no_runner_required=True).mark_completed()

            # Setup: Create minimal preference data
            pe_data_records = [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": Keys.PAIRWISE_PREFERENCE_QUERY.value,
                    "mean": 0.0,
                    "sem": 0.0,
                    "metric_signature": Keys.PAIRWISE_PREFERENCE_QUERY.value,
                },
                {
                    "trial_index": 0,
                    "arm_name": "0_1",
                    "metric_name": Keys.PAIRWISE_PREFERENCE_QUERY.value,
                    "mean": 1.0,
                    "sem": 0.0,
                    "metric_signature": Keys.PAIRWISE_PREFERENCE_QUERY.value,
                },
            ]
            pe_data = Data(df=pd.DataFrame.from_records(pe_data_records))
            pe_experiment.attach_data(pe_data)

            # Setup: Attach PE_EXPERIMENT to main experiment
            aux_exp = AuxiliaryExperiment(experiment=pe_experiment, data=pe_data)
            exp.add_auxiliary_experiment(
                auxiliary_experiment=aux_exp,
                purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            )
            pref_opt_config = self._make_pref_opt_config(
                profile_name="test_profile_with_minimal_data"
            )

            # Execute: With valid data, model computes utility-based trace
            trace = get_trace(exp, pref_opt_config)

            # Assert: Verify trace is valid, monotonically increasing, and non-trivial
            self._assert_valid_trace(
                trace,
                expected_len=3,
            )


class InferReferencePointFromExperimentTest(TestCase):
    def test_infer_reference_point_from_experiment(self) -> None:
        observations = [[-1.0, 1.0], [-0.5, 2.0], [-2.0, 0.5], [-0.1, 0.1]]
        # Getting an experiment with 2 objectives by the above observations.
        experiment = get_experiment_with_observations(
            observations=observations,
            minimize=True,
            scalarized=False,
            constrained=False,
        )
        data = experiment.fetch_data()
        inferred_reference_point = infer_reference_point_from_experiment(
            experiment, data=data
        )
        # The nadir point for this experiment is [-0.5, 0.5]. The function actually
        # deducts 0.1*Y_range from each of the objectives. Since the range for each
        # of the objectives is +/-1.5, the inferred reference point would
        # be [-0.35, 0.35].
        self.assertEqual(inferred_reference_point[0].op, ComparisonOp.LEQ)
        self.assertEqual(inferred_reference_point[0].bound, -0.35)
        self.assertEqual(inferred_reference_point[0].metric.signature, "m1")
        self.assertEqual(inferred_reference_point[1].op, ComparisonOp.GEQ)
        self.assertEqual(inferred_reference_point[1].bound, 0.35)
        self.assertEqual(inferred_reference_point[1].metric.signature, "m2")

        with mock.patch(
            "ax.service.utils.best_point.get_pareto_frontier_and_configs",
            return_value=([], [], [], []),
        ):
            with self.assertRaisesRegex(RuntimeError, "No frontier observations found"):
                infer_reference_point_from_experiment(experiment, data=data)

    def test_constrained_infer_reference_point_from_experiment(self) -> None:
        experiments = []
        observations = [[-1.0, 1.0], [-0.5, 2.0], [-2.0, 0.5], [-0.1, 0.1]]
        # adding constraint observations
        observations = [o + [c] for o, c in zip(observations, [1.0, 0.5, 1.0, 1.0])]
        # Getting an experiment with 2 objectives by the above observations.
        experiment = get_experiment_with_observations(
            observations=observations,
            minimize=True,
            scalarized=False,
            constrained=True,
        )
        experiments.append(experiment)

        # Special case: An experiment with no feasible observations.
        # TODO: Use experiment clone function once D50804778 is landed.
        experiment = copy.deepcopy(experiment)
        # Ensure that no observation is feasible.
        experiment.optimization_config.outcome_constraints[0].bound = 1000.0
        experiments.append(experiment)

        for experiment in experiments:
            # special case logs a warning message.
            data = experiment.fetch_data()
            if experiment.optimization_config.outcome_constraints[0].bound == 1000.0:
                with self.assertLogs(logger, "WARNING"):
                    inferred_reference_point = infer_reference_point_from_experiment(
                        experiment, data=data
                    )
            else:
                inferred_reference_point = infer_reference_point_from_experiment(
                    experiment, data=data
                )
            # The nadir point for this experiment is [-0.5, 0.5]. The function actually
            # deducts 0.1*Y_range from each of the objectives. Since the range for each
            # of the objectives is +/-1.5, the inferred reference point would
            # be [-0.35, 0.35].
            self.assertEqual(inferred_reference_point[0].op, ComparisonOp.LEQ)
            self.assertEqual(inferred_reference_point[0].bound, -0.35)
            self.assertEqual(inferred_reference_point[0].metric.signature, "m1")
            self.assertEqual(inferred_reference_point[1].op, ComparisonOp.GEQ)
            self.assertEqual(inferred_reference_point[1].bound, 0.35)
            self.assertEqual(inferred_reference_point[1].metric.signature, "m2")

    def test_infer_reference_point_from_experiment_shuffled_metrics(self) -> None:
        # Generating an experiment with given data.
        observations = [
            [-1.0, 1.0, 0.1],
            [-0.5, 2.0, 0.2],
            [-2.0, 0.5, 0.3],
            [-0.1, 0.1, 0.4],
        ]
        experiment = get_experiment_with_observations(
            observations=observations,
            minimize=True,
            scalarized=False,
            constrained=True,
        )

        # Constructing fake outputs for `get_pareto_frontier_and_configs` so that
        # the order of metrics `m1`, `m2` and `m3` are reversed.
        frontier_observations_shuffled = [
            Observation(
                features=ObservationFeatures(parameters={"x": 0.0, "y": 0.0}),
                data=ObservationData(
                    metric_signatures=["m3", "m2", "m1"],
                    means=np.array([0.1, 1.0, -1.0]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 0.1, "y": 0.1}),
                data=ObservationData(
                    metric_signatures=["m3", "m2", "m1"],
                    means=np.array([0.2, 2.0, -0.5]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 0.2, "y": 0.2}),
                data=ObservationData(
                    metric_signatures=["m3", "m2", "m1"],
                    means=np.array([0.3, 0.5, -2.0]),
                    covariance=np.diag(np.full(3, float("nan"))),
                ),
            ),
        ]
        f_shuffled = torch.tensor(
            [
                [0.1000, 1.0000, -1.0000],
                [0.2000, 2.0000, -0.5000],
                [0.3000, 0.5000, -2.0000],
            ],
            dtype=torch.float64,
        )
        obj_w_shuffled = torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
        )
        obj_t_shuffled = torch.tensor(
            [-torch.inf, -torch.inf, torch.inf], dtype=torch.float64
        )

        # Test the function with these shuffled output for
        # `get_pareto_frontier_and_configs`.
        with patch(
            "ax.service.utils.best_point.get_pareto_frontier_and_configs",
            return_value=(
                frontier_observations_shuffled,
                f_shuffled,
                obj_w_shuffled,
                obj_t_shuffled,
            ),
        ):
            inferred_reference_point = infer_reference_point_from_experiment(
                experiment, data=experiment.fetch_data()
            )

            self.assertEqual(inferred_reference_point[0].op, ComparisonOp.LEQ)
            self.assertEqual(inferred_reference_point[0].bound, -0.35)
            self.assertEqual(inferred_reference_point[0].metric.signature, "m1")
            self.assertEqual(inferred_reference_point[1].op, ComparisonOp.GEQ)
            self.assertEqual(inferred_reference_point[1].bound, 0.35)
            self.assertEqual(inferred_reference_point[1].metric.signature, "m2")

    def test_get_tensor_converter_adapter(self) -> None:
        # Test that it can convert experiments with different number of observations.
        for num_observations in (1, 10, 2000):
            experiment = get_experiment_with_observations(
                observations=[[0.0] for _ in range(num_observations)],
            )
            self.assertIsInstance(
                get_tensor_converter_adapter(experiment=experiment), TorchAdapter
            )
