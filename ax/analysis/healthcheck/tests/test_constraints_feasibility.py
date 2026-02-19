# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.adapter.factory import get_sobol
from ax.adapter.registry import Generators
from ax.analysis.healthcheck.constraints_feasibility import (
    ConstraintsFeasibilityAnalysis,
    RESTRICTIVE_P_FEAS_THRESHOLD,
)
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance


class TestConstraintsFeasibilityAnalysis(TestCase):
    def _create_experiment_with_data(
        self, branin_d_means: list[float]
    ) -> tuple[Experiment, GenerationStrategy]:
        """
        Helper method to create an experiment with specified branin_d means.

        Args:
            branin_d_means: List of 6 mean values for branin_d metric

        Returns:
            Tuple of (experiment, generation_strategy)
        """
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=False,
            with_status_quo=True,
            with_relative_constraint=True,  # Constraint is >= -0.25%
        )
        df_metric_a = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_a"] * 6,
                "mean": list(np.random.normal(0, 1, 6)),
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_a"] * 6,
            }
        )
        df_metric_b = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_b"] * 6,
                "mean": list(np.random.normal(0, 1, 6)),
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_b"] * 6,
            }
        )
        df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": branin_d_means,
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_d"] * 6,
            }
        )
        df = pd.concat([df_metric_a, df_metric_b, df_metric_d], ignore_index=True)

        sobol = get_sobol(search_space=experiment.search_space)
        experiment.new_batch_trial(generator_run=sobol.gen(5))

        batch_trial = assert_is_instance(experiment.trials[0], BatchTrial)
        batch_trial.add_arm(experiment.status_quo)
        batch_trial.add_status_quo_arm(weight=1.0)
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()

        experiment.attach_data(data=Data(df=df))

        generation_strategy = GenerationStrategy(
            name="gs",
            nodes=[
                GenerationNode(
                    name="test_node",
                    generator_specs=[
                        GeneratorSpec(
                            generator_enum=Generators.BOTORCH_MODULAR,
                        )
                    ],
                )
            ],
        )
        generation_strategy.experiment = experiment
        generation_strategy._curr._fit(experiment=experiment)

        return experiment, generation_strategy

    @mock_botorch_optimize
    def test_compute_all_feasible(self) -> None:
        """Test when all constraints are individually feasible."""
        experiment, generation_strategy = self._create_experiment_with_data(
            branin_d_means=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )

        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Individual Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "All constraints are individually feasible.")
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)

        # Verify the dataframe has the expected structure
        df = card.df
        self.assertIn("constraint", df.columns)
        self.assertIn("num_arms_below_threshold", df.columns)
        self.assertIn("total_arms", df.columns)
        self.assertIn("fraction_below_threshold", df.columns)

    @mock_botorch_optimize
    def test_compute_restrictive_constraint(self) -> None:
        """Test when a constraint is overly restrictive."""
        # Create data where branin_d constraint is overly restrictive
        # (most arms have very negative values, making them violate the constraint)
        experiment, generation_strategy = self._create_experiment_with_data(
            branin_d_means=[1.0, -5.0, -5.0, -5.0, -5.0, -5.0]
        )

        cfa = ConstraintsFeasibilityAnalysis(
            restrictive_threshold=0.5, fraction_arms_threshold=0.5
        )
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Individual Constraints Feasibility Warning")
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)

        # The subtitle should mention the restrictive constraint(s)
        self.assertIn("overly restrictive constraint", card.subtitle)
        self.assertIn("Consider relaxing the bounds", card.subtitle)

        # Verify the dataframe has at least one constraint marked as restrictive
        df_result = card.df
        restrictive_constraints = df_result[
            df_result["fraction_below_threshold"] >= cfa.fraction_arms_threshold
        ]
        self.assertGreater(len(restrictive_constraints), 0)

    @mock_botorch_optimize
    def test_custom_thresholds(self) -> None:
        """Test with custom restrictive and fraction thresholds."""
        # Create data with moderately restrictive constraint
        experiment, generation_strategy = self._create_experiment_with_data(
            branin_d_means=[1.0, -2.0, -2.0, 3.0, 4.0, 5.0]
        )

        # Use a more lenient fraction threshold (only 30% of arms need to be below)
        cfa = ConstraintsFeasibilityAnalysis(
            restrictive_threshold=RESTRICTIVE_P_FEAS_THRESHOLD,
            fraction_arms_threshold=0.3,
        )
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        # This should catch the moderately restrictive constraint
        self.assertIn(
            card.get_status(), [HealthcheckStatus.WARNING, HealthcheckStatus.PASS]
        )

    @mock_botorch_optimize
    def test_no_constraints(self) -> None:
        """Test when experiment has no constraints."""
        # Create experiment with data, then remove constraints
        experiment, generation_strategy = self._create_experiment_with_data(
            branin_d_means=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin_a"), minimize=False),
            outcome_constraints=[],
        )

        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Individual Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "No constraints are specified.")
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)

    def test_no_optimization_config(self) -> None:
        """Test when experiment has no optimization config."""
        experiment = get_branin_experiment(has_optimization_config=False)
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(experiment=experiment, generation_strategy=None)

        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Individual Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "No optimization config is specified.")
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)

    @mock_botorch_optimize
    def test_dataframe_structure(self) -> None:
        """Test that the returned dataframe has the correct structure and values."""
        experiment, generation_strategy = self._create_experiment_with_data(
            branin_d_means=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )

        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )

        df = card.df

        # Check all expected columns are present
        expected_columns = [
            "constraint",
            "num_arms_below_threshold",
            "total_arms",
            "fraction_below_threshold",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check that we have one row per constraint
        # The experiment has one constraint (branin_d)
        self.assertEqual(len(df), 1)

        # Check that fraction_below_threshold is between 0 and 1
        self.assertTrue((df["fraction_below_threshold"] >= 0).all())
        self.assertTrue((df["fraction_below_threshold"] <= 1).all())

    @mock_botorch_optimize
    def test_validate_applicable_state(self) -> None:
        """Test validate_applicable_state for various scenarios."""
        icfa = ConstraintsFeasibilityAnalysis()

        # Test 1: No experiment provided
        validation_error = icfa.validate_applicable_state(
            experiment=None, generation_strategy=None, adapter=None
        )
        self.assertIsNotNone(validation_error)
        self.assertIn("experiment", validation_error.lower())

        # Test 2: Experiment with no optimization config (should be valid)
        experiment_no_opt_config = get_branin_experiment(has_optimization_config=False)
        validation_error = icfa.validate_applicable_state(
            experiment=experiment_no_opt_config, generation_strategy=None, adapter=None
        )
        self.assertIsNone(validation_error)

        # Test 3: Experiment with optimization config but no constraints
        # (should be valid)
        experiment_no_constraints = get_branin_experiment(with_status_quo=True)
        experiment_no_constraints.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin"), minimize=True),
            outcome_constraints=[],
        )
        validation_error = icfa.validate_applicable_state(
            experiment=experiment_no_constraints, generation_strategy=None, adapter=None
        )
        self.assertIsNone(validation_error)

        # Test 4: Experiment with constraints but no adapter (should fail)
        experiment_with_constraints, _ = self._create_experiment_with_data(
            branin_d_means=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        )
        validation_error = icfa.validate_applicable_state(
            experiment=experiment_with_constraints,
            generation_strategy=None,
            adapter=None,
        )
        self.assertIsNotNone(validation_error)

        # Test 5: Experiment with constraints and valid adapter (should be valid)
        experiment_with_adapter, generation_strategy = (
            self._create_experiment_with_data(
                branin_d_means=[1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            )
        )
        validation_error = icfa.validate_applicable_state(
            experiment=experiment_with_adapter,
            generation_strategy=generation_strategy,
            adapter=None,
        )
        self.assertIsNone(validation_error)
