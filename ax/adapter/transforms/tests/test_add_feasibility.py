#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.add_feasibility import AddFeasibility
from ax.core.base_trial import TrialStatus
from ax.core.types import ComparisonOp
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_sobol


class TestAddFeasibility(TestCase):
    def setUp(self) -> None:
        self.experiment = get_branin_experiment(
            with_absolute_constraint=True,
            with_relative_constraint=True,
            with_completed_trial=True,
        )
        self.data = self.experiment.fetch_data()

        for _ in range(3):
            sobol_generator = get_sobol(search_space=self.experiment.search_space)
            sobol_run = sobol_generator.gen(n=1)
            trial = self.experiment.new_trial(generator_run=sobol_run)
            trial.mark_running(no_runner_required=True)
            trial.mark_abandoned()

    def get_adapter(self) -> Adapter:
        return Adapter(
            search_space=self.experiment.search_space,
            generator=Generator(),
            experiment=self.experiment,
            data=self.experiment.lookup_data(),
        )

    def test_transform_optimization_config(self) -> None:
        """Test that transform_optimization_config adds a feasibility constraint."""
        # Create transform with feasibility threshold
        adapter = self.get_adapter()
        t = AddFeasibility(
            experiment_data=extract_experiment_data(
                self.experiment, data_loader_config=DataLoaderConfig()
            ),
            adapter=adapter,
            config={"feasibility_threshold": 0.8},
        )
        original_constraints_count = len(
            self.experiment.optimization_config.outcome_constraints
        )
        new_opt_config = t.transform_optimization_config(
            self.experiment.optimization_config,
        )
        self.assertEqual(
            len(new_opt_config.outcome_constraints), original_constraints_count + 1
        )
        # Check that the last constraint is the feasibility constraint
        feasibility_constraint = new_opt_config.outcome_constraints[-1]
        self.assertEqual(feasibility_constraint.metric.name, "is_feasible")
        self.assertEqual(feasibility_constraint.op, ComparisonOp.GEQ)
        self.assertEqual(feasibility_constraint.bound, 0.8)
        self.assertFalse(feasibility_constraint.relative)

        # Create transform without specifying feasibility threshold
        adapter = self.get_adapter()
        t = AddFeasibility(
            experiment_data=extract_experiment_data(
                self.experiment, data_loader_config=DataLoaderConfig()
            ),
            adapter=adapter,
            config={},
        )
        new_opt_config = t.transform_optimization_config(
            self.experiment.optimization_config,
        )
        feasibility_constraint = new_opt_config.outcome_constraints[-1]
        self.assertEqual(feasibility_constraint.bound, 0.0)

        # Check adapter.outcomes
        self.assertIn("is_feasible", adapter.outcomes)

    def test_transform_experiment_data(self) -> None:
        """Test that transform_experiment_data adds feasibility metric based on
        trial status, including adding synthetic observations for abandoned trials
        without data."""
        adapter = self.get_adapter()
        experiment_data = extract_experiment_data(
            self.experiment, data_loader_config=DataLoaderConfig()
        )
        trials_in_original_data = len(
            experiment_data.observation_data.index.get_level_values(
                "trial_index"
            ).unique()
        )
        abandoned_trials = self.experiment.trials_by_status.get(
            TrialStatus.ABANDONED, []
        )

        t = AddFeasibility(
            experiment_data=experiment_data,
            adapter=adapter,
            config={"feasibility_threshold": 0.8},
        )
        transformed_data = t.transform_experiment_data(experiment_data)

        # Assert: Verify feasibility metric was added
        self.assertIn("is_feasible", transformed_data.metric_signatures)

        # Verify abandoned trials without data were added to transformed data
        trials_in_transformed_data = len(
            transformed_data.observation_data.index.get_level_values(
                "trial_index"
            ).unique()
        )
        self.assertEqual(
            trials_in_transformed_data,
            trials_in_original_data + len(abandoned_trials),
            "Abandoned trials without data should be added to transformed data",
        )

        # Verify abandoned trials have feasibility of 0.0
        for abandoned_trial in abandoned_trials:
            trial_idx = abandoned_trial.index
            self.assertIn(
                trial_idx,
                transformed_data.observation_data.index.get_level_values("trial_index"),
                f"Abandoned trial {trial_idx} should be in transformed data",
            )
            feasibility_value = transformed_data.observation_data.loc[
                (trial_idx, slice(None)), ("mean", "is_feasible")
            ].iloc[0]
            self.assertEqual(
                feasibility_value,
                0.0,
                f"Abandoned trial {trial_idx} should have feasibility of 0.0",
            )

        # Verify completed trials have feasibility of 1.0
        completed_trials = [
            trial
            for trial in self.experiment.trials.values()
            if trial.status != TrialStatus.ABANDONED
        ]
        for trial in completed_trials:
            trial_idx = trial.index
            feasibility_value = transformed_data.observation_data.loc[
                (trial_idx, slice(None)), ("mean", "is_feasible")
            ].iloc[0]
            self.assertEqual(
                feasibility_value,
                1.0,
                f"Non-abandoned trial {trial_idx} should have feasibility of 1.0",
            )

    def test_no_adapter_raises_error(self) -> None:
        """Test that transform raises error when adapter is not provided."""
        # Setup: Create transform without adapter
        experiment_data = extract_experiment_data(
            self.experiment, data_loader_config=DataLoaderConfig()
        )
        t = AddFeasibility(
            experiment_data=experiment_data,
            adapter=None,
            config={},
        )

        # Execute & Assert: Verify error is raised
        with self.assertRaisesRegex(
            ValueError, "Adapter must be provided for using feasibility constraints"
        ):
            t.transform_experiment_data(experiment_data)
