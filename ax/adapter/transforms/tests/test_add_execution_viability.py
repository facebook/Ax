#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.base import Adapter, DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.add_execution_viability import AddExecutionViability
from ax.core.base_trial import TrialStatus
from ax.core.types import ComparisonOp
from ax.generators.base import Generator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_sobol,
)


class TestAddExecutionViability(TestCase):
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
        t = AddExecutionViability(
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
        self.assertEqual(feasibility_constraint.metric.name, "execution_viable")
        self.assertEqual(feasibility_constraint.op, ComparisonOp.GEQ)
        self.assertEqual(feasibility_constraint.bound, 0.8)
        self.assertFalse(feasibility_constraint.relative)

        # Create transform without specifying feasibility threshold
        adapter = self.get_adapter()
        t = AddExecutionViability(
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
        self.assertEqual(feasibility_constraint.bound, 0.8)

        # Check adapter.outcomes
        self.assertIn("execution_viable", adapter.outcomes)

    def test_transform_experiment_data(self) -> None:
        """Test that transform_experiment_data adds execution_viable metric based on
        trial status, including adding synthetic observations for abandoned trials
        without data. Covers both standard (2-level) and map (3-level) indices."""
        # Standard experiment (2-level index: trial_index, arm_name)
        standard_experiment = self.experiment
        standard_adapter = self.get_adapter()

        # Map experiment (3-level index: trial_index, arm_name, step)
        map_experiment = get_branin_experiment_with_timestamp_map_metric(
            with_trials_and_data=True,
        )
        map_experiment.trials[2].mark_abandoned()
        map_adapter = Adapter(
            search_space=map_experiment.search_space,
            generator=Generator(),
            experiment=map_experiment,
            data=map_experiment.lookup_data(),
        )

        cases = [
            ("standard", standard_experiment, standard_adapter, False),
            ("map_data", map_experiment, map_adapter, True),
        ]
        for label, experiment, adapter, expect_step in cases:
            with self.subTest(label=label):
                experiment_data = extract_experiment_data(
                    experiment, data_loader_config=DataLoaderConfig()
                )
                obs_data = experiment_data.observation_data
                has_step = "step" in obs_data.index.names
                self.assertEqual(has_step, expect_step)

                original_trial_indices = set(
                    obs_data.index.get_level_values("trial_index").unique()
                )
                abandoned_trials = experiment.trials_by_status.get(
                    TrialStatus.ABANDONED, []
                )
                abandoned_without_data = [
                    t for t in abandoned_trials if t.index not in original_trial_indices
                ]

                t = AddExecutionViability(
                    experiment_data=experiment_data,
                    adapter=adapter,
                    config={"min_abandoned_trials": 1},
                )
                transformed_data = t.transform_experiment_data(experiment_data)
                transformed_obs = transformed_data.observation_data

                # Index level should be preserved
                self.assertEqual("step" in transformed_obs.index.names, expect_step)

                # execution_viable metric was added
                self.assertIn("execution_viable", transformed_data.metric_signatures)

                # Abandoned trials without data were added
                trials_in_transformed = len(
                    transformed_obs.index.get_level_values("trial_index").unique()
                )
                self.assertEqual(
                    trials_in_transformed,
                    len(original_trial_indices) + len(abandoned_without_data),
                )

                # Abandoned trials have viability 0.0
                for trial in abandoned_trials:
                    loc_key = (
                        (trial.index, slice(None), slice(None))
                        if has_step
                        else (trial.index, slice(None))
                    )
                    viability = transformed_obs.loc[
                        loc_key, ("mean", "execution_viable")
                    ]
                    self.assertTrue(len(viability) > 0)
                    self.assertTrue((viability == 0.0).all())

                # Non-abandoned trials have viability 1.0
                non_abandoned = [
                    t
                    for t in experiment.trials.values()
                    if t.status != TrialStatus.ABANDONED
                ]
                for trial in non_abandoned:
                    loc_key = (
                        (trial.index, slice(None), slice(None))
                        if has_step
                        else (trial.index, slice(None))
                    )
                    viability = transformed_obs.loc[
                        loc_key, ("mean", "execution_viable")
                    ]
                    self.assertTrue((viability == 1.0).all())

    def test_no_adapter_raises_error(self) -> None:
        """Test that transform raises error when adapter is not provided."""
        # Setup: Create transform without adapter
        experiment_data = extract_experiment_data(
            self.experiment, data_loader_config=DataLoaderConfig()
        )
        t = AddExecutionViability(
            experiment_data=experiment_data,
            adapter=None,
            config={},
        )

        # Execute & Assert: Verify error is raised
        with self.assertRaisesRegex(
            ValueError,
            "Adapter must be provided for AddExecutionViability transform",
        ):
            t.transform_experiment_data(experiment_data)
