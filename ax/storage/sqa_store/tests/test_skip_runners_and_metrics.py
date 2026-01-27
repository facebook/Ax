#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for skip_runners_and_metrics functionality with auxiliary experiments."""

from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.trial import Trial
from ax.exceptions.storage import JSONDecodeError
from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.delete import delete_experiment
from ax.storage.sqa_store.load import load_experiment
from ax.storage.sqa_store.save import save_experiment
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    CustomTestMetric,
    CustomTestRunner,
    get_experiment_with_custom_runner_and_metric,
    get_optimization_config,
    get_search_space,
)
from ax.utils.testing.modeling_stubs import get_generation_strategy
from pyre_extensions import assert_is_instance, none_throws


class SkipRunnersAndMetricsTest(TestCase):
    """Tests for loading experiments with skip_runners_and_metrics=True."""

    def setUp(self) -> None:
        super().setUp()
        init_test_engine_and_session_factory(force_init=True)

        # Create registry bundle with custom metric and runner
        registry_bundle = RegistryBundle(
            metric_clss={CustomTestMetric: None, Metric: 0},
            runner_clss={CustomTestRunner: None},
        )
        self.sqa_config = registry_bundle.sqa_config

        # Create auxiliary experiment with a custom metric in gen_metadata
        custom_metric = CustomTestMetric(
            name="custom_metric_in_gen_metadata", test_attribute="test"
        )

        self.aux_exp = Experiment(
            name="test_aux_exp_with_custom_metric_gen_metadata",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="Auxiliary experiment for testing gen_metadata with "
            "custom metrics",
            runner=CustomTestRunner(test_attribute="test"),
            is_test=True,
        )

        # Create a GeneratorRun with custom metric in gen_metadata
        gr = GeneratorRun(
            arms=[Arm(parameters={"w": 0.5, "x": 1, "y": "foo", "z": True})],
            gen_metadata={"custom_metric": custom_metric},
        )
        trial = self.aux_exp.new_trial(generator_run=gr)
        trial.mark_running(no_runner_required=True)

        # Save auxiliary experiment with custom registries
        save_experiment(self.aux_exp, config=self.sqa_config)

        # Create target experiment with auxiliary experiment attached
        self.purpose = AuxiliaryExperimentPurpose.PE_EXPERIMENT

        self.target_exp = Experiment(
            name="test_target_exp_with_aux_custom_metric_gen_metadata",
            search_space=get_search_space(),
            optimization_config=get_optimization_config(),
            description="Target experiment with aux experiment containing "
            "custom metric in gen_metadata",
            runner=CustomTestRunner(test_attribute="test"),
            tracking_metrics=[
                CustomTestMetric(name="tracking_metric", test_attribute="test")
            ],
            is_test=True,
            auxiliary_experiments_by_purpose={
                self.purpose: [AuxiliaryExperiment(experiment=self.aux_exp)]
            },
        )
        target_exp_gs = get_generation_strategy()
        self.target_exp.new_trial(
            target_exp_gs.gen_single_trial(experiment=self.target_exp)
        )

        # Save target experiment with custom registries
        save_experiment(self.target_exp, config=self.sqa_config)

    def tearDown(self) -> None:
        delete_experiment(exp_name=self.target_exp.name)
        delete_experiment(exp_name=self.aux_exp.name)
        super().tearDown()

    def test_load_without_skip_raises_json_decode_error(self) -> None:
        """Verify that loading with skip_runners_and_metrics but without reduced_state
        raises JSONDecodeError for custom metrics in gen_metadata.

        This test ensures the suite would catch issues like the Deltoid3Metric error
        where auxiliary experiments contain unregistered custom metric types in
        gen_metadata. With skip_runners_and_metrics=True but reduced_state=False,
        loading the experiment fails because the serialized CustomTestMetric in
        gen_metadata cannot be decoded.
        """
        # Without skip_runners_and_metrics, we get SQADecodeError for the metric type.
        # With skip_runners_and_metrics=True but reduced_state=False (default),
        # we get JSONDecodeError for the custom metric in gen_metadata.
        with self.assertRaises(JSONDecodeError) as cm:
            load_experiment(
                self.target_exp.name,
                skip_runners_and_metrics=True,
                # reduced_state defaults to False, so gen_metadata will be decoded
            )

        self.assertIn("CustomTestMetric", str(cm.exception))
        self.assertIn("not registered", str(cm.exception))

    def test_load_experiment_with_aux_exp_and_custom_metric_in_gen_metadata(
        self,
    ) -> None:
        """Test that loading with skip_runners_and_metrics=True works when
        auxiliary experiment has custom metric objects in gen_metadata.

        This verifies that experiments with auxiliary experiments containing
        serialized custom objects in gen_metadata can be loaded successfully
        when using skip_runners_and_metrics=True with reduced_state=True.
        """
        # Load with skip_runners_and_metrics=True and reduced_state=True
        # This should succeed without raising JSONDecodeError.
        loaded_exp = load_experiment(
            self.target_exp.name,
            skip_runners_and_metrics=True,
            reduced_state=True,
        )

        # Verify the experiment loaded successfully
        self.assertIsNotNone(loaded_exp)
        self.assertEqual(loaded_exp.name, self.target_exp.name)

        # Verify runner is None (due to skip_runners_and_metrics=True)
        self.assertIsNone(loaded_exp.runner)

        # Verify metrics are base Metric class
        self.assertEqual(
            loaded_exp.metrics["tracking_metric"].__class__,
            Metric,
        )

        # Verify auxiliary experiment was loaded
        self.assertEqual(len(loaded_exp.auxiliary_experiments_by_purpose), 1)
        loaded_aux_exp = loaded_exp.auxiliary_experiments_by_purpose[self.purpose][0]
        self.assertIsNotNone(loaded_aux_exp.experiment)

        # Verify the aux experiment's runner is also None
        self.assertIsNone(loaded_aux_exp.experiment.runner)

        # Verify the aux experiment's generator run gen_metadata is None
        # (due to reduced_state=True)
        loaded_gr = none_throws(
            assert_is_instance(loaded_aux_exp.experiment.trials[0], Trial).generator_run
        )
        self.assertIsNone(loaded_gr.gen_metadata)

    def test_resave_experiment_with_aux_exp_loses_custom_metrics_and_runner(
        self,
    ) -> None:
        """Test that re-saving after loading with skip_runners_and_metrics=True
        results in data loss for custom metrics and runners.

        This documents the expected behavior: when loading with
        skip_runners_and_metrics=True and then re-saving, custom metrics
        are converted to base Metric class and runners become None.
        """
        # Create experiment with custom runner and metric using helper
        experiment = get_experiment_with_custom_runner_and_metric(
            constrain_search_space=False,
            num_trials=1,
        )
        experiment.name = "test_exp_resave_data_loss"

        # Add auxiliary experiment from setUp
        experiment.auxiliary_experiments_by_purpose = {
            self.purpose: [AuxiliaryExperiment(experiment=self.aux_exp)]
        }

        # Verify original has custom classes
        self.assertEqual(experiment.runner.__class__, CustomTestRunner)
        self.assertEqual(
            experiment.metrics["custom_test_metric"].__class__,
            CustomTestMetric,
        )
        self.assertEqual(len(experiment.auxiliary_experiments_by_purpose), 1)

        # Save with custom registries from setUp
        save_experiment(experiment, config=self.sqa_config)

        try:
            # Load with skip_runners_and_metrics=True and reduced_state=True
            # This avoids JSONDecodeError from custom metrics in gen_metadata
            loaded_experiment = load_experiment(
                experiment.name,
                skip_runners_and_metrics=True,
                reduced_state=True,
            )

            # Verify loaded experiment has base classes
            self.assertIsNone(loaded_experiment.runner)
            self.assertEqual(
                loaded_experiment.metrics["custom_test_metric"].__class__,
                Metric,
            )

            # Re-save the loaded experiment (without custom registries)
            # This simulates what happens when a user loads and re-saves
            save_experiment(loaded_experiment)  # Using default config

            # Load again WITH custom registries to verify data loss
            # Even with custom registries, the data should now be base classes
            loaded_after_resave = load_experiment(
                experiment.name,
                config=self.sqa_config,
            )

            # Verify data loss occurred
            # Runner should be None (not CustomTestRunner)
            self.assertIsNone(loaded_after_resave.runner)

            # Metric should be base Metric (not CustomTestMetric)
            # Note: The metric will now be base Metric class since that's what was saved
            self.assertEqual(
                loaded_after_resave.metrics["custom_test_metric"].__class__,
                Metric,
            )

            # Verify trial runner is also None
            trial = loaded_after_resave.trials[0]
            self.assertIsNone(trial.runner)

            # Verify auxiliary experiment also has None runner
            loaded_aux = loaded_after_resave.auxiliary_experiments_by_purpose[
                self.purpose
            ][0]
            self.assertIsNone(loaded_aux.experiment.runner)
        finally:
            # Cleanup (aux_exp is cleaned up by tearDown)
            delete_experiment(exp_name=experiment.name)
