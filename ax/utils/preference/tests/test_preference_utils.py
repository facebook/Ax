#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.preference.preference_utils import get_preference_adapter
from ax.utils.testing.preference_stubs import get_pbo_experiment


class TestGetPreferenceAdapter(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Preference experiment with data for testing get_preference_adapter
        self.pe_experiment = get_pbo_experiment(
            parameter_names=["metric1", "metric2"],
            num_preference_trials=2,
            unbounded_search_space=True,
        )

    def test_get_preference_adapter(self) -> None:
        """Verify get_preference_adapter behavior with empty and valid data."""
        with self.subTest("raises_on_empty_data"):
            with self.assertRaisesRegex(
                DataRequiredError,
                "No preference data available",
            ):
                get_preference_adapter(experiment=self.pe_experiment, data=Data())

        with self.subTest("succeeds_with_valid_data"):
            data = self.pe_experiment.lookup_data()
            self.assertFalse(data.df.empty)

            adapter = get_preference_adapter(
                experiment=self.pe_experiment, data=self.pe_experiment.lookup_data()
            )
            self.assertIsNotNone(adapter)

    def test_registers_metric_when_not_present(self) -> None:
        """Verify get_preference_adapter registers the pref metric if not present.

        This tests the case when a PE experiment is loaded from storage without
        the pairwise_pref_query metric already registered (e.g., when called from
        best_point.py via find_auxiliary_experiment_by_name).
        """
        # Setup: Create a minimal PE experiment WITHOUT the metric registered
        pe_experiment = Experiment(
            name="test_pe_experiment_no_metric",
            search_space=SearchSpace(
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
            ),
        )

        # Verify the metric is NOT registered before calling get_preference_adapter
        pref_metric_name = Keys.PAIRWISE_PREFERENCE_QUERY.value
        self.assertNotIn(pref_metric_name, pe_experiment.metrics)

        # check it will err with empty data
        with self.assertRaisesRegex(
            DataRequiredError,
            "No preference data available",
        ):
            get_preference_adapter(
                experiment=pe_experiment, data=pe_experiment.lookup_data()
            )

        # Setup: Add a trial with preference data
        trial = pe_experiment.new_batch_trial()
        trial.add_arm(Arm(name="0_0", parameters={"m1": 0.5, "m2": 1.0}))
        trial.add_arm(Arm(name="0_1", parameters={"m1": 1.0, "m2": 1.5}))
        trial.mark_running(no_runner_required=True).mark_completed()

        # Setup: Create preference data
        pe_data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "trial_index": 0,
                        "arm_name": "0_0",
                        "metric_name": pref_metric_name,
                        "mean": 0.0,
                        "sem": 0.0,
                        "metric_signature": pref_metric_name,
                    },
                    {
                        "trial_index": 0,
                        "arm_name": "0_1",
                        "metric_name": pref_metric_name,
                        "mean": 1.0,
                        "sem": 0.0,
                        "metric_signature": pref_metric_name,
                    },
                ]
            )
        )

        # Execute: Call get_preference_adapter
        adapter = get_preference_adapter(experiment=pe_experiment, data=pe_data)

        # Assert: The adapter was created successfully
        self.assertIsNotNone(adapter)

        # Assert: The metric is now registered on the experiment
        self.assertIn(pref_metric_name, pe_experiment.metrics)
