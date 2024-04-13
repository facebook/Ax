#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Sequence, Union

import numpy as np

from ax.core.types import TParamValue
from ax.exceptions.core import UnsupportedError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.telemetry.ax_client import AxClientCompletedRecord, AxClientCreatedRecord
from ax.telemetry.experiment import ExperimentCompletedRecord, ExperimentCreatedRecord
from ax.telemetry.generation_strategy import GenerationStrategyCreatedRecord
from ax.utils.common.testutils import TestCase


class TestAxClient(TestCase):
    def test_ax_client_created_record_from_ax_client(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            is_test=True,
        )

        record = AxClientCreatedRecord.from_ax_client(ax_client=ax_client)

        expected = AxClientCreatedRecord(
            experiment_created_record=ExperimentCreatedRecord.from_experiment(
                experiment=ax_client.experiment
            ),
            generation_strategy_created_record=(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=ax_client.generation_strategy
                )
            ),
            arms_per_trial=1,
            early_stopping_strategy_cls=None,
            global_stopping_strategy_cls=None,
            transformed_dimensionality=2,
        )
        self.assertEqual(record, expected)

        # Test with HSS & MOO.
        ax_client = AxClient()
        parameters: List[
            Dict[str, Union[TParamValue, Sequence[TParamValue], Dict[str, List[str]]]]
        ] = [
            {
                "name": "SearchSpace.optimizer",
                "type": "choice",
                "values": ["Adam", "SGD", "Adagrad"],
                "dependents": None,
                "is_ordered": False,
            },
            {"name": "SearchSpace.lr", "type": "range", "bounds": [0.001, 0.1]},
            {"name": "SearchSpace.fixed", "type": "fixed", "value": 12.0},
            {
                "name": "SearchSpace",
                "type": "fixed",
                "value": "SearchSpace",
                "dependents": {
                    "SearchSpace": [
                        "SearchSpace.optimizer",
                        "SearchSpace.lr",
                        "SearchSpace.fixed",
                    ]
                },
            },
        ]
        ax_client.create_experiment(
            name="hss_experiment",
            parameters=parameters,
            objectives={
                "branin": ObjectiveProperties(minimize=True),
                "b2": ObjectiveProperties(minimize=False),
            },
            is_test=True,
        )
        record = AxClientCreatedRecord.from_ax_client(ax_client=ax_client)

        expected = AxClientCreatedRecord(
            experiment_created_record=ExperimentCreatedRecord.from_experiment(
                experiment=ax_client.experiment
            ),
            generation_strategy_created_record=(
                GenerationStrategyCreatedRecord.from_generation_strategy(
                    generation_strategy=ax_client.generation_strategy
                )
            ),
            arms_per_trial=1,
            early_stopping_strategy_cls=None,
            global_stopping_strategy_cls=None,
            transformed_dimensionality=4,
        )
        self.assertEqual(record, expected)
        self.assertEqual(record.experiment_created_record.hierarchical_tree_height, 2)

    def test_ax_client_completed_record_from_ax_client(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={"branin": ObjectiveProperties(minimize=True)},
            is_test=True,
        )

        record = AxClientCompletedRecord.from_ax_client(ax_client=ax_client)

        expected = AxClientCompletedRecord(
            experiment_completed_record=ExperimentCompletedRecord.from_experiment(
                experiment=ax_client.experiment
            ),
            best_point_quality=float("nan"),
            model_fit_quality=float("nan"),
            model_std_quality=float("nan"),
            model_fit_generalization=float("nan"),
            model_std_generalization=float("nan"),
        )
        self._compare_axclient_completed_records(record, expected)

    def test_batch_trial_warning(self) -> None:
        ax_client = AxClient()
        error_msg = (
            "AxClient API does not support batch trials yet."
            " We plan to add this support in coming versions."
        )
        with self.assertRaisesRegex(UnsupportedError, error_msg):
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                ],
                objectives={"branin": ObjectiveProperties(minimize=True)},
                is_test=True,
                choose_generation_strategy_kwargs={
                    "use_batch_trials": True,
                },
            )

    def _compare_axclient_completed_records(
        self, record: AxClientCompletedRecord, expected: AxClientCompletedRecord
    ) -> None:
        self.assertEqual(
            record.experiment_completed_record, expected.experiment_completed_record
        )
        numeric_fields = [
            "best_point_quality",
            "model_fit_quality",
            "model_std_quality",
            "model_fit_generalization",
            "model_std_generalization",
        ]
        for field in numeric_fields:
            rec_field = getattr(record, field)
            exp_field = getattr(expected, field)
            if np.isnan(rec_field):
                self.assertTrue(np.isnan(exp_field))
            else:
                self.assertAlmostEqual(rec_field, exp_field)
