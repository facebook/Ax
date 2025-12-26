# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd

from ax.analysis.healthcheck.metric_fetching_errors import MetricFetchingErrorsAnalysis

from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.generation_strategy.dispatch_utils import choose_generation_strategy_legacy
from ax.orchestration.orchestrator import Orchestrator, OrchestratorOptions
from ax.utils.common.result import Err, Ok
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment

TEST_DATA = Data(
    df=pd.DataFrame(
        [
            {
                "arm_name": "0_0",
                "metric_name": "test_metric",
                "mean": 1.0,
                "sem": 2.0,
                "trial_index": 0,
                "metric_signature": "test_metric",
            }
        ]
    )
)


class TestMetricWithException(Metric):
    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @classmethod
    def fetch_trial_data_multi(
        cls,
        trial: BaseTrial,
        metrics: Iterable[Metric],
        **kwargs: Any,
    ) -> dict[str, MetricFetchResult]:
        return {
            metric.name: Err(
                MetricFetchE(
                    message="This is what I do",
                    exception=ValueError(
                        "The metric you are fetching is a test metric!"
                    ),
                )
            )
            for metric in metrics
        }


class TestMetricNoException(Metric):
    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @classmethod
    def fetch_trial_data_multi(
        cls,
        trial: BaseTrial,
        metrics: Iterable[Metric],
        **kwargs: Any,
    ) -> dict[str, MetricFetchResult]:
        return {
            metric.name: Err(MetricFetchE(message="This is what I do", exception=None))
            for metric in metrics
        }


class TestMetricSuccess(Metric):
    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @classmethod
    def fetch_trial_data_multi(
        cls,
        trial: BaseTrial,
        metrics: Iterable[Metric],
        **kwargs: Any,
    ) -> dict[str, MetricFetchResult]:
        return {metric.name: Ok(value=TEST_DATA) for metric in metrics}


def create_dummy_traceback_pastes(
    experiment_name: str,
    metric_fetch_errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for e in metric_fetch_errors:
        if e.get("paste") is not None:
            continue

        if e["traceback"]:
            e["paste"] = "P123"
        else:
            e["paste"] = "no traceback"
    return metric_fetch_errors


class TestMetricFetchingErrors(TestCase):
    def test_metric_fetching_errors_with_traceback(self) -> None:
        # GIVEN an experiment with a test metric and running trial
        exp = get_branin_experiment(with_batch=True)
        # it won't fetch an already completed trial
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.add_tracking_metric(TestMetricWithException(name="test_metric"))
        # AND GIVEN that experiment has tried to fetch data through the orchestrator
        orchestrator = Orchestrator(
            experiment=exp,
            generation_strategy=choose_generation_strategy_legacy(
                search_space=exp.search_space
            ),
            options=OrchestratorOptions(),
        )
        orchestrator.poll_and_process_results()
        self.assertEqual(len(exp._metric_fetching_errors), 1)
        # WHEN we compute MetricFetchingErrorsAnalysis with a traceback creator
        card = MetricFetchingErrorsAnalysis(
            add_traceback_paste_callable=create_dummy_traceback_pastes
        ).compute(experiment=exp)
        # THEN we get a card with a dataframe of errors
        self.assertEqual(len(card.df), 1)
        # AND the dataframe has the right columns in the right order
        self.assertEqual(
            list(card.df.columns),
            [
                "trial_index",
                "metric_name",
                "timestamp",
                "reason",
                "traceback",
            ],
        )
        self.assertEqual(
            card.df["trial_index"].iloc[0],
            0,
        )
        self.assertEqual(
            card.df["metric_name"].iloc[0],
            "test_metric",
        )
        self.assertEqual(
            card.df["reason"].iloc[0],
            "Ran into the following exception: ValueError: "
            "The metric you are fetching is a test metric!",
        )
        self.assertEqual(
            card.df["traceback"].iloc[0],
            "P123",
        )
        self.assertLessEqual(
            card.df["timestamp"].iloc[0],
            datetime.now().isoformat(),
        )
        self.assertGreaterEqual(
            card.df["timestamp"].iloc[0],
            (datetime.now() - timedelta(minutes=1)).isoformat(),
        )

    def test_metric_fetching_errors_without_traceback(self) -> None:
        # GIVEN an experiment with a test metric and running trial
        exp = get_branin_experiment(with_batch=True)
        # it won't fetch an already completed trial
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.add_tracking_metric(TestMetricNoException(name="test_metric"))
        # AND GIVEN that experiment has tried to fetch data through the orchestrator
        orchestrator = Orchestrator(
            experiment=exp,
            generation_strategy=choose_generation_strategy_legacy(
                search_space=exp.search_space
            ),
            options=OrchestratorOptions(),
        )
        orchestrator.poll_and_process_results()
        self.assertEqual(len(exp._metric_fetching_errors), 1)
        # WHEN we compute MetricFetchingErrorsAnalysis without a traceback creator
        card = MetricFetchingErrorsAnalysis().compute(experiment=exp)
        # THEN we get a card with a dataframe of errors
        self.assertEqual(len(card.df), 1)
        # AND the dataframe has the right columns in the right order
        self.assertEqual(
            list(card.df.columns),
            [
                "trial_index",
                "metric_name",
                "timestamp",
                "reason",
                "traceback",
            ],
        )
        self.assertEqual(
            card.df["trial_index"].iloc[0],
            0,
        )
        self.assertEqual(
            card.df["metric_name"].iloc[0],
            "test_metric",
        )
        self.assertEqual(card.df["reason"].iloc[0], "This is what I do")
        self.assertEqual(
            card.df["traceback"].iloc[0],
            "No traceback available",
        )
        self.assertLessEqual(
            card.df["timestamp"].iloc[0],
            datetime.now().isoformat(),
        )
        self.assertGreaterEqual(
            card.df["timestamp"].iloc[0],
            (datetime.now() - timedelta(minutes=1)).isoformat(),
        )

    def test_error_order(self) -> None:
        # GIVEN an experiment with a test metric and running trial
        exp = get_branin_experiment(with_batch=True)
        # it won't fetch an already completed trial
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.add_tracking_metric(TestMetricWithException(name="test_metric1"))
        exp.add_tracking_metric(TestMetricWithException(name="test_metric2"))
        # AND GIVEN that experiment has tried to fetch data through the orchestrator
        orchestrator = Orchestrator(
            experiment=exp,
            generation_strategy=choose_generation_strategy_legacy(
                search_space=exp.search_space
            ),
            options=OrchestratorOptions(),
        )
        orchestrator.poll_and_process_results()
        self.assertEqual(len(exp._metric_fetching_errors), 2)
        # WHEN we compute MetricFetchingErrorsAnalysis
        card = MetricFetchingErrorsAnalysis().compute(experiment=exp)
        # THEN we get a cards in descending ts order
        self.assertEqual(len(card.df), 2)
        self.assertGreater(
            card.df["timestamp"].iloc[0],
            card.df["timestamp"].iloc[1],
        )

    def test_error_gets_updated_for_same_metric(self) -> None:
        # This tests that an error in exp._metric_fetching_errors is updated
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.add_tracking_metric(TestMetricWithException(name="test_metric"))

        orchestrator = Orchestrator(
            experiment=exp,
            generation_strategy=choose_generation_strategy_legacy(
                search_space=exp.search_space
            ),
            options=OrchestratorOptions(),
        )
        orchestrator.poll_and_process_results()
        original_ts = exp._metric_fetching_errors[(0, "test_metric")]["timestamp"]
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        orchestrator.poll_and_process_results()

        self.assertEqual(len(exp._metric_fetching_errors), 1)
        card = MetricFetchingErrorsAnalysis().compute(experiment=exp)
        self.assertEqual(len(card.df), 1)
        self.assertGreater(card.df["timestamp"].iloc[0], original_ts)

    def test_error_gets_popped_on_successful_fetch(self) -> None:
        # This tests that an error in exp._metric_fetching_errors is popped
        # on a successful fetch
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.add_tracking_metric(TestMetricWithException(name="test_metric"))

        orchestrator = Orchestrator(
            experiment=exp,
            generation_strategy=choose_generation_strategy_legacy(
                search_space=exp.search_space
            ),
            options=OrchestratorOptions(),
        )
        orchestrator.poll_and_process_results()
        self.assertEqual(len(exp._metric_fetching_errors), 1)

        exp.trials[0].mark_running(no_runner_required=True, unsafe=True)
        exp.remove_tracking_metric("test_metric")
        exp.add_tracking_metric(TestMetricSuccess(name="test_metric"))
        orchestrator.poll_and_process_results()

        self.assertEqual(len(exp._metric_fetching_errors), 0)
