# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.metric import Metric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.tests.scheduler_test_utils import (
    AxSchedulerTestCase,
    BrokenRunnerRuntimeError,
    BrokenRunnerValueError,
    InfinitePollRunner,
    RunnerToAllowMultipleMapMetricFetches,
    RunnerWithAllFailedTrials,
    RunnerWithEarlyStoppingStrategy,
    RunnerWithFailedAndAbandonedTrials,
    RunnerWithFrequentFailedTrials,
    SyntheticRunnerWithPredictableStatusPolling,
    SyntheticRunnerWithSingleRunningTrial,
    SyntheticRunnerWithStatusPolling,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_metric,
    get_branin_search_space,
    get_map_metric,
    get_multi_type_experiment,
)


class TestAxScheduler(AxSchedulerTestCase):
    """IMPORTANT! This class inherits AxSchedulerTestCase and will also
    run its associated tests.
    """

    pass


class TestAxSchedulerMultiTypeExperiment(AxSchedulerTestCase):
    """IMPORTANT! This class inherits AxSchedulerTestCase and will also
    run its associated tests.
    """

    EXPECTED_SCHEDULER_REPR: str = (
        "Scheduler(experiment=MultiTypeExperiment(branin_test_experiment), "
        "generation_strategy=GenerationStrategy(name='Sobol+BoTorch', "
        "steps=[Sobol for 5 trials, BoTorch for subsequent trials]), "
        "options=SchedulerOptions(max_pending_trials=10, "
        "trial_type=<TrialType.TRIAL: 0>, batch_size=None, "
        "total_trials=0, tolerated_trial_failure_rate=0.2, "
        "min_failed_trials_for_failure_rate_check=5, log_filepath=None, "
        "logging_level=20, ttl_seconds_for_trials=None, init_seconds_between_"
        "polls=10, min_seconds_before_poll=1.0, seconds_between_polls_backoff_"
        "factor=1.5, run_trials_in_batches=False, "
        "debug_log_run_metadata=False, early_stopping_strategy=None, "
        "global_stopping_strategy=None, suppress_storage_errors_after_"
        "retries=False, wait_for_running_trials=True, fetch_kwargs={}, "
        "validate_metrics=True, status_quo_weight=0.0, "
        "enforce_immutable_search_space_and_opt_config=True, "
        "mt_experiment_trial_type='type1', force_candidate_generation=False))"
    )

    def setUp(self) -> None:
        TestCase.setUp(self)
        self.branin_experiment = get_multi_type_experiment()
        self.branin_experiment.name = "branin_test_experiment"
        self.branin_experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=BraninMetric("m1", ["x1", "x2"]), minimize=True)
        )

        self.runner = SyntheticRunnerWithStatusPolling()
        self.branin_experiment.update_runner(trial_type="type1", runner=self.runner)

        self.branin_timestamp_map_metric_experiment = get_multi_type_experiment()
        self.branin_timestamp_map_metric_experiment.optimization_config = (
            OptimizationConfig(
                objective=Objective(
                    metric=get_map_metric(name="branin_map"), minimize=True
                )
            )
        )
        self.branin_timestamp_map_metric_experiment.update_runner(
            trial_type="type1", runner=RunnerToAllowMultipleMapMetricFetches()
        )

        self.branin_experiment_no_impl_runner_or_metrics = MultiTypeExperiment(
            search_space=get_branin_search_space(),
            optimization_config=OptimizationConfig(
                Objective(Metric(name="branin"), minimize=True)
            ),
            default_trial_type="type1",
            default_runner=None,
            name="branin_experiment_no_impl_runner_or_metrics",
        )
        self.sobol_MBM_GS = choose_generation_strategy(
            search_space=get_branin_search_space()
        )
        self.two_sobol_steps_GS = GenerationStrategy(  # Contrived GS to ensure
            steps=[  # that `DataRequiredError` is property handled in scheduler.
                GenerationStep(  # This error is raised when not enough trials
                    model=Models.SOBOL,  # have been observed to proceed to next
                    num_trials=5,  # geneneration step.
                    min_trials_observed=3,
                    max_parallelism=2,
                ),
                GenerationStep(model=Models.SOBOL, num_trials=-1, max_parallelism=3),
            ]
        )
        # GS to force the scheduler to poll completed trials after each ran trial.
        self.sobol_GS_no_parallelism = GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=-1, max_parallelism=1)]
        )
        self.scheduler_options_kwargs: dict[str, str | None] = {
            "mt_experiment_trial_type": "type1"
        }

    def test_init_with_no_impl_with_runner(self) -> None:
        self.branin_experiment_no_impl_runner_or_metrics.update_runner(
            trial_type="type1", runner=self.runner
        )
        super().test_init_with_no_impl_with_runner()

    def test_update_options_with_validate_metrics(self) -> None:
        self.branin_experiment_no_impl_runner_or_metrics.update_runner(
            trial_type="type1", runner=self.runner
        )
        super().test_update_options_with_validate_metrics()

    def test_retries(self) -> None:
        self.branin_experiment.update_runner("type1", BrokenRunnerRuntimeError())
        super().test_retries()

    def test_retries_nonretriable_error(self) -> None:
        self.branin_experiment.update_runner("type1", BrokenRunnerValueError())
        super().test_retries_nonretriable_error()

    def test_failure_rate_some_failed(self) -> None:
        self.branin_experiment.update_runner("type1", RunnerWithFrequentFailedTrials())
        super().test_failure_rate_some_failed()

    def test_failure_rate_all_failed(self) -> None:
        self.branin_experiment.update_runner("type1", RunnerWithAllFailedTrials())
        super().test_failure_rate_all_failed()

    def test_run_trials_and_yield_results_with_early_stopper(self) -> None:
        self.branin_experiment.update_runner("type1", InfinitePollRunner())
        super().test_run_trials_and_yield_results_with_early_stopper()

    def test_scheduler_with_metric_with_new_data_after_completion(self) -> None:
        self.branin_experiment.update_runner(
            "type1", SyntheticRunnerWithPredictableStatusPolling()
        )
        super().test_scheduler_with_metric_with_new_data_after_completion()

    def test_poll_and_process_results_with_reasons(self) -> None:
        self.branin_experiment.update_runner(
            "type1", RunnerWithFailedAndAbandonedTrials()
        )
        super().test_poll_and_process_results_with_reasons()

    def test_generate_candidates_works_for_iteration(self) -> None:
        self.branin_experiment.update_runner("type1", InfinitePollRunner())
        super().test_generate_candidates_works_for_iteration()

    def test_scheduler_with_odd_index_early_stopping_strategy(self) -> None:
        self.branin_timestamp_map_metric_experiment.update_runner(
            "type1", RunnerWithEarlyStoppingStrategy()
        )
        super().test_scheduler_with_odd_index_early_stopping_strategy()

    def test_fetch_and_process_trials_data_results_failed_non_objective(
        self,
    ) -> None:
        # add a tracking metric
        self.branin_timestamp_map_metric_experiment.add_tracking_metric(
            BraninMetric("branin", ["x1", "x2"]), trial_type="type1"
        )
        super().test_fetch_and_process_trials_data_results_failed_non_objective()

    def test_validate_options_not_none_mt_trial_type(
        self, msg: str | None = None
    ) -> None:
        # test if a MultiTypeExperiment with `mt_experiment_trial_type=None`
        self.scheduler_options_kwargs["mt_experiment_trial_type"] = None
        super().test_validate_options_not_none_mt_trial_type(
            msg="Must specify `mt_experiment_trial_type` for MultiTypeExperiment."
        )

    def test_run_n_trials_single_step_existing_experiment(
        self, all_completed_trials: bool = False
    ) -> None:
        self.branin_experiment.update_runner(
            "type1", SyntheticRunnerWithSingleRunningTrial()
        )
        super().test_run_n_trials_single_step_existing_experiment()
        metric_names = list(
            self.branin_experiment.lookup_data().df.metric_name.unique()
        )
        # assert only metric m1 is fetched (the metric for the current
        # trial_type)
        self.assertEqual(metric_names, ["m1"])

    def test_generate_candidates_does_not_generate_if_missing_data(self) -> None:
        self.branin_experiment.update_runner("type1", InfinitePollRunner())
        super().test_generate_candidates_does_not_generate_if_missing_data()

    def test_generate_candidates_does_not_generate_if_missing_opt_config(self) -> None:
        self.branin_experiment.update_runner("type1", InfinitePollRunner())
        self.branin_experiment.add_tracking_metric(
            get_branin_metric(), trial_type="type1"
        )
        super().test_generate_candidates_does_not_generate_if_missing_opt_config()
