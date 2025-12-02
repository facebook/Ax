#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from typing import cast
from unittest.mock import MagicMock, Mock, patch

from ax.core import OptimizationConfig
from ax.core.experiment import Experiment
from ax.core.map_data import MAP_KEY, MapData
from ax.core.metric import Metric
from ax.core.objective import MultiObjective
from ax.core.trial_status import TrialStatus
from ax.early_stopping.strategies import (
    BaseEarlyStoppingStrategy,
    ModelBasedEarlyStoppingStrategy,
    PercentileEarlyStoppingStrategy,
    ThresholdEarlyStoppingStrategy,
)
from ax.early_stopping.strategies.base import logger
from ax.early_stopping.strategies.logical import (
    AndEarlyStoppingStrategy,
    OrEarlyStoppingStrategy,
)
from ax.early_stopping.utils import align_partial_results
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.generation_node import GenerationNode
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_arms,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_experiment_with_multi_objective,
    get_test_map_data_experiment,
)
from pyre_extensions import assert_is_instance, none_throws


class FakeStrategy(BaseEarlyStoppingStrategy):
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        return False

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        return {}


class FakeStrategyRequiresNode(BaseEarlyStoppingStrategy):
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        return False

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        if current_node is None:
            raise ValueError("current_node is required")
        return {}


class ModelBasedFakeStrategy(ModelBasedEarlyStoppingStrategy):
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        return False

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        return {}


class TestBaseEarlyStoppingStrategy(TestCase):
    def test_early_stopping_strategy(self) -> None:
        # can't instantiate abstract class
        with self.assertRaises(TypeError):
            # pyre-ignore[45]: Cannot instantiate abstract class
            #  `BaseEarlyStoppingStrategy`.
            BaseEarlyStoppingStrategy()

    def test_all_objectives_and_directions_raises_error_when_lower_is_better_is_none(
        self,
    ) -> None:
        """Test that UnsupportedError is raised when a metric does not specify
        lower_is_better."""
        metric_without_direction = Metric(name="test_metric", lower_is_better=None)
        test_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        test_experiment.add_tracking_metric(metric_without_direction)

        # Execute & Assert: Verify that error is raised when using
        # metric_signatures
        es_strategy = FakeStrategy(
            metric_signatures=[metric_without_direction.signature]
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "Metrics used for early stopping must specify lower_is_better. ",
        ):
            es_strategy._all_objectives_and_directions(experiment=test_experiment)

    @patch.object(logger, "warning")
    def test_default_objective_and_direction(self, _: MagicMock) -> None:
        test_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        test_objective = none_throws(test_experiment.optimization_config).objective
        with self.subTest("provide metric names"):
            es_strategy = FakeStrategy(
                metric_signatures=[test_objective.metric.signature]
            )
            (
                actual_metric_name,
                actual_minimize,
            ) = es_strategy._default_objective_and_direction(experiment=test_experiment)

            self.assertEqual(
                actual_metric_name,
                test_objective.metric.name,
            )
            self.assertEqual(
                actual_minimize,
                test_objective.minimize,
            )

        with self.subTest("infer from optimization config"):
            # should be the same as above
            es_strategy = FakeStrategy()
            (
                actual_metric_name,
                actual_minimize,
            ) = es_strategy._default_objective_and_direction(experiment=test_experiment)

            self.assertEqual(
                actual_metric_name,
                test_objective.metric.name,
            )
            self.assertEqual(
                actual_minimize,
                test_objective.minimize,
            )

        test_multi_objective_experiment = get_experiment_with_multi_objective()
        test_multi_objective = cast(
            MultiObjective,
            none_throws(test_multi_objective_experiment.optimization_config).objective,
        )
        with self.subTest("infer from optimization config -- multi-objective"):
            es_strategy = FakeStrategy()
            (
                actual_metric_name,
                actual_minimize,
            ) = es_strategy._default_objective_and_direction(
                experiment=test_multi_objective_experiment
            )
            self.assertEqual(
                actual_metric_name,
                test_multi_objective.objectives[0].metric.name,
            )
            self.assertEqual(
                actual_minimize,
                test_multi_objective.objectives[0].minimize,
            )

        with self.subTest("provide metric names -- multi-objective"):
            es_strategy = FakeStrategy(
                metric_signatures=[test_multi_objective.objectives[1].metric.signature]
            )
            (
                actual_metric_name,
                actual_minimize,
            ) = es_strategy._default_objective_and_direction(
                experiment=test_multi_objective_experiment
            )
            self.assertEqual(
                actual_metric_name,
                test_multi_objective.objectives[1].metric.name,
            )
            self.assertEqual(
                actual_minimize,
                test_multi_objective.objectives[1].minimize,
            )

    @patch.object(logger, "warning")
    def test_is_eligible(self, _: MagicMock) -> None:
        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        es_strategy = FakeStrategy(min_progression=3, max_progression=5)
        metric_signature, __ = es_strategy._default_objective_and_direction(
            experiment=experiment
        )

        map_data = es_strategy._lookup_and_validate_data(
            experiment,
            metric_signatures=[metric_signature],
        )
        map_data = assert_is_instance(map_data, MapData)
        self.assertTrue(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
            )[0]
        )

        # try to get data from different metric name
        fake_df = deepcopy(map_data.map_df)
        trial_index = 0
        fake_df = fake_df.drop(fake_df.index[fake_df["trial_index"] == trial_index])
        fake_es, fake_reason = es_strategy.is_eligible(
            trial_index=trial_index,
            experiment=experiment,
            df=fake_df,
        )
        self.assertFalse(fake_es)
        self.assertEqual(
            fake_reason, "No data available to make an early stopping decision."
        )

        fake_map_data = es_strategy._lookup_and_validate_data(
            experiment,
            metric_signatures=["fake_metric_name"],
        )
        self.assertIsNone(fake_map_data)

        es_strategy = FakeStrategy(min_progression=5)
        self.assertFalse(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
            )[0]
        )

        es_strategy = FakeStrategy(min_progression=2, max_progression=3)
        self.assertFalse(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
            )[0]
        )

        # testing batch trial error
        experiment.new_batch_trial()
        with self.assertRaisesRegex(
            ValueError, "is a BatchTrial, which is not yet supported"
        ):
            es_strategy.is_eligible_any(
                trial_indices={0},
                experiment=experiment,
                df=map_data.map_df,
            )

    def test_progression_interval(self) -> None:
        """Test progression interval with min_progression=0."""
        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        # Set interval=2.0 with min_progression=0 -> boundaries at 0, 2, 4, 6...
        es_strategy = PercentileEarlyStoppingStrategy(
            min_progression=0.0,
            interval=2.0,
        )
        metric_signature, _ = es_strategy._default_objective_and_direction(
            experiment=experiment
        )

        map_data = es_strategy._lookup_and_validate_data(
            experiment,
            metric_signatures=[metric_signature],
        )
        map_df = assert_is_instance(map_data, MapData).map_df

        # Trial 0 has progressions at 0, 1, 2, 3, 4
        # Simulate orchestrator checks at different progressions

        # Check 1: Trial at progression 1 (between boundaries 0 and 2)
        # First check, so should be eligible
        df_at_1 = map_df[map_df[MAP_KEY] <= 1]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_1,
        )
        self.assertTrue(is_eligible)
        self.assertIsNone(reason)

        # Check 2: Trial at progression 2 (at boundary 2)
        # Has crossed boundary from 1 to 2, should be eligible
        df_at_2 = map_df[map_df[MAP_KEY] <= 2]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_2,
        )
        self.assertTrue(is_eligible)
        self.assertIsNone(reason)

        # Check 3: Trial at progression 3 (between boundaries 2 and 4)
        # Has NOT crossed boundary from 2 to 3, should NOT be eligible
        df_at_3 = map_df[map_df[MAP_KEY] <= 3]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_3,
        )
        self.assertFalse(is_eligible)
        self.assertIsNotNone(reason)
        # Validate message format: mentions boundary not crossed, shows interval,
        # and tells user what progression is needed
        self.assertRegex(
            reason,
            r"not crossed an interval boundary.*"
            r"both are in the same interval \[2\.00, 4\.00\).*"
            r"Must reach progression 4\.00",
        )

        # Check 4: Trial at progression 4 (at boundary 4)
        # Has crossed boundary from 3 to 4, should be eligible
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=map_df,
        )
        self.assertTrue(is_eligible)
        self.assertIsNone(reason)

    def test_progression_interval_with_min_progression(self) -> None:
        """Test progression interval with min_progression > 0."""
        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        # Set interval=2.0 with min_progression=1.0 -> boundaries at 1, 3, 5, 7...
        es_strategy = FakeStrategy(min_progression=1.0, interval=2.0)
        metric_signature, _ = es_strategy._default_objective_and_direction(
            experiment=experiment
        )

        map_data = es_strategy._lookup_and_validate_data(
            experiment,
            metric_signatures=[metric_signature],
        )
        map_df = assert_is_instance(map_data, MapData).map_df

        # Trial 0 has progressions at 0, 1, 2, 3, 4
        # With min_progression=1.0, boundaries are at 1, 3, 5, 7...

        # Check 1: Trial at progression 0 (below min_progression)
        # Should NOT be eligible due to min_progression requirement
        df_at_0 = map_df[map_df[MAP_KEY] <= 0]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_0,
        )
        self.assertFalse(is_eligible)
        self.assertIsNotNone(reason)
        self.assertIn("falls out of the min/max_progression range", reason)

        # Check 2: Trial at progression 2 (between boundaries 1 and 3)
        # First check at valid progression, should be eligible
        df_at_2 = map_df[map_df[MAP_KEY] <= 2]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_2,
        )
        self.assertTrue(is_eligible)
        self.assertIsNone(reason)

        # Check 3: Trial still at progression 2 (same interval)
        # Has NOT crossed boundary, should NOT be eligible
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_2,
        )
        self.assertFalse(is_eligible)
        self.assertIsNotNone(reason)
        self.assertRegex(
            reason,
            r"not crossed an interval boundary.*"
            r"both are in the same interval \[1\.00, 3\.00\).*"
            r"Must reach progression 3\.00",
        )

        # Check 4: Trial at progression 3 (at boundary 3)
        # Has crossed boundary from 2 to 3, should be eligible
        df_at_3 = map_df[map_df[MAP_KEY] <= 3]
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=df_at_3,
        )
        self.assertTrue(is_eligible)
        self.assertIsNone(reason)

        # Check 5: Trial at progression 4 (between boundaries 3 and 5)
        # Has NOT crossed boundary, should NOT be eligible
        is_eligible, reason = es_strategy.is_eligible(
            trial_index=0,
            experiment=experiment,
            df=map_df,
        )
        self.assertFalse(is_eligible)
        self.assertIsNotNone(reason)
        self.assertRegex(
            reason,
            r"not crossed an interval boundary.*"
            r"both are in the same interval \[3\.00, 5\.00\).*"
            r"Must reach progression 5\.00",
        )

    def test_validation(self) -> None:
        """Test validation of BaseEarlyStoppingStrategy parameters."""
        with self.subTest("interval_zero"):
            with self.assertRaisesRegex(
                UserInputError, "Option `interval` must be positive"
            ):
                FakeStrategy(interval=0)

        with self.subTest("interval_negative"):
            with self.assertRaisesRegex(
                UserInputError, "Option `interval` must be positive"
            ):
                FakeStrategy(interval=-1.0)

        with self.subTest("min_progression_negative"):
            with self.assertRaisesRegex(
                UserInputError,
                "Option `min_progression` must be nonnegative",
            ):
                FakeStrategy(min_progression=-1.0)

        with self.subTest("min_progression_zero_valid"):
            strategy = FakeStrategy(min_progression=0)
            self.assertEqual(strategy.min_progression, 0)

        with self.subTest("min_progression_positive_valid"):
            strategy = FakeStrategy(min_progression=5.0)
            self.assertEqual(strategy.min_progression, 5.0)

        with self.subTest("min_progression_equals_max_progression"):
            with self.assertRaisesRegex(
                UserInputError, "Expect min_progression < max_progression"
            ):
                FakeStrategy(min_progression=5.0, max_progression=5.0)

        with self.subTest("min_progression_greater_than_max_progression"):
            with self.assertRaisesRegex(
                UserInputError, "Expect min_progression < max_progression"
            ):
                FakeStrategy(min_progression=10, max_progression=5)

        with self.subTest("min_max_progression_valid"):
            strategy = FakeStrategy(min_progression=2.0, max_progression=10.0)
            self.assertEqual(strategy.min_progression, 2.0)
            self.assertEqual(strategy.max_progression, 10.0)

        with self.subTest("min_zero_max_positive_valid"):
            strategy = FakeStrategy(min_progression=0, max_progression=5.0)
            self.assertEqual(strategy.min_progression, 0)
            self.assertEqual(strategy.max_progression, 5.0)

    def test_check_safe_parameter(self) -> None:
        """Test that check_safe parameter controls whether _is_harmful is called."""
        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        trial_indices = {0, 1}

        with self.subTest("check_safe_false_bypasses_is_harmful"):
            # Setup: Create strategy with check_safe=False (default)
            strategy = FakeStrategy(check_safe=False)

            # Execute: Patch _is_harmful to verify it's not called
            with patch.object(strategy, "_is_harmful") as mock_is_harmful:
                strategy.should_stop_trials_early(
                    trial_indices=trial_indices,
                    experiment=experiment,
                )

                # Assert: _is_harmful should not be called when check_safe=False
                mock_is_harmful.assert_not_called()

        with self.subTest("check_safe_true_calls_is_harmful"):
            # Setup: Create strategy with check_safe=True
            strategy = FakeStrategy(check_safe=True)

            # Execute: Patch _is_harmful to verify it's called
            with patch.object(
                strategy, "_is_harmful", return_value=False
            ) as mock_is_harmful:
                strategy.should_stop_trials_early(
                    trial_indices=trial_indices,
                    experiment=experiment,
                )

                # Assert: _is_harmful should be called when check_safe=True
                mock_is_harmful.assert_called_once_with(
                    trial_indices=trial_indices,
                    experiment=experiment,
                )

        with self.subTest("check_safe_true_returns_empty_dict_when_harmful"):
            # Setup: Create strategy with check_safe=True
            strategy = FakeStrategy(check_safe=True)

            # Execute: Patch _is_harmful to return True (indicating harmful)
            with patch.object(strategy, "_is_harmful", return_value=True):
                result = strategy.should_stop_trials_early(
                    trial_indices=trial_indices,
                    experiment=experiment,
                )

                # Assert: Should return empty dict when early stopping is harmful
                self.assertEqual(result, {})

    def test_early_stopping_savings(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric()
        es_strategy = ModelBasedFakeStrategy(min_progression=3, max_progression=5)

        self.assertEqual(
            es_strategy.estimate_early_stopping_savings(
                experiment=exp,
            ),
            0,
        )

    def test_with_current_node(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric()
        es_strategy = FakeStrategyRequiresNode(min_progression=3, max_progression=5)

        with self.assertRaisesRegex(ValueError, "current_node is required"):
            es_strategy.should_stop_trials_early(
                trial_indices={0},
                experiment=exp,
            )

        es_strategy.should_stop_trials_early(
            trial_indices={0}, experiment=exp, current_node=Mock()
        )


class TestPercentileEarlyStoppingStrategy(TestCase):
    @patch.object(logger, "warning")
    def test_percentile_early_stopping_strategy_validation(self, _: MagicMock) -> None:
        exp = get_branin_experiment()

        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()
            trial.mark_as(status=TrialStatus.COMPLETED)

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        idcs = set(exp.trials.keys())
        exp.attach_data(data=exp.fetch_data())

        # Non-MapData attached
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

        exp = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        # No data attached
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

        exp.attach_data(data=exp.fetch_data())

        # Not enough learning curves
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            min_curves=6,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

        # Most recent progression below minimum
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            min_progression=3,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

    def test_percentile_early_stopping_strategy(self) -> None:
        with patch.object(logger, "debug") as logger_mock:
            self._test_percentile_early_stopping_strategy(
                logger_mock=logger_mock, non_objective_metric=False
            )

    def test_percentile_early_stopping_strategy_non_objective_metric(self) -> None:
        with patch.object(logger, "debug") as logger_mock:
            self._test_percentile_early_stopping_strategy(
                logger_mock=logger_mock, non_objective_metric=True
            )

        with self.assertRaisesRegex(
            UnsupportedError,
            "PercentileEarlyStoppingStrategy only supports a single metric.",
        ):
            PercentileEarlyStoppingStrategy(
                metric_signatures=["tracking_branin_map", "foo"],
                percentile_threshold=75,
                min_curves=5,
                min_progression=0.1,
            )

    def _test_percentile_early_stopping_strategy(
        self,
        logger_mock: MagicMock,
        non_objective_metric: bool,
    ) -> None:
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
            map_tracking_metric=non_objective_metric,
        )
        """
        Data looks like this:
        arm_name metric_name        mean  sem  trial_index  timestamp
        0       0_0      branin  146.138620  0.0            0          0
        1       0_0      branin  117.388086  0.0            0          1
        2       0_0      branin   99.950007  0.0            0          2
        3       1_0      branin  113.057480  0.0            1          0
        4       1_0      branin   90.815154  0.0            1          1
        5       1_0      branin   77.324501  0.0            1          2
        6       2_0      branin   44.627226  0.0            2          0
        7       2_0      branin   35.847504  0.0            2          1
        8       2_0      branin   30.522333  0.0            2          2
        9       3_0      branin  143.375669  0.0            3          0
        10      3_0      branin  115.168704  0.0            3          1
        11      3_0      branin   98.060315  0.0            3          2
        12      4_0      branin   65.033535  0.0            4          0
        13      4_0      branin   52.239184  0.0            4          1
        14      4_0      branin   44.479018  0.0            4          2

        Looking at the most recent fidelity only (timestamp==2), we have
        the following metric values for each trial:
        0: 99.950007 <-- worst
        3: 98.060315
        1: 77.324501
        4: 44.479018
        2: 30.522333 <-- best
        """
        if non_objective_metric:
            metric_signatures = ["tracking_branin_map"]
            # remove the optimization config to force that only the tracking metric can
            # be used for early stopping
            exp._optimization_config = None
            data = assert_is_instance(exp.fetch_data(), MapData)
            self.assertTrue((data.map_df["metric_name"] == "tracking_branin_map").all())
        else:
            metric_signatures = None

        idcs = set(exp.trials.keys())

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=metric_signatures,
            percentile_threshold=25,
            min_curves=4,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        if metric_signatures is None:
            logger_mock.assert_called_once_with(
                "No metric signatures specified. "
                "Defaulting to the objective metric(s).",
                stacklevel=2,
            )
        else:
            logger_mock.assert_not_called()

        self.assertEqual(set(should_stop), {0})

        # test ignore trial indices
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=metric_signatures,
            percentile_threshold=25,
            min_curves=4,
            min_progression=0.1,
            trial_indices_to_ignore=[0],
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=metric_signatures,
            percentile_threshold=50,
            min_curves=4,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 3})

        # respect trial_indices argument
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices={0}, experiment=exp
        )
        self.assertEqual(set(should_stop), {0})

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=metric_signatures,
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 3, 1})

        # not enough completed trials
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=metric_signatures,
            percentile_threshold=75,
            min_curves=5,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

    def test_percentile_early_stopping_with_n_best_trials_to_complete(self) -> None:
        """Test that top `n_best_trials_to_complete` trials are protected from
        early stopping."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )
        """
        Data looks like this (at step==2, the most recent progression):
        0: 99.950007 <-- worst
        3: 98.060315
        1: 77.324501
        4: 44.479018
        2: 30.522333 <-- best

        With percentile_threshold=50, trials 0 and 3 would normally be stopped.
        """
        idcs = set(exp.trials.keys())

        # Test 1: Preserve top 3 trials - should protect trial 1 from being stopped
        # even though it would normally be stopped at 75th percentile
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=3,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        # Only trials 0 and 3 should be stopped (trial 1 is protected as it's in top 3)
        self.assertEqual(set(should_stop), {0, 3})

        # Test 2: Preserve top 4 trials - should protect even more trials
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=4,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        # Only trial 0 should be stopped (trials 1, 2, 3, 4 are protected as top 4)
        self.assertEqual(set(should_stop), {0})

        # Test 3: Preserve all trials (n_best_trials_to_complete == total trials)
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=5,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        # No trials should be stopped (all 5 are protected)
        self.assertEqual(should_stop, {})

        # Test 4: Preserve all trials (edge case: n_best_trials_to_complete > total)
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=10,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        # No trials should be stopped (all 5 are protected)
        self.assertEqual(should_stop, {})

        # Test 5: With lower percentile threshold,
        # verify non-top-n_best_trials_to_complete trials still get stopped
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=25,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=2,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        # Trial 0 is worst and not in top 2, so should be stopped
        # Trials 2 and 4 are in top 2, so should be protected
        self.assertEqual(set(should_stop), {0})

    def test_percentile_reason_messages(self) -> None:
        """Test that appropriate reason messages are returned for different
        scenarios."""
        experiment = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )
        """
        Data at step==2:
        0: 99.950007 <-- worst
        3: 98.060315
        1: 77.324501
        4: 44.479018
        2: 30.522333 <-- best
        """
        trial_indices = {*experiment.trials.keys()}

        # Test 1: Verify reason message for trial that should be stopped
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=25,
            min_curves=4,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=trial_indices, experiment=experiment
        )
        # Trial 0 should be stopped
        self.assertIn(0, should_stop)
        reason = none_throws(should_stop[0])
        # Verify reason contains key information in correct format
        self.assertRegex(
            reason,
            r"Trial objective values at progressions in \[[\d\.]+, "
            r"[\d\.]+\] are all worse than 25\.0-th percentile across "
            r"comparable trials",
        )
        self.assertRegex(reason, r"Progressions: \[2.0\]")
        self.assertRegex(reason, r"Underperforms: \[True\]")
        self.assertRegex(reason, r"Trial objective values: \[[\d\.]+\]")
        self.assertRegex(reason, r"Thresholds: \[[\d\.]+\]")
        self.assertRegex(reason, r"Number of trials: \[5\]")

        # Test 2: Verify reason message for trial that should NOT be stopped
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
        )
        # Use _should_stop_trial_early directly to get reason for non-stopped trial
        data = none_throws(
            early_stopping_strategy._lookup_and_validate_data(
                experiment, metric_signatures=["branin_map"]
            )
        )
        aligned_df = align_partial_results(df=data.map_df, metrics=["branin_map"])
        aligned_means = aligned_df["mean"]["branin_map"]

        should_stop, reason = early_stopping_strategy._should_stop_trial_early(
            trial_index=2,  # Best trial
            experiment=experiment,
            wide_df=aligned_means,
            long_df=data.map_df,
            minimize=True,
        )
        self.assertFalse(should_stop)
        reason = none_throws(reason)
        # Verify reason contains key information in correct format
        self.assertRegex(
            reason,
            r"Trial objective values at progressions in \[[\d\.]+, "
            r"[\d\.]+\] are not all worse than 75\.0-th percentile across "
            r"comparable trials",
        )
        self.assertRegex(reason, r"Progressions: \[2.0\]")
        self.assertRegex(reason, r"Underperforms: \[False\]")
        self.assertRegex(reason, r"Trial objective values: \[[\d\.]+\]")
        self.assertRegex(reason, r"Thresholds: \[[\d\.]+\]")
        self.assertRegex(reason, r"Number of trials: \[5\]")

    def test_top_trials_reason_messages_with_percentile_info(self) -> None:
        """Test that reason messages for top trials include both protection and
        percentile information."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )
        """
        Data at step==2:
        0: 99.950007 <-- worst
        3: 98.060315
        1: 77.324501
        4: 44.479018
        2: 30.522333 <-- best
        """
        # Use 75th percentile which would stop trials 0, 3, and 1
        # But protect top 3 trials (2, 4, 1)
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=3,
        )

        data = none_throws(
            early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
        )
        aligned_df = align_partial_results(df=data.map_df, metrics=["branin_map"])
        aligned_means = aligned_df["mean"]["branin_map"]

        # Test trial 1 which is in top 3 but below percentile threshold
        should_stop, reason = early_stopping_strategy._should_stop_trial_early(
            trial_index=1,  # Trial 1 is in top 3 but below percentile
            experiment=exp,
            wide_df=aligned_means,
            long_df=data.map_df,
            minimize=True,
        )

        # Should not be stopped because it's in top 3
        self.assertFalse(should_stop)
        reason = none_throws(reason)
        # Verify reason contains both protection info and percentile threshold info
        # Pattern validates: protection message + percentile threshold explanation
        self.assertRegex(
            reason, r"Trial 1 is among the top-3 trials.*so will not be early-stopped"
        )

    def test_early_stopping_with_n_best_protection_handles_ties(self) -> None:
        """Test that all trials with tied objective values are protected when they
        fall within the top n_best_trials_to_complete ranks.

        This test verifies the fix for a bug where the old sort_values().head(n)
        approach would only protect the first n trials based on DataFrame ordering,
        potentially leaving other trials with identical performance unprotected.
        """
        # Create experiment with 5 trials
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        data = assert_is_instance(exp.fetch_data(), MapData)

        # Manually set objective values to create ties
        # At progression=2, set trials 0, 1, 2 to have the same best value (30.0)
        # and trials 3, 4 to have worse values
        progression_2_mask = (data.map_df["metric_name"] == "branin_map") & (
            data.map_df[MAP_KEY] == 2
        )

        # Set values: trials 0, 1, 2 all have value 30.0 (tied for best)
        # trials 3, 4 have worse values 90.0, 95.0
        for trial_idx, value in [(0, 30.0), (1, 30.0), (2, 30.0), (3, 90.0), (4, 95.0)]:
            trial_mask = progression_2_mask & (data.map_df["trial_index"] == trial_idx)
            data.map_df.loc[trial_mask, "mean"] = value

        exp.attach_data(data=data)

        # Use n_best_trials_to_complete=2, which is less than the 3 tied top trials
        # With rank-based logic, all 3 tied trials (rank=1) should be protected
        # With old head-based logic, only 2 of the 3 would be protected
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=50,  # Would stop bottom 50%
            min_curves=4,
            min_progression=0.1,
            n_best_trials_to_complete=2,  # Less than the 3 tied top trials
        )

        data_lookup = none_throws(
            early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
        )
        aligned_df = align_partial_results(
            df=data_lookup.map_df, metrics=["branin_map"]
        )
        aligned_means = aligned_df["mean"]["branin_map"]

        # Test that ALL three tied top trials (0, 1, 2) are protected
        # even though n_best_trials_to_complete=2
        for trial_idx in [0, 1, 2]:
            should_stop, reason = early_stopping_strategy._should_stop_trial_early(
                trial_index=trial_idx,
                experiment=exp,
                wide_df=aligned_means,
                long_df=data_lookup.map_df,
                minimize=True,
            )

            # All three tied trials should be protected
            self.assertFalse(
                should_stop,
                f"Trial {trial_idx} should be protected (tied with rank 1) "
                f"but should_stop={should_stop}",
            )
            self.assertIsNotNone(reason)
            self.assertRegex(
                none_throws(reason),
                rf"Trial {trial_idx} is among the top-2 trials.*"
                rf"so will not be early-stopped",
            )

    def test_patience_parameter_validation(self) -> None:
        """Test that patience parameter is validated correctly."""
        with self.assertRaisesRegex(
            UserInputError, "patience must be non-negative, got -1"
        ):
            PercentileEarlyStoppingStrategy(patience=-1)

        # Valid patience values should not raise
        strategy_zero = PercentileEarlyStoppingStrategy(patience=0)
        self.assertEqual(strategy_zero.patience, 0)

        strategy_positive = PercentileEarlyStoppingStrategy(patience=5)
        self.assertEqual(strategy_positive.patience, 5)

    def test_patience_basic_functionality(self) -> None:
        """Test basic patience functionality."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )
        """
        Data looks like (timestamps 0, 1, 2):
        Trial 0: [146.14, 117.39, 99.95] - consistently worst
        Trial 1: [113.06, 90.82, 77.32]
        Trial 2: [44.63, 35.85, 30.52] - consistently best
        Trial 3: [143.38, 115.17, 98.06] - consistently second worst
        Trial 4: [65.03, 52.24, 44.48]
        """
        with self.subTest("patience_zero_uses_single_point_evaluation"):
            # With patience=0 and percentile_threshold=25, only trial 0 should stop
            # based on its performance at step==2 (latest).
            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=25,
                min_progression=0,
                min_curves=4,
                patience=0,
            )
            should_stop = early_stopping_strategy.should_stop_trials_early(
                trial_indices=set(exp.trials.keys()), experiment=exp
            )
            self.assertEqual(set(should_stop), {0})

        with self.subTest("patience_with_consistent_underperformance"):
            # With patience=2 (window of 3 steps: [0, 1, 2]), we check if trial
            # underperforms at ALL steps in the window.
            # Trial 0 is in bottom 25% at all steps [0, 1, 2], so should stop
            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=25,
                min_progression=0,
                min_curves=4,
                patience=2,
            )
            should_stop = early_stopping_strategy.should_stop_trials_early(
                trial_indices=set(exp.trials.keys()), experiment=exp
            )
            self.assertEqual(set(should_stop), {0})

    def test_patience_underperformance_patterns(self) -> None:
        """Test patience with different underperformance patterns."""
        with self.subTest("inconsistent_underperformance"):
            # Create experiment with custom data to ensure inconsistent performance
            exp = get_test_map_data_experiment(
                num_trials=5,
                num_fetches=3,
                num_complete=5,
            )
            data = assert_is_instance(exp.fetch_data(), MapData)

            # Manually modify trial 0's performance to be:
            # - Bad at step 0 (100.0)
            # - Good at step 1 (20.0) <- breaks consistency
            # - Bad at step 2 (100.0)
            # This creates inconsistent underperformance
            modified_df = data.map_df.copy()
            branin_mask = modified_df["metric_name"] == "branin_map"
            trial_0_mask = branin_mask & (modified_df["trial_index"] == 0)
            modified_df.loc[trial_0_mask, "mean"] = [100.0, 20.0, 100.0]

            # Set other trials to have medium performance (50.0)
            other_trials_mask = branin_mask & (
                modified_df["trial_index"].isin([1, 2, 3, 4])
            )
            modified_df.loc[other_trials_mask, "mean"] = 50.0

            # Create new MapData with modified dataframe
            data = MapData(df=modified_df)
            exp.attach_data(data=data)

            """
            With patience=2 (window [0, 1, 2]):
            - Trial 0 at step 0: 100.0 (worse than 25th percentile ~50.0)
            - Trial 0 at step 1: 20.0 (better than 25th percentile ~50.0) <- NOT WORSE
            - Trial 0 at step 2: 100.0 (worse than 25th percentile ~50.0)

            Since trial 0 does NOT underperform at ALL steps, it should NOT be stopped.
            """
            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=25,
                min_curves=4,
                patience=2,
            )
            should_stop = early_stopping_strategy.should_stop_trials_early(
                trial_indices={0}, experiment=exp
            )
            # Trial 0 should NOT be stopped due to inconsistent performance
            self.assertEqual(should_stop, {})

        with self.subTest("noisy_curves"):
            # Test that patience prevents stopping trials with noisy/volatile curves
            exp = get_test_map_data_experiment(
                num_trials=5,
                num_fetches=5,  # More steps to show volatility
                num_complete=5,
            )
            data = assert_is_instance(exp.fetch_data(), MapData)

            # Create a volatile trial that alternates between good and bad
            # Trial 0: [10, 90, 10, 90, 10] - volatile but has good steps
            # Other trials: consistent medium values around 50
            modified_df = data.map_df.copy()
            metric_mask = modified_df["metric_name"] == "branin_map"
            trial_0_mask = metric_mask & (modified_df["trial_index"] == 0)

            volatile_values = [10.0, 90.0, 10.0, 90.0, 10.0]
            modified_df.loc[trial_0_mask, "mean"] = volatile_values

            # Set other trials to consistent medium performance
            trial_mask = metric_mask & (modified_df["trial_index"].isin([1, 2, 3, 4]))
            modified_df.loc[trial_mask, "mean"] = 50.0

            # Create new MapData with modified dataframe
            data = MapData(df=modified_df)
            exp.attach_data(data=data)

            """
            At latest step (4), trial 0 has value 10.0 (best).
            With patience=0, we'd only look at step 4 where trial 0 is best.
            With patience=2, window is [2, 3, 4] with values [10, 90, 10]:
            - At step 2: 10 (good)
            - At step 3: 90 (bad)
            - At step 4: 10 (good)
            Trial doesn't consistently underperform, so shouldn't be stopped.
            """
            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=50,
                min_curves=4,
                patience=2,
            )

            should_stop = early_stopping_strategy.should_stop_trials_early(
                trial_indices={0}, experiment=exp
            )

            # Trial 0 should NOT be stopped because it doesn't consistently
            # underperform in the window [2, 3, 4]
            self.assertEqual(should_stop, {})

    def test_patience_with_insufficient_data(self) -> None:
        """Test that trials are not stopped when there is insufficient data."""
        with self.subTest("insufficient_progressions"):
            # Test with too few progressions in the window
            exp = get_test_map_data_experiment(
                num_trials=5,
                num_fetches=1,  # Only 1 fetch (step 0 only)
                num_complete=4,
            )
            """
            With only 1 progression (0) and patience=2:
            - Latest progression: 0
            - Window: [0-2, 0] = [-2, 0]
            - Only progression 0 exists in this range
            - We need >1 progressions in window when patience > 0
            """
            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=25,
                min_curves=3,
                min_progression=0.0,  # Set to 0 to allow step 0
                patience=2,
            )

            data_lookup = none_throws(
                early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
            )
            aligned_df = align_partial_results(
                df=data_lookup.map_df, metrics=["branin_map"]
            )
            aligned_means = aligned_df["mean"]["branin_map"]

            should_stop, reason = early_stopping_strategy._should_stop_trial_early(
                trial_index=0,
                experiment=exp,
                wide_df=aligned_means,
                long_df=data_lookup.map_df,
                minimize=True,
            )

            # Should not stop due to insufficient progressions
            self.assertFalse(should_stop)
            reason = none_throws(reason)
            self.assertRegex(
                reason,
                r"Fewer than 2 progressions in patience window.*Not stopping trial",
            )

        with self.subTest("insufficient_curves_at_progression"):
            # Test with insufficient curves/trials at a progression
            MIN_CURVES = 4

            exp = get_test_map_data_experiment(
                num_trials=5,
                num_fetches=3,
                num_complete=5,
            )

            early_stopping_strategy = PercentileEarlyStoppingStrategy(
                percentile_threshold=50,
                min_curves=MIN_CURVES,
                min_progression=0.0,
                patience=1,  # Window [1, 2]
            )

            data_lookup = none_throws(
                early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
            )

            # Modify the data to simulate insufficient trials at progression 2
            modified_df = data_lookup.map_df.copy()
            selector = ~(
                (modified_df["metric_name"] == "branin_map")
                & (modified_df[MAP_KEY] == 2)
                & (modified_df["trial_index"].isin([2, 3, 4]))
            )
            modified_df = modified_df[selector]
            """
            After modification at progression 2: only trials 0, 1 have data
            (2 < min_curves=4). Trial should NOT be stopped due to
            insufficient curves at progression 2.
            """
            aligned_df = align_partial_results(df=modified_df, metrics=["branin_map"])
            aligned_means = aligned_df["mean"]["branin_map"]

            should_stop, reason = early_stopping_strategy._should_stop_trial_early(
                trial_index=0,
                experiment=exp,
                wide_df=aligned_means,
                long_df=modified_df,
                minimize=True,
            )

            # Should not stop due to insufficient trials at progression 2
            self.assertFalse(should_stop)
            reason = none_throws(reason)
            self.assertRegex(
                reason,
                r"Insufficiently many trials with data at progressions in window",
            )
            self.assertRegex(reason, rf"Minimum required: {MIN_CURVES}")

    def test_patience_with_n_best_trials_interaction(self) -> None:
        """Test that n_best_trials_to_complete protection works correctly
        with patience parameter."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )
        """
        Data at all steps:
        Trial 0: consistently worst across all steps
        Trial 1: medium performance
        Trial 2: consistently best across all steps
        Trial 3: consistently second worst
        Trial 4: medium-good performance

        With patience=2, percentile_threshold=75, n_best_trials_to_complete=2:
        - Trials 2 and 4 should be protected (top 2)
        - Trial 1 would normally be stopped (below 75th percentile)
        - But we need to check if it's protected
        """
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=75,
            min_progression=0,
            min_curves=4,
            patience=2,
            n_best_trials_to_complete=3,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=set(exp.trials.keys()), experiment=exp
        )

        # Trials 0 and 3 should be stopped (consistently worst and not in top 3)
        # Trial 1 should be protected (in top 3)
        self.assertEqual(set(should_stop), {0, 3})

    def test_patience_reason_messages(self) -> None:
        """Test that reason messages include patience window information."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=3,
            num_complete=4,
        )

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=25,
            min_progression=0,
            min_curves=4,
            patience=2,
        )

        data = none_throws(
            early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
        )
        aligned_df = align_partial_results(df=data.map_df, metrics=["branin_map"])
        aligned_means = aligned_df["mean"]["branin_map"]

        # Test trial that should be stopped
        should_stop, reason = early_stopping_strategy._should_stop_trial_early(
            trial_index=0,
            experiment=exp,
            wide_df=aligned_means,
            long_df=data.map_df,
            minimize=True,
        )

        self.assertTrue(should_stop)
        reason = none_throws(reason)
        # Verify reason contains window information
        self.assertRegex(
            reason,
            r"Trial objective values at progressions in \[[\d\.]+, [\d\.]+\] "
            r"are all worse than 25\.0-th percentile",
        )
        self.assertRegex(reason, r"Progressions:")
        self.assertRegex(reason, r"Underperforms:")
        self.assertRegex(reason, r"Trial objective values:")
        self.assertRegex(reason, r"Thresholds:")

    def test_patience_respects_min_progression(self) -> None:
        """Test that patience window respects min_progression boundary."""
        exp = get_test_map_data_experiment(
            num_trials=5,
            num_fetches=5,  # Create progressions [0, 1, 2, 3, 4]
            num_complete=5,
        )
        data = assert_is_instance(exp.fetch_data(), MapData)

        # Create a trial that is consistently bad from step 0-4
        # Trial 0: [100, 100, 100, 100, 100] - consistently worst
        # Other trials: [50, 50, 50, 50, 50] - medium performance
        modified_df = data.map_df.copy()

        # Create base mask for branin_map metric once
        branin_mask = modified_df["metric_name"] == "branin_map"

        # Set trial 0 to consistently bad performance
        trial_0_mask = branin_mask & (modified_df["trial_index"] == 0)
        modified_df.loc[trial_0_mask, "mean"] = 100.0

        # Set other trials to medium performance
        other_trials_mask = branin_mask & (
            modified_df["trial_index"].isin([1, 2, 3, 4])
        )
        modified_df.loc[other_trials_mask, "mean"] = 50.0

        # Create new MapData with modified dataframe
        data = MapData(df=modified_df)
        exp.attach_data(data=data)

        """
        Test scenario:
        - min_progression = 2.0 (we don't trust data before step 2)
        - patience = 10 (very large, would normally look back to step -6)
        - window_end = 4
        - window_start should be max(2.0, 4 - 10) = max(2.0, -6) = 2.0

        So the patience window should be [2, 3, 4], NOT [0, 1, 2, 3, 4].
        This ensures we only evaluate based on data
        from progressions >= min_progression.
        """
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=50,
            min_curves=4,
            min_progression=2.0,  # Don't trust data before step 2
            patience=10,  # Very large patience
        )

        data_lookup = none_throws(
            early_stopping_strategy._lookup_and_validate_data(exp, ["branin_map"])
        )
        aligned_df = align_partial_results(
            df=data_lookup.map_df, metrics=["branin_map"]
        )
        aligned_means = aligned_df["mean"]["branin_map"]

        should_stop, reason = early_stopping_strategy._should_stop_trial_early(
            trial_index=0,
            experiment=exp,
            wide_df=aligned_means,
            long_df=data_lookup.map_df,
            minimize=True,
        )

        # Trial 0 should be stopped (underperforms at all steps in window [2, 3, 4])
        self.assertTrue(should_stop)
        reason = none_throws(reason)
        # Verify the reason message shows window starting at min_progression (2.0)
        # not at window_end - patience (-6)
        self.assertRegex(
            reason,
            r"Trial objective values at progressions in \[2\.00, 4\.00\]",
        )

    def test_early_stopping_with_unaligned_results(self) -> None:
        # test case 1
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = assert_is_instance(exp.fetch_data(), MapData)

        unaligned_timestamps = [0, 1, 4, 1, 2, 3, 1, 3, 4, 0, 1, 2, 0, 2, 4]
        data.map_df.loc[data.map_df["metric_name"] == "branin_map", MAP_KEY] = (
            unaligned_timestamps
        )
        exp.attach_data(data=data)

        """
        Dataframe after interpolation:
                    0           1          2           3          4
        timestamp
        0          146.138620         NaN        NaN  143.375669  65.033535
        1          117.388086  113.057480  44.627226  115.168704  58.636359
        2          111.575393   90.815154  40.237365   98.060315  52.239184
        3          105.762700   77.324501  35.847504         NaN  48.359101
        4           99.950007         NaN  30.522333         NaN  44.479018
        """
        # We consider trials 0, 2, and 4 for early stopping at progression 4,
        #    and choose to stop trial 0.
        # We consider trial 1 for early stopping at progression 3, and
        #    choose to stop it.
        # We consider trial 3 for early stopping at progression 2, and
        #    choose to stop it.
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=["branin_map"],
            percentile_threshold=50,
            min_curves=3,
            min_progression=0.1,
        )
        should_stop = _evaluate_early_stopping_with_df(
            early_stopping_strategy=early_stopping_strategy,
            experiment=exp,
            metric_name="branin_map",
        )
        self.assertEqual(set(should_stop), {0, 1, 3})

        # test case 2, where trial 3 has only 1 data point
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)

        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = assert_is_instance(exp.fetch_data(), MapData)
        data.map_df.sort_values(by=["metric_name", "arm_name"], inplace=True)
        data.map_df.reset_index(drop=True, inplace=True)

        unaligned_timestamps = [0, 1, 4, 1, 2, 3, 1, 3, 4, 0, 1, 2, 0, 2, 4]
        data.map_df.loc[data.map_df["metric_name"] == "branin_map", MAP_KEY] = (
            unaligned_timestamps
        )
        # manually remove timestamps 1 and 2 for arm 3
        trial_3_data = next(iter(exp._data_by_trial[3].values()))
        trial_3_data.full_df = trial_3_data.full_df.loc[lambda x: x["step"] < 1]

        df = data.map_df
        df.drop(
            df.index[
                (df["metric_name"] == "branin_map")
                & (df["trial_index"] == 3)
                & (df[MAP_KEY].isin([1.0, 2.0]))
            ],
            inplace=True,
        )  # TODO this wont work once we make map_df immutable (which we should)
        # Create a new experiment without those
        exp.attach_data(data=data)

        """
        Dataframe after interpolation:
                    0           1          2           3          4
        timestamp
        0          146.138620         NaN        NaN  143.375669  65.033535
        1          117.388086  113.057480  44.627226         NaN  58.636359
        2          111.575393   90.815154  40.237365         NaN  52.239184
        3          105.762700   77.324501  35.847504         NaN  48.359101
        4           99.950007         NaN  30.522333         NaN  44.479018
        """

        # We consider trials 0, 2, and 4 for early stopping at progression 4,
        #    and choose to stop trial 0.
        # We consider trial 1 for early stopping at progression 3, and
        #    choose to stop it.
        # We consider trial 3 for early stopping at progression 0, and
        #    choose not to stop it.
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_signatures=["branin_map"],
            percentile_threshold=50,
            min_curves=3,
            min_progression=0.1,
        )
        should_stop = _evaluate_early_stopping_with_df(
            early_stopping_strategy=early_stopping_strategy,
            experiment=exp,
            metric_name="branin_map",
        )
        self.assertEqual(set(should_stop), {0, 1})

        # test error throwing in align partial results, with non-unique trial / arm name
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=2)

        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = assert_is_instance(exp.fetch_data(), MapData)
        df_with_single_arm_name = data.map_df.copy()
        df_with_single_arm_name["arm_name"] = "0_0"
        with self.assertRaisesRegex(
            UnsupportedError,
            "Arm 0_0 has multiple trial indices",
        ):
            align_partial_results(df=df_with_single_arm_name, metrics=["branin_map"])

        df_with_single_trial_index = data.map_df.copy()
        df_with_single_trial_index["trial_index"] = 0
        with self.assertRaisesRegex(
            UnsupportedError,
            "Trial 0 has multiple arm names",
        ):
            align_partial_results(df=df_with_single_trial_index, metrics=["branin_map"])


class TestThresholdEarlyStoppingStrategy(TestCase):
    # to avoid log spam in tests, we test the logger output explicitly in the percentile
    # early stopping strategy test
    @patch.object(logger, "warning")
    def test_threshold_early_stopping_strategy(self, _: MagicMock) -> None:
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        """
        Data looks like this:
        arm_name metric_name        mean  sem  trial_index  timestamp
        0       0_0      branin  146.138620  0.0            0          0
        1       0_0      branin  117.388086  0.0            0          1
        2       0_0      branin   99.950007  0.0            0          2
        3       1_0      branin  113.057480  0.0            1          0
        4       1_0      branin   90.815154  0.0            1          1
        5       1_0      branin   77.324501  0.0            1          2
        6       2_0      branin   44.627226  0.0            2          0
        7       2_0      branin   35.847504  0.0            2          1
        8       2_0      branin   30.522333  0.0            2          2
        9       3_0      branin  143.375669  0.0            3          0
        10      3_0      branin  115.168704  0.0            3          1
        11      3_0      branin   98.060315  0.0            3          2
        12      4_0      branin   65.033535  0.0            4          0
        13      4_0      branin   52.239184  0.0            4          1
        14      4_0      branin   44.479018  0.0            4          2
        """
        idcs = set(exp.trials.keys())

        early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=50, min_progression=1
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 1, 3})

        # respect trial_indices argument
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices={0}, experiment=exp
        )
        self.assertEqual(set(should_stop), {0})

        # test ignore trial indices
        early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=50,
            min_progression=1,
            trial_indices_to_ignore=[0],
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {1, 3})

        # test did not reach min progression
        early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=50, min_progression=3
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})


class TestLogicalEarlyStoppingStrategy(TestCase):
    @patch.object(logger, "warning")
    def test_and_early_stopping_strategy(self, _: MagicMock) -> None:
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        """
        Data looks like this:
        arm_name metric_name        mean  sem  trial_index  timestamp
        0       0_0      branin  146.138620  0.0            0          0
        1       0_0      branin  117.388086  0.0            0          1
        2       0_0      branin   99.950007  0.0            0          2
        3       1_0      branin  113.057480  0.0            1          0
        4       1_0      branin   90.815154  0.0            1          1
        5       1_0      branin   77.324501  0.0            1          2
        6       2_0      branin   44.627226  0.0            2          0
        7       2_0      branin   35.847504  0.0            2          1
        8       2_0      branin   30.522333  0.0            2          2
        9       3_0      branin  143.375669  0.0            3          0
        10      3_0      branin  115.168704  0.0            3          1
        11      3_0      branin   98.060315  0.0            3          2
        12      4_0      branin   65.033535  0.0            4          0
        13      4_0      branin   52.239184  0.0            4          1
        14      4_0      branin   44.479018  0.0            4          2
        """
        idcs = set(exp.trials.keys())

        left_early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=50, min_progression=1
        )

        right_early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=80, min_progression=1
        )

        and_early_stopping_strategy = AndEarlyStoppingStrategy(
            left=left_early_stopping_strategy, right=right_early_stopping_strategy
        )

        left_should_stop = left_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        right_should_stop = right_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        and_should_stop = and_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )

        intersection = set(left_should_stop.keys()).intersection(
            set(right_should_stop.keys())
        )

        for idc in idcs:
            if idc in intersection:
                self.assertIn(idc, and_should_stop.keys())
            else:
                self.assertNotIn(idc, and_should_stop.keys())

    @patch.object(logger, "warning")
    def test_or_early_stopping_strategy(self, _: MagicMock) -> None:
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        """
        Data looks like this:
        arm_name metric_name        mean  sem  trial_index  timestamp
        0       0_0      branin  146.138620  0.0            0          0
        1       0_0      branin  117.388086  0.0            0          1
        2       0_0      branin   99.950007  0.0            0          2
        3       1_0      branin  113.057480  0.0            1          0
        4       1_0      branin   90.815154  0.0            1          1
        5       1_0      branin   77.324501  0.0            1          2
        6       2_0      branin   44.627226  0.0            2          0
        7       2_0      branin   35.847504  0.0            2          1
        8       2_0      branin   30.522333  0.0            2          2
        9       3_0      branin  143.375669  0.0            3          0
        10      3_0      branin  115.168704  0.0            3          1
        11      3_0      branin   98.060315  0.0            3          2
        12      4_0      branin   65.033535  0.0            4          0
        13      4_0      branin   52.239184  0.0            4          1
        14      4_0      branin   44.479018  0.0            4          2
        """
        idcs = set(exp.trials.keys())

        left_early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=50, min_progression=1
        )

        right_early_stopping_strategy = ThresholdEarlyStoppingStrategy(
            metric_threshold=80, min_progression=1
        )

        or_early_stopping_strategy = OrEarlyStoppingStrategy(
            left=left_early_stopping_strategy, right=right_early_stopping_strategy
        )
        or_early_stopping_strategy_from_collection = (
            OrEarlyStoppingStrategy.from_early_stopping_strategies(
                strategies=[left_early_stopping_strategy, right_early_stopping_strategy]
            )
        )

        left_should_stop = left_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        right_should_stop = right_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        or_should_stop = or_early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        or_from_collection_should_stop = (
            or_early_stopping_strategy_from_collection.should_stop_trials_early(
                trial_indices=idcs, experiment=exp
            )
        )

        union = set(left_should_stop.keys()).union(set(right_should_stop.keys()))

        for idc in idcs:
            if idc in union:
                self.assertIn(idc, or_should_stop.keys())
                self.assertIn(idc, or_from_collection_should_stop.keys())
            else:
                self.assertNotIn(idc, or_should_stop.keys())
                self.assertNotIn(idc, or_from_collection_should_stop.keys())


def _evaluate_early_stopping_with_df(
    early_stopping_strategy: PercentileEarlyStoppingStrategy,
    experiment: Experiment,
    metric_name: str,
) -> dict[int, str | None]:
    """Helper function for testing PercentileEarlyStoppingStrategy
    on an arbitrary (MapData) df."""
    data = none_throws(
        early_stopping_strategy._lookup_and_validate_data(experiment, [metric_name])
    )
    aligned_df = align_partial_results(df=data.map_df, metrics=[metric_name])
    metric_to_aligned_means = aligned_df["mean"]
    aligned_means = metric_to_aligned_means[metric_name]
    decisions = {
        trial_index: early_stopping_strategy._should_stop_trial_early(
            trial_index=trial_index,
            experiment=experiment,
            wide_df=aligned_means,
            long_df=data.map_df,
            minimize=cast(
                OptimizationConfig, experiment.optimization_config
            ).objective.minimize,
        )
        for trial_index in set(experiment.trials.keys())
    }
    return {
        trial_index: reason
        for trial_index, (should_stop, reason) in decisions.items()
        if should_stop
    }
