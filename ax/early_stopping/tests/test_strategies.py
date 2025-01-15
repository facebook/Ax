#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy
from typing import Any, cast

import numpy as np
from ax.core import OptimizationConfig
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import MultiObjective
from ax.early_stopping.strategies import (
    BaseEarlyStoppingStrategy,
    ModelBasedEarlyStoppingStrategy,
    PercentileEarlyStoppingStrategy,
    ThresholdEarlyStoppingStrategy,
)
from ax.early_stopping.strategies.logical import (
    AndEarlyStoppingStrategy,
    OrEarlyStoppingStrategy,
)
from ax.early_stopping.utils import align_partial_results
from ax.exceptions.core import UnsupportedError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_arms,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
    get_experiment_with_multi_objective,
    get_test_map_data_experiment,
)
from pyre_extensions import assert_is_instance, none_throws


class TestBaseEarlyStoppingStrategy(TestCase):
    def test_early_stopping_strategy(self) -> None:
        # can't instantiate abstract class
        with self.assertRaises(TypeError):
            # pyre-fixme[45]: Cannot instantiate abstract class
            #  `BaseEarlyStoppingStrategy`.
            BaseEarlyStoppingStrategy()

    def test_default_objective_and_direction(self) -> None:
        class FakeStrategy(BaseEarlyStoppingStrategy):
            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, Any],
            ) -> dict[int, str | None]:
                return {}

        test_experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        test_objective = none_throws(test_experiment.optimization_config).objective
        with self.subTest("provide metric names"):
            es_strategy = FakeStrategy(metric_names=[test_objective.metric.name])
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
                metric_names=[test_multi_objective.objectives[1].metric.name]
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

    def test_is_eligible(self) -> None:
        class FakeStrategy(BaseEarlyStoppingStrategy):
            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, Any],
            ) -> dict[int, str | None]:
                return {}

        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=5, num_complete=3
        )
        es_strategy = FakeStrategy(min_progression=3, max_progression=5)
        metric_name, _ = es_strategy._default_objective_and_direction(
            experiment=experiment
        )

        map_data = es_strategy._check_validity_and_get_data(
            experiment,
            metric_names=[metric_name],
        )
        map_data = assert_is_instance(map_data, MapData)
        self.assertTrue(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
                map_key=map_data.map_keys[0],
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
            map_key=map_data.map_keys[0],
        )
        self.assertFalse(fake_es)
        self.assertEqual(
            fake_reason, "No data available to make an early stopping decision."
        )

        fake_map_data = es_strategy._check_validity_and_get_data(
            experiment,
            metric_names=["fake_metric_name"],
        )
        self.assertIsNone(fake_map_data)

        es_strategy = FakeStrategy(min_progression=5)
        self.assertFalse(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
                map_key=map_data.map_keys[0],
            )[0]
        )

        es_strategy = FakeStrategy(min_progression=2, max_progression=3)
        self.assertFalse(
            es_strategy.is_eligible(
                trial_index=0,
                experiment=experiment,
                df=map_data.map_df,
                map_key=map_data.map_keys[0],
            )[0]
        )

    def test_early_stopping_savings(self) -> None:
        class FakeStrategy(BaseEarlyStoppingStrategy):
            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, Any],
            ) -> dict[int, str | None]:
                return {}

        exp = get_branin_experiment_with_timestamp_map_metric()
        es_strategy = FakeStrategy(min_progression=3, max_progression=5)

        self.assertEqual(
            es_strategy.estimate_early_stopping_savings(
                experiment=exp,
            ),
            0,
        )


class TestModelBasedEarlyStoppingStrategy(TestCase):
    def test_get_training_data(self) -> None:
        class FakeStrategy(ModelBasedEarlyStoppingStrategy):
            def should_stop_trials_early(
                self,
                trial_indices: set[int],
                experiment: Experiment,
                **kwargs: dict[str, Any],
            ) -> dict[int, str | None]:
                return {}

        experiment = get_test_map_data_experiment(
            num_trials=3, num_fetches=2, num_complete=3
        )
        training_data = FakeStrategy().get_training_data(
            experiment,
            map_data=cast(MapData, experiment.lookup_data()),
        )
        # check that there is a map dimension in the training data
        X = training_data.X
        self.assertEqual(X.shape[-1], 3)
        # check that the default Ax transform is applied, i.e., that the
        # parameters are normalized to [0, 1]
        self.assertTrue(np.all((X[:, :2] >= 0.0) & (X[:, :2] <= 1.0)))


class TestPercentileEarlyStoppingStrategy(TestCase):
    def test_percentile_early_stopping_strategy_validation(self) -> None:
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
        self._test_percentile_early_stopping_strategy(non_objective_metric=False)

    def test_percentile_early_stopping_strategy_non_objective_metric(self) -> None:
        self._test_percentile_early_stopping_strategy(non_objective_metric=True)

        with self.assertRaisesRegex(
            UnsupportedError,
            "PercentileEarlyStoppingStrategy only supports a single metric.",
        ):
            PercentileEarlyStoppingStrategy(
                metric_names=["tracking_branin_map", "foo"],
                percentile_threshold=75,
                min_curves=5,
                min_progression=0.1,
            )

    def _test_percentile_early_stopping_strategy(
        self, non_objective_metric: bool
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
            metric_names = ["tracking_branin_map"]
            # remove the optimization config to force that only the tracking metric can
            # be used for early stopping
            exp._optimization_config = None
            data = assert_is_instance(exp.fetch_data(), MapData)
            self.assertTrue((data.map_df["metric_name"] == "tracking_branin_map").all())
        else:
            metric_names = None

        idcs = set(exp.trials.keys())

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_names=metric_names,
            percentile_threshold=25,
            min_curves=4,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0})

        # test ignore trial indices
        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            metric_names=metric_names,
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
            metric_names=metric_names,
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
            metric_names=metric_names,
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
            metric_names=metric_names,
            percentile_threshold=75,
            min_curves=5,
            min_progression=0.1,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(should_stop, {})

    def test_early_stopping_with_unaligned_results(self) -> None:
        # test case 1
        exp = get_test_map_data_experiment(num_trials=5, num_fetches=3, num_complete=5)
        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = assert_is_instance(exp.fetch_data(), MapData)

        unaligned_timestamps = [0, 1, 4, 1, 2, 3, 1, 3, 4, 0, 1, 2, 0, 2, 4]
        data.map_df.loc[data.map_df["metric_name"] == "branin_map", "timestamp"] = (
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
        data.map_df.loc[data.map_df["metric_name"] == "branin_map", "timestamp"] = (
            unaligned_timestamps
        )
        # manually remove timestamps 1 and 2 for arm 3
        df = data.map_df
        df.drop(
            df.index[
                (df["metric_name"] == "branin_map")
                & (df["trial_index"] == 3)
                & (df["timestamp"].isin([1.0, 2.0]))
            ],
            inplace=True,
        )  # TODO this wont work once we make map_df immutable (which we should)
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


class TestThresholdEarlyStoppingStrategy(TestCase):
    def test_threshold_early_stopping_strategy(self) -> None:
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
    def test_and_early_stopping_strategy(self) -> None:
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

    def test_or_early_stopping_strategy(self) -> None:
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
        early_stopping_strategy._check_validity_and_get_data(experiment, [metric_name])
    )
    metric_to_aligned_means, _ = align_partial_results(
        df=data.map_df,
        progr_key="timestamp",
        metrics=[metric_name],
    )
    aligned_means = metric_to_aligned_means[metric_name]
    decisions = {
        trial_index: early_stopping_strategy._should_stop_trial_early(
            trial_index=trial_index,
            experiment=experiment,
            df=aligned_means,
            df_raw=data.map_df,
            map_key="timestamp",
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
