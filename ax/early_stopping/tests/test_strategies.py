#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.early_stopping.strategies import (
    BaseEarlyStoppingStrategy,
    PercentileEarlyStoppingStrategy,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_arms,
    get_branin_experiment,
    get_branin_experiment_with_timestamp_map_metric,
)


class TestEarlyStoppingStrategy(TestCase):
    def test_early_stopping_strategy(self):
        # can't instantiate abstract class
        with self.assertRaises(TypeError):
            BaseEarlyStoppingStrategy()

    def test_percentile_early_stopping_strategy_validation(self):
        exp = get_branin_experiment()

        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        early_stopping_strategy = PercentileEarlyStoppingStrategy()
        idcs = set(exp.trials.keys())
        exp.attach_data(data=exp.fetch_data())

        # Non-MapData attached
        with self.assertRaisesRegex(ValueError, "expects MapData"):
            early_stopping_strategy.should_stop_trials_early(
                trial_indices=idcs, experiment=exp
            )

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

    def test_percentile_early_stopping_strategy(self):
        exp = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        exp.attach_data(data=exp.fetch_data())

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
        idcs = set(exp.trials.keys())

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=25,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0})

        early_stopping_strategy = PercentileEarlyStoppingStrategy(
            percentile_threshold=50,
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
            percentile_threshold=75,
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=idcs, experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 3, 1})

    def test_early_stopping_with_unaligned_results(self):
        # test case 1
        exp = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = exp.fetch_data()
        unaligned_timestamps = [0, 1, 4, 1, 2, 3, 1, 3, 4, 0, 1, 2, 0, 2, 4]
        data.df.loc[
            data.df["metric_name"] == "branin", "timestamp"
        ] = unaligned_timestamps
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
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=set(exp.trials.keys()), experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 1, 3})

        # test case 2, where trial 3 has only 1 data point
        exp = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
        for i in range(5):
            trial = exp.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        # manually "unalign" timestamps to simulate real-world scenario
        # where each curve reports results at different steps
        data = exp.fetch_data()
        unaligned_timestamps = [0, 1, 4, 1, 2, 3, 1, 3, 4, 0, 1, 2, 0, 2, 4]
        data.df.loc[
            data.df["metric_name"] == "branin", "timestamp"
        ] = unaligned_timestamps
        # manually remove timestamps 1 and 2 for arm 3
        data.df.drop([22, 23], inplace=True)
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
        )
        should_stop = early_stopping_strategy.should_stop_trials_early(
            trial_indices=set(exp.trials.keys()), experiment=exp
        )
        self.assertEqual(set(should_stop), {0, 1})
