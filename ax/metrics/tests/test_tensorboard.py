#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, Union
from unittest import mock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.tensorboard import TensorboardCurveMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space


class TensorboardCurveMetricTest(TestCase):
    def test_tensorboard_curve_metric(self) -> None:
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def mock_get_tb_from_posix(path):
            data = np.array([10, 3, 5, 2, 7, 1])
            return {"test_curve": pd.Series((int(path) + 1) * data)}

        mock_path = "ax.metrics.tensorboard.get_tb_from_posix"

        class FakeTensorboardCurveMetric(TensorboardCurveMetric):
            @classmethod
            def get_ids_from_trials(
                cls, trials: Iterable[BaseTrial]
            ) -> Dict[int, Union[int, str]]:
                result = {}
                for trial in trials:
                    result[trial.index] = trial.index
                return result

        with mock.patch(mock_path, side_effect=mock_get_tb_from_posix):
            # test simple
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=False,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    # pyre-fixme[16]: `Data` has no attribute `map_df`.
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [10.0, 3.0, 5.0, 2.0, 7.0, 1.0, 20.0, 6.0, 10.0, 4.0, 14.0, 2.0]
                    ),
                )
            )

            # test cumulative best
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=True,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [10.0, 3.0, 3.0, 2.0, 2.0, 1.0, 20.0, 6.0, 6.0, 4.0, 4.0, 2.0]
                    ),
                )
            )

            # test cumulative best (lower is worse)
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=False,
                            cumulative_best=True,
                        ),
                        minimize=False,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                        ]
                    ),
                ),
            )

            # test smoothing
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=False,
                            smoothing_window=3,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()
            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [
                            6.00000000,
                            6.00000000,
                            6.00000000,
                            3.33333333,
                            4.66666667,
                            3.33333333,
                            12.0,
                            12.0,
                            12.0,
                            6.66666667,
                            9.33333333,
                            6.66666667,
                        ]
                    ),
                )
            )
