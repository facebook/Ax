#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from enum import Enum
from math import sqrt
from unittest import mock

import numpy as np
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.sklearn import (
    SklearnMetric,
    SklearnDataset,
    SklearnModelType,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial
from sklearn.ensemble import RandomForestClassifier


class DummyEnum(Enum):
    DUMMY: str = "dummy"


class SklearnMetricTest(TestCase):
    def testSklearnMetric(self):
        # test not implemented dataset
        with self.assertRaises(NotImplementedError):
            SklearnMetric(
                name="test_metric",
                dataset=DummyEnum.DUMMY,
            )
        # test not implemented model type
        with self.assertRaises(NotImplementedError):
            SklearnMetric(
                name="test_metric",
                model_type=DummyEnum.DUMMY,
            )
        # basic test
        data = {
            "data": np.random.random((5, 3)),
            "target": np.random.randint(0, 3, (5,)),
        }
        with ExitStack() as es:
            mock_load_digits = es.enter_context(
                mock.patch(
                    "ax.metrics.sklearn.datasets.load_digits",
                    return_value=data,
                )
            )
            cv_scores = np.random.random(5)
            mock_cv = es.enter_context(
                mock.patch(
                    "ax.metrics.sklearn.cross_val_score",
                    return_value=cv_scores,
                )
            )
            mock_rf = es.enter_context(
                mock.patch(
                    "ax.metrics.sklearn.RandomForestClassifier",
                    wraps=RandomForestClassifier,
                )
            )
            metric = SklearnMetric(
                name="test_metric",
            )
            self.assertIs(metric.dataset, SklearnDataset.DIGITS)
            self.assertIs(metric.model_type, SklearnModelType.RF)
            self.assertFalse(metric.lower_is_better)
            params = {"max_depth": 2, "min_samples_split": 0.5}
            trial = get_trial()
            trial._generator_run = GeneratorRun(
                arms=[Arm(name="0_0", parameters=params)]
            )
            df = metric.fetch_trial_data(trial).df
            mock_load_digits.assert_called_once()
            mock_cv.assert_called_once()
            cargs, ckwargs = mock_cv.call_args
            self.assertIsInstance(cargs[0], RandomForestClassifier)
            self.assertIs(cargs[1], data["data"])
            self.assertIs(cargs[2], data["target"])
            self.assertEqual(ckwargs["cv"], 5)
            _, ckwargs = mock_rf.call_args
            self.assertEqual(ckwargs, params)
            self.assertEqual(df["arm_name"].tolist(), ["0_0"])
            self.assertEqual(df["metric_name"].tolist(), ["test_metric"])
            self.assertEqual(df["mean"].tolist(), [cv_scores.mean()])
            self.assertTrue(np.isnan(df["sem"].values[0]))
            # test caching
            metric.fetch_trial_data(trial)
            mock_load_digits.assert_called_once()

            # test observed noise
            metric = SklearnMetric(name="test_metric", observed_noise=True)
            df = metric.fetch_trial_data(trial).df
            self.assertEqual(
                df["sem"].values[0], cv_scores.std() / sqrt(cv_scores.shape[0])
            )
            # test num_folds
            mock_cv.reset_mock()
            metric = SklearnMetric(name="test_metric", num_folds=10)
            df = metric.fetch_trial_data(trial).df
            _, ckwargs = mock_cv.call_args
            self.assertEqual(ckwargs["cv"], 10)
