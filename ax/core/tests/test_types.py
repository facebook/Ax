#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.types import (
    merge_model_predict,
    validate_evaluation_outcome,
    validate_floatlike,
    validate_map_dict,
    validate_param_value,
    validate_parameterization,
    validate_single_metric_data,
    validate_trial_evaluation,
)
from ax.utils.common.testutils import TestCase


class TypesTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.num_arms = 2
        mu = {"m1": [0.0, 0.5], "m2": [0.1, 0.6]}
        cov = {
            "m1": {"m1": [0.0, 0.0], "m2": [0.0, 0.0]},
            "m2": {"m1": [0.0, 0.0], "m2": [0.0, 0.0]},
        }
        self.predict = (mu, cov)

    def test_MergeModelPredict(self) -> None:
        mu_append = {"m1": [0.6], "m2": [0.7]}
        cov_append = {
            "m1": {"m1": [0.0], "m2": [0.0]},
            "m2": {"m1": [0.0], "m2": [0.0]},
        }
        merged_predicts = merge_model_predict(self.predict, (mu_append, cov_append))
        self.assertEqual(len(merged_predicts[0]["m1"]), 3)

    def test_MergeModelPredictFail(self) -> None:
        mu_append = {"m1": [0.6]}
        cov_append = {
            "m1": {"m1": [0.0], "m2": [0.0]},
            "m2": {"m1": [0.0], "m2": [0.0]},
        }
        with self.assertRaises(ValueError):
            merge_model_predict(self.predict, (mu_append, cov_append))

        mu_append = {"m1": [0.6], "m2": [0.7]}
        cov_append = {"m1": {"m1": [0.0], "m2": [0.0]}}
        with self.assertRaises(ValueError):
            merge_model_predict(self.predict, (mu_append, cov_append))

    def test_Validate(self) -> None:
        trial_evaluation = {"foo": 0.0}
        trial_evaluation_with_noise = {"foo": (0.0, 0.0)}
        fidelity_trial_evaluation = [({"a": 0.0}, trial_evaluation)]
        map_trial_evaluation = [({"a": 0.0}, trial_evaluation)]

        validate_evaluation_outcome(outcome=trial_evaluation)  # pyre-ignore[6]
        validate_evaluation_outcome(
            outcome=trial_evaluation_with_noise  # pyre-ignore[6]
        )
        validate_evaluation_outcome(outcome=fidelity_trial_evaluation)  # pyre-ignore[6]
        validate_evaluation_outcome(outcome=map_trial_evaluation)  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_floatlike(floatlike="foo")  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_single_metric_data(data=(0, 1, 2))  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_trial_evaluation(evaluation={0: 0})  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_param_value(param_value=[])  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_parameterization(parameterization={0: 0})  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_map_dict(map_dict={0: 0})  # pyre-ignore[6]

        with self.assertRaises(TypeError):
            validate_map_dict(map_dict={"foo": []})  # pyre-ignore[6]
