#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.numpy_base import NumpyModel
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_search_space_for_range_value,
)
from ax.utils.testing.modeling_stubs import get_observation1


# Prepare mock transforms
class t1(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._lower += 1.0
        new_ss.parameters["x"]._upper += 1.0
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config + 1
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            if "x" in obsf.parameters:
                obsf.parameters["x"] += 1
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means += 1
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            obsf.parameters["x"] -= 1
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means -= 1
        return x


class t2(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._lower *= 2
        new_ss.parameters["x"]._upper *= 2
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config ** 2
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            if "x" in obsf.parameters:
                obsf.parameters["x"] = obsf.parameters["x"] ** 2
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = obsd.means ** 2
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            obsf.parameters["x"] = np.sqrt(obsf.parameters["x"])
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = np.sqrt(obsd.means)
        return x


class ArrayModelBridgeTest(TestCase):
    @patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=([Arm(parameters={})], {}),
    )
    @patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(f"{ModelBridge.__module__}.ModelBridge._fit", autospec=True)
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.best_point",
        return_value=(np.array([1, 2])),
        autospec=True,
    )
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.gen",
        return_value=(np.array([[1, 2]]), np.array([1]), {}, []),
        autospec=True,
    )
    def test_best_point(
        self,
        _mock_gen,
        _mock_best_point,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
        exp = Experiment(get_search_space_for_range_value(), "test")
        modelbridge = ArrayModelBridge(
            get_search_space_for_range_value(), NumpyModel(), [t1, t2], exp, 0
        )
        self.assertEqual(list(modelbridge.transforms.keys()), ["Cast", "t1", "t2"])
        # _fit is mocked, which typically sets this.
        modelbridge.outcomes = ["a"]
        run = modelbridge.gen(
            n=1,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric("a"), minimize=False),
                outcome_constraints=[],
            ),
        )
        arm, predictions = run.best_arm_predictions
        self.assertEqual(arm.parameters, {})
        self.assertEqual(predictions[0], {"m": 1.0})
        self.assertEqual(predictions[1], {"m": {"m": 2.0}})
        # test check that optimization config is required
        with self.assertRaises(ValueError):
            run = modelbridge.gen(n=1, optimization_config=None)

    @patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=[Arm(parameters={})],
    )
    @patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(f"{ModelBridge.__module__}.ModelBridge._fit", autospec=True)
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.feature_importances",
        return_value=np.array([[[1.0]], [[2.0]]]),
        autospec=True,
    )
    def test_importances(
        self,
        _mock_feature_importances,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
        exp = Experiment(get_search_space_for_range_value(), "test")
        modelbridge = ArrayModelBridge(
            get_search_space_for_range_value(), NumpyModel(), [t1, t2], exp, 0
        )
        modelbridge.outcomes = ["a", "b"]
        self.assertEqual(modelbridge.feature_importances("a"), {"x": [1.0]})
        self.assertEqual(modelbridge.feature_importances("b"), {"x": [2.0]})

    @patch(
        f"{NumpyModel.__module__}.NumpyModel.gen",
        return_value=(
            np.array([[1, 2], [2, 3]]),
            np.array([1, 2]),
            {},
            [{"some_key": "some_value_0"}, {"some_key": "some_value_1"}],
        ),
        autospec=True,
    )
    @patch(f"{NumpyModel.__module__}.NumpyModel.update", autospec=True)
    @patch(f"{NumpyModel.__module__}.NumpyModel.fit", autospec=True)
    def test_candidate_metadata_propagation(
        self, mock_model_fit, mock_model_update, mock_model_gen
    ):
        exp = get_branin_experiment(with_status_quo=True, with_batch=True)
        # Check that the metadata is correctly re-added to observation
        # features during `fit`.
        preexisting_batch_gr = exp.trials[0]._generator_run_structs[0].generator_run
        preexisting_batch_gr._candidate_metadata_by_arm_signature = {
            preexisting_batch_gr.arms[0].signature: {
                "preexisting_batch_cand_metadata": "some_value"
            }
        }
        modelbridge = ArrayModelBridge(
            search_space=exp.search_space,
            experiment=exp,
            model=NumpyModel(),
            data=get_branin_data(),
        )
        self.assertTrue(
            np.array_equal(
                mock_model_fit.call_args[1].get("Xs"),
                np.array([[list(exp.trials[0].arms[0].parameters.values())]]),
            )
        )
        self.assertEqual(
            mock_model_fit.call_args[1].get("candidate_metadata"),
            [[{"preexisting_batch_cand_metadata": "some_value"}]],
        )

        # Check that `gen` correctly propagates the metadata to the GR.
        gr = modelbridge.gen(n=1)
        self.assertEqual(
            gr.candidate_metadata_by_arm_signature,
            {
                gr.arms[0].signature: {"some_key": "some_value_0"},
                gr.arms[1].signature: {"some_key": "some_value_1"},
            },
        )
        # Check that the metadata is correctly re-added to observation
        # features during `update`.
        batch = exp.new_batch_trial(gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
        self.assertTrue(
            np.array_equal(
                mock_model_update.call_args[1].get("Xs"),
                np.array([[list(exp.trials[0].arms[0].parameters.values()), [1, 2]]]),
            )
        )
        self.assertEqual(
            mock_model_update.call_args[1].get("candidate_metadata"),
            [
                [
                    {"preexisting_batch_cand_metadata": "some_value"},
                    # new data contained data just for arm '1_0', not for '1_1',
                    # so we don't expect to see '{"some_key": "some_value_1"}'
                    # in candidate metadata.
                    {"some_key": "some_value_0"},
                ]
            ],
        )

        # Check that `None` candidate metadata is handled correctly.
        mock_model_gen.return_value = (
            np.array([[2, 4], [3, 5]]),
            np.array([1, 2]),
            None,
            {},
        )
        gr = modelbridge.gen(n=1)
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)
        # Check that the metadata is correctly re-added to observation
        # features during `update`.
        batch = exp.new_batch_trial(gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
        self.assertTrue(
            np.array_equal(
                mock_model_update.call_args[1].get("Xs"),
                np.array(
                    [[list(exp.trials[0].arms[0].parameters.values()), [1, 2], [2, 4]]]
                ),
            )
        )
        self.assertEqual(
            mock_model_update.call_args[1].get("candidate_metadata"),
            [
                [
                    {"preexisting_batch_cand_metadata": "some_value"},
                    {"some_key": "some_value_0"},
                    None,
                ]
            ],
        )

        # Check that no candidate metadata is handled correctly.
        exp = get_branin_experiment(with_status_quo=True)
        modelbridge = ArrayModelBridge(
            search_space=exp.search_space, experiment=exp, model=NumpyModel()
        )
        # Hack in outcome names to bypass validation (since we instantiated model
        # without data).
        modelbridge.outcomes = modelbridge._metric_names = next(iter(exp.metrics))
        gr = modelbridge.gen(n=1)
        self.assertIsNone(mock_model_fit.call_args[1].get("candidate_metadata"))
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)
        batch = exp.new_batch_trial(gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
        self.assertIsNone(mock_model_update.call_args[1].get("candidate_metadata"))
