#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
import torch
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.numpy_base import NumpyModel
from ax.models.torch.botorch import BotorchModel
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class TransformCallbackTest(TestCase):
    @patch("ax.modelbridge.torch.TorchModelBridge._model_fit", return_value=None)
    def test_transform_callback_int(self, _):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.INT, lower=1, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.INT, lower=5, upper=15
            ),
        ]
        gpei = TorchModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=BotorchModel(),
            transforms=[IntToFloat],
            torch_dtype=torch.double,
        )
        transformed = gpei._transform_callback([5.4, 7.6])
        self.assertTrue(np.allclose(transformed, [5, 8]))
        np_mb = ArrayModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=NumpyModel(),
            transforms=[IntToFloat],
        )
        transformed = np_mb._transform_callback(np.array([5.4, 7.6]))
        self.assertTrue(np.allclose(transformed, [5, 8]))

    @patch("ax.modelbridge.torch.TorchModelBridge._model_fit", return_value=None)
    def test_transform_callback_log(self, _):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                lower=1,
                upper=3,
                log_scale=True,
            ),
            RangeParameter(
                name="x2",
                parameter_type=ParameterType.FLOAT,
                lower=1,
                upper=3,
                log_scale=True,
            ),
        ]
        gpei = TorchModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=BotorchModel(),
            transforms=[Log],
            torch_dtype=torch.double,
        )
        transformed = gpei._transform_callback([1.2, 2.5])
        self.assertTrue(np.allclose(transformed, [1.2, 2.5]))

    @patch("ax.modelbridge.torch.TorchModelBridge._model_fit", return_value=None)
    def test_transform_callback_unitx(self, _):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.FLOAT, lower=0, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=100
            ),
        ]
        gpei = TorchModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=BotorchModel(),
            transforms=[UnitX],
        )
        transformed = gpei._transform_callback([0.75, 0.35])
        self.assertTrue(np.allclose(transformed, [0.75, 0.35]))

    @patch("ax.modelbridge.torch.TorchModelBridge._model_fit", return_value=None)
    def test_transform_callback_int_log(self, _):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=100,
                log_scale=True,
            ),
            RangeParameter(
                name="x2",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=100,
                log_scale=True,
            ),
        ]
        gpei = TorchModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=BotorchModel(),
            transforms=[IntToFloat, Log],
            torch_dtype=torch.double,
        )
        transformed = gpei._transform_callback([0.5, 1.5])
        self.assertTrue(np.allclose(transformed, [0.47712, 1.50515]))

    @patch("ax.modelbridge.torch.TorchModelBridge._model_fit", return_value=None)
    def test_transform_callback_int_unitx(self, _):
        exp = get_branin_experiment()
        parameters = [
            RangeParameter(
                name="x1", parameter_type=ParameterType.INT, lower=0, upper=10
            ),
            RangeParameter(
                name="x2", parameter_type=ParameterType.INT, lower=0, upper=100
            ),
        ]
        gpei = TorchModelBridge(
            experiment=exp,
            data=exp.fetch_data(),
            search_space=SearchSpace(parameters=parameters),
            model=BotorchModel(),
            transforms=[IntToFloat, UnitX],
            torch_dtype=torch.double,
        )
        transformed = gpei._transform_callback([0.75, 0.35])
        self.assertTrue(np.allclose(transformed, [0.8, 0.35]))
