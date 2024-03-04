# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.storage.botorch_modular_registry import (
    ACQUISITION_FUNCTION_REGISTRY,
    ACQUISITION_REGISTRY,
    MODEL_REGISTRY,
    register_acquisition,
    register_acquisition_function,
    register_model,
    REVERSE_ACQUISITION_FUNCTION_REGISTRY,
    REVERSE_ACQUISITION_REGISTRY,
    REVERSE_MODEL_REGISTRY,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model


class NewModel(Model):
    pass


class NewAcquisition(Acquisition):
    pass


class NewAcquisitionFunction(AcquisitionFunction):
    pass


class RegisterNewClassTest(TestCase):
    def test_register_model(self) -> None:
        self.assertNotIn(NewModel, MODEL_REGISTRY)
        self.assertNotIn(NewModel, REVERSE_MODEL_REGISTRY.values())
        register_model(NewModel)
        self.assertIn(NewModel, MODEL_REGISTRY)
        self.assertIn(NewModel, REVERSE_MODEL_REGISTRY.values())

    def test_register_acquisition(self) -> None:
        self.assertNotIn(NewAcquisition, ACQUISITION_REGISTRY)
        self.assertNotIn(NewAcquisition, REVERSE_ACQUISITION_REGISTRY.values())
        register_acquisition(NewAcquisition)
        self.assertIn(NewAcquisition, ACQUISITION_REGISTRY)
        self.assertIn(NewAcquisition, REVERSE_ACQUISITION_REGISTRY.values())

    def test_register_acquisition_function(self) -> None:
        self.assertNotIn(NewAcquisitionFunction, ACQUISITION_FUNCTION_REGISTRY)
        self.assertNotIn(
            NewAcquisitionFunction, REVERSE_ACQUISITION_FUNCTION_REGISTRY.values()
        )
        register_acquisition_function(NewAcquisitionFunction)
        self.assertIn(NewAcquisitionFunction, ACQUISITION_FUNCTION_REGISTRY)
        self.assertIn(
            NewAcquisitionFunction, REVERSE_ACQUISITION_FUNCTION_REGISTRY.values()
        )
