#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.exceptions.core import UserInputError
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.model_spec import ModelSpec, FactoryFunctionModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ModelSpecTest(TestCase):
    def setUp(self) -> None:
        self.experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=self.experiment.search_space)
        sobol_run = sobol.gen(n=20)
        self.experiment.new_batch_trial().add_generator_run(
            sobol_run
        ).run().mark_completed()
        self.data = self.experiment.fetch_data()

    def test_construct(self):
        ms = ModelSpec(model_enum=Models.GPEI)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)
        with self.assertRaises(NotImplementedError):
            ms.update(experiment=self.experiment, new_data=self.data)

    def test_model_key(self):
        ms = ModelSpec(model_enum=Models.GPEI)
        self.assertEqual(ms.model_key, "GPEI")


class FactoryFunctionModelSpecTest(ModelSpecTest):
    def test_construct(self):
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)
        with self.assertRaises(NotImplementedError):
            ms.update(experiment=self.experiment, new_data=self.data)

    def test_model_key(self):
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        self.assertEqual(ms.model_key, "get_sobol")
