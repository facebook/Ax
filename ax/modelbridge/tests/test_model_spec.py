#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest.mock import Mock, patch

from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.model_spec import FactoryFunctionModelSpec, ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class BaseModelSpecTest(TestCase):
    def setUp(self) -> None:
        self.experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=self.experiment.search_space)
        sobol_run = sobol.gen(n=20)
        self.experiment.new_batch_trial().add_generator_run(
            sobol_run
        ).run().mark_completed()
        self.data = self.experiment.fetch_data()


class ModelSpecTest(BaseModelSpecTest):
    @fast_botorch_optimize
    def test_construct(self) -> None:
        ms = ModelSpec(model_enum=Models.GPEI)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)
        with self.assertRaises(NotImplementedError):
            ms.update(experiment=self.experiment, new_data=self.data)

    def test_model_key(self) -> None:
        ms = ModelSpec(model_enum=Models.GPEI)
        self.assertEqual(ms.model_key, "GPEI")

    @patch(f"{ModelSpec.__module__}.compute_diagnostics")
    @patch(f"{ModelSpec.__module__}.cross_validate", return_value=["fake-cv-result"])
    # pyre-fixme[3]: Return type must be annotated.
    def test_cross_validate_with_GP_model(self, mock_cv: Mock, mock_diagnostics: Mock):
        mock_enum = Mock()
        mock_enum.return_value = "fake-modelbridge"
        ms = ModelSpec(model_enum=mock_enum, model_cv_kwargs={"test_key": "test-value"})
        ms.fit(
            experiment=self.experiment,
            data=self.experiment.trials[0].fetch_data(),
        )
        cv_results, cv_diagnostics = ms.cross_validate()
        mock_cv.assert_called_with(model="fake-modelbridge", test_key="test-value")
        mock_diagnostics.assert_called_with(["fake-cv-result"])

        self.assertIsNotNone(cv_results)
        self.assertIsNotNone(cv_diagnostics)

        with self.subTest("it caches CV results"):
            mock_cv.reset_mock()
            mock_diagnostics.reset_mock()

            cv_results, cv_diagnostics = ms.cross_validate()

            self.assertIsNotNone(cv_results)
            self.assertIsNotNone(cv_diagnostics)
            mock_cv.assert_not_called()
            mock_diagnostics.assert_not_called()

        with self.subTest("fit clears the CV cache"):
            mock_cv.reset_mock()
            mock_diagnostics.reset_mock()

            ms.fit(
                experiment=self.experiment,
                data=self.experiment.trials[0].fetch_data(),
            )
            cv_results, cv_diagnostics = ms.cross_validate()

            self.assertIsNotNone(cv_results)
            self.assertIsNotNone(cv_diagnostics)
            mock_cv.assert_called_with(model="fake-modelbridge", test_key="test-value")
            mock_diagnostics.assert_called_with(["fake-cv-result"])

    @patch(f"{ModelSpec.__module__}.compute_diagnostics")
    @patch(f"{ModelSpec.__module__}.cross_validate", side_effect=NotImplementedError)
    # pyre-fixme[3]: Return type must be annotated.
    def test_cross_validate_with_non_GP_model(
        self, mock_cv: Mock, mock_diagnostics: Mock
    ):
        mock_enum = Mock()
        mock_enum.return_value = "fake-modelbridge"
        ms = ModelSpec(model_enum=mock_enum, model_cv_kwargs={"test_key": "test-value"})
        ms.fit(
            experiment=self.experiment,
            data=self.experiment.trials[0].fetch_data(),
        )
        with warnings.catch_warnings(record=True) as w:
            cv_results, cv_diagnostics = ms.cross_validate()

        self.assertEqual(len(w), 1)
        self.assertIn("cannot be cross validated", str(w[0].message))
        self.assertIsNone(cv_results)
        self.assertIsNone(cv_diagnostics)

        mock_cv.assert_called_with(model="fake-modelbridge", test_key="test-value")
        mock_diagnostics.assert_not_called()

    def test_fixed_features(self) -> None:
        ms = ModelSpec(model_enum=Models.GPEI)
        self.assertIsNone(ms.fixed_features)
        new_features = ObservationFeatures(parameters={"a": 1.0})
        ms.fixed_features = new_features
        self.assertEqual(ms.fixed_features, new_features)
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertEqual(ms.model_gen_kwargs["fixed_features"], new_features)


class FactoryFunctionModelSpecTest(BaseModelSpecTest):
    def test_construct(self) -> None:
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)
        with self.assertRaises(NotImplementedError):
            ms.update(experiment=self.experiment, new_data=self.data)

    def test_model_key(self) -> None:
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        self.assertEqual(ms.model_key, "get_sobol")
