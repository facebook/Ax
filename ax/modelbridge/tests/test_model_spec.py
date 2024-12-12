#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from unittest import mock
from unittest.mock import MagicMock, Mock, patch

from ax.core.observation import ObservationFeatures
from ax.exceptions.core import UserInputError
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.model_spec import FactoryFunctionModelSpec, ModelSpec
from ax.modelbridge.modelbridge_utils import extract_search_space_digest
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import none_throws


class BaseModelSpecTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.experiment = get_branin_experiment()
        sobol = Models.SOBOL(search_space=self.experiment.search_space)
        sobol_run = sobol.gen(n=20)
        self.experiment.new_batch_trial().add_generator_run(
            sobol_run
        ).run().mark_completed()
        self.data = self.experiment.fetch_data()


class ModelSpecTest(BaseModelSpecTest):
    @mock_botorch_optimize
    def test_construct(self) -> None:
        ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)

    @mock_botorch_optimize
    # We can use `extract_search_space_digest` as a surrogate for executing
    # the full TorchModelBridge._fit.
    @mock.patch(
        "ax.modelbridge.torch.extract_search_space_digest",
        wraps=extract_search_space_digest,
    )
    def test_fit(self, wrapped_extract_ssd: Mock) -> None:
        ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        # This should fit the model as usual.
        ms.fit(experiment=self.experiment, data=self.data)
        wrapped_extract_ssd.assert_called_once()
        self.assertIsNotNone(ms._last_fit_arg_ids)
        self.assertEqual(ms._last_fit_arg_ids["experiment"], id(self.experiment))
        # This should skip the model fit.
        with mock.patch("ax.modelbridge.torch.logger") as mock_logger:
            ms.fit(experiment=self.experiment, data=self.data)
        mock_logger.debug.assert_called_with(
            "The observations are identical to the last set of observations "
            "used to fit the model. Skipping model fitting."
        )
        wrapped_extract_ssd.assert_called_once()

    def test_model_key(self) -> None:
        ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        self.assertEqual(ms.model_key, "BoTorch")
        ms = ModelSpec(
            model_enum=Models.BOTORCH_MODULAR, model_key_override="MBM with defaults"
        )
        self.assertEqual(ms.model_key, "MBM with defaults")

    @patch(f"{ModelSpec.__module__}.compute_diagnostics")
    @patch(f"{ModelSpec.__module__}.cross_validate", return_value=["fake-cv-result"])
    def test_cross_validate_with_GP_model(
        self, mock_cv: Mock, mock_diagnostics: Mock
    ) -> None:
        mock_enum = Mock()
        fake_mb = MagicMock()
        fake_mb._process_and_transform_data = MagicMock(return_value=(None, None))
        mock_enum.return_value = fake_mb
        ms = ModelSpec(model_enum=mock_enum, model_cv_kwargs={"test_key": "test-value"})
        ms.fit(
            experiment=self.experiment,
            data=self.experiment.trials[0].fetch_data(),
        )
        cv_results, cv_diagnostics = ms.cross_validate()
        mock_cv.assert_called_with(model=fake_mb, test_key="test-value")
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
            self.assertEqual(ms._last_cv_kwargs, {"test_key": "test-value"})

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
            mock_cv.assert_called_with(model=fake_mb, test_key="test-value")
            mock_diagnostics.assert_called_with(["fake-cv-result"])

        with self.subTest("pass in optional kwargs"):
            mock_cv.reset_mock()
            mock_diagnostics.reset_mock()
            # Cache is not empty, but CV will be called since there are new kwargs.
            assert ms._cv_results is not None

            cv_results, cv_diagnostics = ms.cross_validate(model_cv_kwargs={"test": 1})

            self.assertIsNotNone(cv_results)
            self.assertIsNotNone(cv_diagnostics)
            mock_cv.assert_called_with(model=fake_mb, test_key="test-value", test=1)
            self.assertEqual(ms._last_cv_kwargs, {"test": 1, "test_key": "test-value"})

    @patch(f"{ModelSpec.__module__}.compute_diagnostics")
    @patch(f"{ModelSpec.__module__}.cross_validate", side_effect=NotImplementedError)
    def test_cross_validate_with_non_GP_model(
        self, mock_cv: Mock, mock_diagnostics: Mock
    ) -> None:
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
        ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        self.assertIsNone(ms.fixed_features)
        new_features = ObservationFeatures(parameters={"a": 1.0})
        ms.fixed_features = new_features
        self.assertEqual(ms.fixed_features, new_features)
        self.assertEqual(ms.model_gen_kwargs["fixed_features"], new_features)

    def test_gen_attaches_empty_model_fit_metadata_if_fit_not_applicable(self) -> None:
        ms = ModelSpec(model_enum=Models.SOBOL)
        ms.fit(experiment=self.experiment, data=self.data)
        gr = ms.gen(n=1)
        gen_metadata = none_throws(gr.gen_metadata)
        self.assertEqual(gen_metadata["model_fit_quality"], None)
        self.assertEqual(gen_metadata["model_std_quality"], None)
        self.assertEqual(gen_metadata["model_fit_generalization"], None)
        self.assertEqual(gen_metadata["model_std_generalization"], None)

    def test_gen_attaches_model_fit_metadata_if_applicable(self) -> None:
        ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
        ms.fit(experiment=self.experiment, data=self.data)
        gr = ms.gen(n=1)
        gen_metadata = none_throws(gr.gen_metadata)
        self.assertIsInstance(gen_metadata["model_fit_quality"], float)
        self.assertIsInstance(gen_metadata["model_std_quality"], float)
        self.assertIsInstance(gen_metadata["model_fit_generalization"], float)
        self.assertIsInstance(gen_metadata["model_std_generalization"], float)

    def test_spec_string_representation(self) -> None:
        ms = ModelSpec(
            model_enum=Models.BOTORCH_MODULAR,
            model_kwargs={"test_model_kwargs": 1},
            model_gen_kwargs={"test_gen_kwargs": 1},
            model_cv_kwargs={"test_cv_kwargs": 1},
        )
        ms.model_key_override = "test_model_key_override"

        repr_str = repr(ms)

        self.assertNotIn("\n", repr_str)
        self.assertIn("test_model_kwargs", repr_str)
        self.assertIn("test_gen_kwargs", repr_str)
        self.assertIn("test_cv_kwargs", repr_str)
        self.assertIn("test_model_key_override", repr_str)


class FactoryFunctionModelSpecTest(BaseModelSpecTest):
    def test_construct(self) -> None:
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        with self.assertRaises(UserInputError):
            ms.gen(n=1)
        ms.fit(experiment=self.experiment, data=self.data)
        ms.gen(n=1)

    def test_model_key(self) -> None:
        ms = FactoryFunctionModelSpec(factory_function=get_sobol)
        self.assertEqual(ms.model_key, "get_sobol")
        with self.assertRaisesRegex(TypeError, "cannot extract name"):
            # pyre-ignore[6] - Invalid factory function for testing.
            FactoryFunctionModelSpec(factory_function="test")
        ms = FactoryFunctionModelSpec(
            factory_function=get_sobol, model_key_override="fancy sobol"
        )
        self.assertEqual(ms.model_key, "fancy sobol")
