#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from importlib import reload
from unittest.mock import patch

from ax.models.torch.botorch_modular import optimizer_argparse as Argparse
from ax.models.torch.botorch_modular.optimizer_argparse import (
    _argparse_base,
    MaybeType,
    optimizer_argparse,
)
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)


class DummyAcquisitionFunction(AcquisitionFunction):
    pass


class OptimizerArgparseTest(TestCase):
    def test_notImplemented(self) -> None:
        with self.assertRaises(NotImplementedError) as e:
            optimizer_argparse[type(None)]  # passing `None` produces a different error
            self.assertTrue("Could not find signature for" in str(e))

    def test_register(self) -> None:
        with patch.dict(optimizer_argparse.funcs, {}):

            @optimizer_argparse.register(DummyAcquisitionFunction)
            # pyre-fixme[3]: Return type must be annotated.
            def _argparse(acqf: MaybeType[DummyAcquisitionFunction]):
                pass

            self.assertEqual(optimizer_argparse[DummyAcquisitionFunction], _argparse)

    def test_fallback(self) -> None:
        with patch.dict(optimizer_argparse.funcs, {}):

            @optimizer_argparse.register(AcquisitionFunction)
            # pyre-fixme[3]: Return type must be annotated.
            def _argparse(acqf: MaybeType[DummyAcquisitionFunction]):
                pass

            self.assertEqual(optimizer_argparse[DummyAcquisitionFunction], _argparse)

    def test_optimizer_options(self) -> None:
        skipped_funcs = {  # These should all have bespoke tests
            optimizer_argparse[acqf_class]
            for acqf_class in (
                qExpectedHypervolumeImprovement,
                qKnowledgeGradient,
                qMaxValueEntropy,
            )
        }
        user_options = {"foo": "bar", "num_restarts": 13}
        for func in optimizer_argparse.funcs.values():
            if func in skipped_funcs:
                continue

            parsed_options = func(None, optimizer_options=user_options)
            for key, val in user_options.items():
                self.assertEqual(val, parsed_options.get(key))

    def test_ehvi(self) -> None:
        user_options = {"foo": "bar", "num_restarts": 651}
        inner_options = {"init_batch_limit": 23, "batch_limit": 67}
        generic_options = _argparse_base(None, optimizer_options=user_options)
        generic_options.pop("options")
        for acqf in (
            qExpectedHypervolumeImprovement,
            qNoisyExpectedHypervolumeImprovement,
        ):
            with self.subTest(acqf=acqf):
                options = optimizer_argparse(
                    acqf,
                    sequential=False,
                    optimizer_options=user_options,
                    **inner_options,
                )
                self.assertEqual(options.pop("sequential"), False)
                self.assertEqual(options.pop("options"), inner_options)
                self.assertEqual(options, generic_options)

                # Defaults
                options = optimizer_argparse(
                    acqf,
                    sequential=False,
                    optimizer_options=user_options,
                )
                self.assertEqual(options.pop("sequential"), False)
                self.assertEqual(
                    options.pop("options"), {"init_batch_limit": 32, "batch_limit": 5}
                )
                self.assertEqual(options, generic_options)

    def test_kg(self) -> None:
        with patch(
            "botorch.optim.initializers.gen_one_shot_kg_initial_conditions"
        ) as mock_gen_initial_conditions:
            mock_gen_initial_conditions.return_value = "TEST"
            reload(Argparse)

            user_options = {"foo": "bar", "num_restarts": 114}
            generic_options = _argparse_base(None, optimizer_options=user_options)
            for acqf in (qKnowledgeGradient, qMultiFidelityKnowledgeGradient):
                with self.subTest(acqf=acqf):
                    options = optimizer_argparse(
                        acqf,
                        q=None,
                        bounds=None,
                        optimizer_options=user_options,
                    )
                    self.assertEqual(options.pop(Keys.BATCH_INIT_CONDITIONS), "TEST")
                    self.assertEqual(options, generic_options)

    def test_mes(self) -> None:
        user_options = {"foo": "bar", "num_restarts": 83}
        generic_options = _argparse_base(None, optimizer_options=user_options)
        for acqf in (qMaxValueEntropy, qMultiFidelityMaxValueEntropy):
            with self.subTest(acqf=acqf):
                options = optimizer_argparse(
                    acqf,
                    sequential=False,
                    optimizer_options=user_options,
                )
                self.assertEqual(options.pop("sequential"), False)
                self.assertEqual(options, generic_options)
