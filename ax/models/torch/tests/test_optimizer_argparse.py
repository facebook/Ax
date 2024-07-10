#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from importlib import reload
from unittest.mock import patch

from ax.models.torch.botorch_modular import optimizer_argparse as Argparse
from ax.models.torch.botorch_modular.optimizer_argparse import (
    _argparse_base,
    INIT_BATCH_LIMIT,
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
        # This has a bespoke test
        skipped_func = optimizer_argparse[qKnowledgeGradient]
        user_options = {"foo": "bar", "num_restarts": 13}
        for func in optimizer_argparse.funcs.values():
            if func is skipped_func:
                continue

            parsed_options = func(None, optimizer_options=user_options)
            for key, val in user_options.items():
                self.assertEqual(val, parsed_options.get(key))

        # Also test sub-options.
        func = _argparse_base
        parsed_options = func(
            None, optimizer_options={"options": {"batch_limit": 10, "maxiter": 20}}
        )
        self.assertEqual(
            parsed_options["options"],
            {"batch_limit": 10, "init_batch_limit": INIT_BATCH_LIMIT, "maxiter": 20},
        )

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
