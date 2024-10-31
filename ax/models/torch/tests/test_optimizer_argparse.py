#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from itertools import product
from unittest.mock import patch

from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch_modular.optimizer_argparse import (
    _argparse_base,
    BATCH_LIMIT,
    INIT_BATCH_LIMIT,
    MaybeType,
    NUM_RESTARTS,
    optimizer_argparse,
    RAW_SAMPLES,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)


class DummyAcquisitionFunction(AcquisitionFunction):
    pass


class OptimizerArgparseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.default_expected_options = {
            "optimize_acqf": {
                "num_restarts": NUM_RESTARTS,
                "raw_samples": RAW_SAMPLES,
                "options": {
                    "init_batch_limit": INIT_BATCH_LIMIT,
                    "batch_limit": BATCH_LIMIT,
                },
                "sequential": True,
            },
            "optimize_acqf_discrete_local_search": {
                "num_restarts": NUM_RESTARTS,
                "raw_samples": RAW_SAMPLES,
            },
            "optimize_acqf_discrete": {},
            "optimize_acqf_mixed": {
                "num_restarts": NUM_RESTARTS,
                "raw_samples": RAW_SAMPLES,
                "options": {
                    "init_batch_limit": INIT_BATCH_LIMIT,
                    "batch_limit": BATCH_LIMIT,
                },
            },
            "optimize_acqf_mixed_alternating": {
                "num_restarts": NUM_RESTARTS,
                "raw_samples": RAW_SAMPLES,
                "options": {
                    "init_batch_limit": INIT_BATCH_LIMIT,
                    "batch_limit": BATCH_LIMIT,
                },
            },
        }

    def test_notImplemented(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError, "Could not find signature for"
        ):
            optimizer_argparse[type(None)]  # passing `None` produces a different error

    def test_unsupported_optimizer(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "optimizer=`wishful thinking` is not supported"
        ):
            optimizer_argparse(LogExpectedImprovement, optimizer="wishful thinking")

    def test_register(self) -> None:
        with patch.dict(optimizer_argparse.funcs, {}):

            @optimizer_argparse.register(DummyAcquisitionFunction)
            def _argparse(acqf: MaybeType[DummyAcquisitionFunction]) -> None:
                pass

            self.assertEqual(optimizer_argparse[DummyAcquisitionFunction], _argparse)

    def test_fallback(self) -> None:
        with patch.dict(optimizer_argparse.funcs, {}):

            @optimizer_argparse.register(AcquisitionFunction)
            def _argparse(acqf: MaybeType[DummyAcquisitionFunction]) -> None:
                pass

            self.assertEqual(optimizer_argparse[DummyAcquisitionFunction], _argparse)

    def test_optimizer_options(self) -> None:
        # qKG should have a bespoke test
        # currently there is only one function in fns_to_test
        fns_to_test = [
            elt
            for elt in optimizer_argparse.funcs.values()
            if elt is not optimizer_argparse[qKnowledgeGradient]
        ]
        user_options = {"foo": "bar", "num_restarts": 13}
        for func, optimizer in product(
            fns_to_test,
            [
                "optimize_acqf",
                "optimize_acqf_discrete",
                "optimize_acqf_mixed",
                "optimize_acqf_discrete_local_search",
            ],
        ):
            with self.subTest(func=func, optimizer=optimizer):
                parsed_options = func(
                    None, optimizer_options=user_options, optimizer=optimizer
                )
                self.assertDictEqual(
                    {**self.default_expected_options[optimizer], **user_options},
                    parsed_options,
                )

        # Also test sub-options.
        inner_options = {"batch_limit": 10, "maxiter": 20}
        options = {"options": inner_options}
        for func in fns_to_test:
            for optimizer in [
                "optimize_acqf",
                "optimize_acqf_mixed",
                "optimize_acqf_mixed_alternating",
            ]:
                default = self.default_expected_options[optimizer]
                parsed_options = func(
                    None, optimizer_options=options, optimizer=optimizer
                )
                expected_options = {k: v for k, v in default.items() if k != "options"}
                if "options" in default:
                    expected_options["options"] = {
                        **default["options"],
                        **inner_options,
                    }
                else:
                    expected_options["options"] = inner_options
                self.assertDictEqual(expected_options, parsed_options)

            # Error out if options is specified for an optimizer that does
            # not support the arg.
            for optimizer in [
                "optimize_acqf_discrete",
                "optimize_acqf_discrete_local_search",
            ]:
                with self.assertRaisesRegex(UnsupportedError, "`options` argument"):
                    func(
                        None,
                        optimizer_options={
                            "options": {"batch_limit": 10, "maxiter": 20}
                        },
                        optimizer=optimizer,
                    )

            # `sequential=False` with optimizers other than `optimize_acqf`.
            for optimizer in [
                "optimize_acqf_homotopy",
                "optimize_acqf_mixed",
                "optimize_acqf_mixed_alternating",
                "optimize_acqf_discrete",
                "optimize_acqf_discrete_local_search",
            ]:
                with self.assertRaisesRegex(
                    UnsupportedError, "does not support `sequential=False`"
                ):
                    func(None, sequential=False, optimizer=optimizer)

    def test_kg(self) -> None:
        user_options = {"foo": "bar", "num_restarts": 114}
        generic_options = _argparse_base(
            None, optimizer_options=user_options, optimizer="optimize_acqf"
        )
        for acqf in (qKnowledgeGradient, qMultiFidelityKnowledgeGradient):
            with self.subTest(acqf=acqf):
                options = optimizer_argparse(
                    acqf,
                    q=None,
                    bounds=None,
                    optimizer_options=user_options,
                )
                self.assertEqual(options, generic_options)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "Ax is attempting to use a discrete or mixed optimizer, "
                    "`optimize_acqf_mixed`, ",
                ):
                    optimizer_argparse(acqf, optimizer="optimize_acqf_mixed")
