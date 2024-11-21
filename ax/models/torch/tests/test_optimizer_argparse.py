#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest.mock import MagicMock

from ax.exceptions.core import UnsupportedError
from ax.models.torch.botorch_modular.optimizer_argparse import (
    BATCH_LIMIT,
    INIT_BATCH_LIMIT,
    NUM_RESTARTS,
    optimizer_argparse,
    RAW_SAMPLES,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)


class DummyAcquisitionFunction(AcquisitionFunction):
    def __init__(self) -> None:
        return

    # pyre-fixme[14]: Inconsistent override
    # pyre-fixme[15]: Inconsistent override
    def forward(self) -> int:
        return 0


class OptimizerArgparseTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.acqf = DummyAcquisitionFunction()
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

    def test_unsupported_optimizer(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "optimizer=`wishful thinking` is not supported"
        ):
            optimizer_argparse(self.acqf, optimizer="wishful thinking")

    def test_optimizer_options(self) -> None:
        # currently there is only one function in fns_to_test
        user_options = {"foo": "bar", "num_restarts": 13}
        for optimizer in [
            "optimize_acqf",
            "optimize_acqf_discrete",
            "optimize_acqf_mixed",
            "optimize_acqf_discrete_local_search",
        ]:
            with self.subTest(optimizer=optimizer):
                parsed_options = optimizer_argparse(
                    self.acqf, optimizer_options=user_options, optimizer=optimizer
                )
                self.assertDictEqual(
                    {**self.default_expected_options[optimizer], **user_options},
                    parsed_options,
                )

        # Also test sub-options.
        inner_options = {"batch_limit": 10, "maxiter": 20}
        options = {"options": inner_options}
        for optimizer in [
            "optimize_acqf",
            "optimize_acqf_mixed",
            "optimize_acqf_mixed_alternating",
        ]:
            default = self.default_expected_options[optimizer]
            parsed_options = optimizer_argparse(
                self.acqf, optimizer_options=options, optimizer=optimizer
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
                optimizer_argparse(
                    self.acqf,
                    optimizer_options={"options": {"batch_limit": 10, "maxiter": 20}},
                    optimizer=optimizer,
                )

    def test_kg(self) -> None:
        user_options = {"foo": "bar", "num_restarts": 114}
        generic_options = optimizer_argparse(
            self.acqf, optimizer_options=user_options, optimizer="optimize_acqf"
        )
        for acqf in (
            qKnowledgeGradient(model=MagicMock(), posterior_transform=MagicMock()),
            qMultiFidelityKnowledgeGradient(
                model=MagicMock(), posterior_transform=MagicMock()
            ),
        ):
            with self.subTest(acqf=acqf):
                options = optimizer_argparse(
                    acqf, optimizer_options=user_options, optimizer="optimize_acqf"
                )
                self.assertEqual(options, generic_options)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "Ax is attempting to use a discrete or mixed optimizer, "
                    "`optimize_acqf_mixed`, ",
                ):
                    optimizer_argparse(acqf, optimizer="optimize_acqf_mixed")
