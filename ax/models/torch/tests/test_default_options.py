#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.models.torch.botorch_modular.default_options import (
    register_default_optimizer_options,
    get_default_optimizer_options,
    mk_generic_default_optimizer_options,
    DEFAULT_OPTIMIZER_OPTIONS,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)


class DummyACQF(AcquisitionFunction):
    pass


class AcquisitionTest(TestCase):
    def test_register_and_get_default_options(self):
        register_default_optimizer_options(
            acqf_class=DummyACQF, default_options={"foo": "bar"}
        )
        self.assertEqual(
            get_default_optimizer_options(acqf_class=DummyACQF), {"foo": "bar"}
        )
        DEFAULT_OPTIMIZER_OPTIONS.pop(DummyACQF)  # Clean up for other tests.

    def test_get_default_options_non_registered_acqf(self):
        self.assertEqual(
            get_default_optimizer_options(acqf_class=DummyACQF),
            mk_generic_default_optimizer_options(),
        )

    def test_default_options_qNEI_and_qEI(self):
        self.assertIn(qExpectedImprovement, DEFAULT_OPTIMIZER_OPTIONS)
        self.assertEqual(
            get_default_optimizer_options(acqf_class=qExpectedImprovement),
            mk_generic_default_optimizer_options(),
        )
        self.assertIn(qNoisyExpectedImprovement, DEFAULT_OPTIMIZER_OPTIONS)
        self.assertEqual(
            get_default_optimizer_options(acqf_class=qNoisyExpectedImprovement),
            mk_generic_default_optimizer_options(),
        )

    def test_default_options_qEHVI(self):
        for acqf_class in (
            qExpectedHypervolumeImprovement,
            qNoisyExpectedHypervolumeImprovement,
        ):
            self.assertIn(acqf_class, DEFAULT_OPTIMIZER_OPTIONS)
            self.assertEqual(
                get_default_optimizer_options(acqf_class=acqf_class),
                {
                    "sequential": True,
                    "num_restarts": 40,
                    "raw_samples": 1024,
                    "options": {
                        "init_batch_limit": 128,
                        "batch_limit": 5,
                    },
                },
            )
