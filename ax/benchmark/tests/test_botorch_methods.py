#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.botorch_methods import (
    fixed_noise_gp_model_constructor,
    make_basic_generation_strategy,
)
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.utils.common.testutils import TestCase


class TestBoTorchMethods(TestCase):
    def test_make_basic_generation_strategy(self):
        singletask_NEI = make_basic_generation_strategy(
            name="NEI + SingleTaskGP", acquisition="NEI", num_initial_trials=6
        )
        self.assertIsInstance(singletask_NEI, GenerationStrategy)
        fixednoise_NEI = make_basic_generation_strategy(
            name="NEI + FixedNoiseGP",
            acquisition="NEI",
            num_initial_trials=6,
            surrogate_model_constructor=fixed_noise_gp_model_constructor,
        )
        self.assertIsInstance(fixednoise_NEI, GenerationStrategy)
