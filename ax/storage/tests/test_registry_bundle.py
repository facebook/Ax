# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.metrics.botorch_test_problem import BotorchTestProblemMetric
from ax.metrics.branin import BraninMetric
from ax.runners.botorch_test_problem import BotorchTestProblemRunner
from ax.runners.synthetic import SyntheticRunner
from ax.storage.registry_bundle import RegistryBundle
from ax.utils.common.testutils import TestCase


class RegistryBundleTest(TestCase):
    def test_from_registry_bundles(self) -> None:
        left = RegistryBundle(
            metric_clss={BraninMetric: None},
            runner_clss={SyntheticRunner: None},
            json_encoder_registry={},
            json_class_encoder_registry={},
            json_decoder_registry={},
            json_class_decoder_registry={},
        )

        right = RegistryBundle(
            metric_clss={BotorchTestProblemMetric: None},
            runner_clss={BotorchTestProblemRunner: None},
            json_encoder_registry={},
            json_class_encoder_registry={},
            json_decoder_registry={},
            json_class_decoder_registry={},
        )

        self.assertIn(BraninMetric, left.encoder_registry)
        self.assertNotIn(BotorchTestProblemMetric, left.encoder_registry)

        combined = RegistryBundle.from_registry_bundles(left, right)

        self.assertIn(BraninMetric, combined.encoder_registry)
        self.assertIn(SyntheticRunner, combined.encoder_registry)
        self.assertIn(BotorchTestProblemMetric, combined.encoder_registry)
        self.assertIn(BotorchTestProblemRunner, combined.encoder_registry)
