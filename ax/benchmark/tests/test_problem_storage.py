# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os

from ax.benchmark.problems.hpo.torchvision import PyTorchCNNTorchvisionBenchmarkProblem
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def tearDown(self) -> None:
        mnist_dir_path = "data/MNIST/raw"
        files = glob.glob(mnist_dir_path + "/*")
        for f in files:
            os.remove(f)
        os.rmdir(mnist_dir_path)
        super().tearDown()

    def test_torchvision_encode_decode(self) -> None:
        original_object = PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name(
            name="MNIST", num_trials=50
        )

        json_object = object_to_json(
            original_object,
        )
        converted_object = object_from_json(
            json_object,
        )

        self.assertEqual(original_object, converted_object)
