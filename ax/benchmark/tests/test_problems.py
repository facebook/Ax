# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark.benchmark_result import AggregatedBenchmarkResult
from ax.benchmark.problems.hpo.torchvision import PyTorchCNNTorchvisionBenchmarkProblem
from ax.benchmark.problems.synthetic import (
    get_problem_and_baseline_from_botorch,
    _REGISTRY,
)
from ax.utils.common.testutils import TestCase


class TestProblems(TestCase):
    def test_load_baselines(self):

        # Make sure the json parsing suceeded
        for name in _REGISTRY.keys():
            _problem, baseline = get_problem_and_baseline_from_botorch(
                problem_name=name
            )

            self.assertTrue(isinstance(baseline, AggregatedBenchmarkResult))

    def test_pytorch_cnn(self):
        # Just check data loading and construction succeeds
        PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name(name="MNIST")
