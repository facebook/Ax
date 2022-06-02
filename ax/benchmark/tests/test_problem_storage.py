# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.benchmark.problems.hpo.torchvision import PyTorchCNNTorchvisionBenchmarkProblem
from ax.benchmark.problems.surrogate import SurrogateBenchmarkProblem

from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.testutils import TestCase
from botorch.models import SingleTaskGP
from botorch.utils.datasets import SupervisedDataset


class TestProblems(TestCase):
    def test_torchvision_encode_decode(self):
        original_object = PyTorchCNNTorchvisionBenchmarkProblem.from_dataset_name(
            name="MNIST"
        )

        json_object = object_to_json(
            original_object,
        )
        converted_object = object_from_json(
            json_object,
        )

        self.assertEqual(original_object, converted_object)

    def test_gp_surrogate_encode_decode(self):
        X = torch.tensor([[1.0, 1.0]])
        Y = torch.tensor([[18.0]])

        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name=f"x{i}", parameter_type=ParameterType.INT, upper=1, lower=0
                )
                for i in range(2)
            ]
        )

        original_object = SurrogateBenchmarkProblem.from_surrogate(
            name="surrogate",
            search_space=search_space,
            surrogate=Surrogate(SingleTaskGP),
            datasets=[SupervisedDataset(X=X, Y=Y)],
            minimize=True,
            optimal_value=0,
        )

        json_object = object_to_json(
            original_object,
        )
        converted_object = object_from_json(
            json_object,
        )

        self.assertEqual(original_object, converted_object)
