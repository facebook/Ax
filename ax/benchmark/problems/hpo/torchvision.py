# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from ax.benchmark.problems.hpo.pytorch_cnn import (
    PyTorchCNNBenchmarkProblem,
    PyTorchCNNRunner,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.typeutils import checked_cast

try:  # We don't require TorchVision by default.
    from torchvision import datasets, transforms

    _REGISTRY = {"MNIST": datasets.MNIST, "FashionMNIST": datasets.FashionMNIST}
except ModuleNotFoundError:
    transforms = None
    datasets = None
    _REGISTRY = {}


class PyTorchCNNTorchvisionBenchmarkProblem(PyTorchCNNBenchmarkProblem):
    @classmethod
    def from_dataset_name(
        cls, name: str, num_trials: int
    ) -> "PyTorchCNNTorchvisionBenchmarkProblem":
        if name not in _REGISTRY:
            raise UserInputError(
                f"Unrecognized torchvision dataset {name}. Please ensure it is listed"
                "in PyTorchCNNTorchvisionBenchmarkProblem registry."
            )
        dataset_fn = _REGISTRY[name]

        train_set = dataset_fn(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        test_set = dataset_fn(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        problem = cls.from_datasets(
            name=name, num_trials=num_trials, train_set=train_set, test_set=test_set
        )
        runner = PyTorchCNNTorchvisionRunner(
            name=name, train_set=train_set, test_set=test_set
        )

        return cls(
            name=f"HPO_PyTorchCNN_Torchvision::{name}",
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            runner=runner,
            num_trials=num_trials,
            optimal_value=problem.optimal_value,
        )


class PyTorchCNNTorchvisionRunner(PyTorchCNNRunner):
    """
    A subclass to aid in serialization. This allows us to save only the name of the
    dataset and reload it from TorchVision at deserialization time.
    """

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        pytorch_cnn_runner = checked_cast(PyTorchCNNRunner, obj)

        return {"name": pytorch_cnn_runner.name}

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]

        dataset_fn = _REGISTRY[name]

        train_set = dataset_fn(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        test_set = dataset_fn(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        return {"name": name, "train_set": train_set, "test_set": test_set}
