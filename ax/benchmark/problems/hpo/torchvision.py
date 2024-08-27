# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass, field, InitVar
from functools import lru_cache
from typing import Mapping

import torch
from ax.benchmark.benchmark_problem import (
    BenchmarkProblem,
    get_soo_config_and_outcome_names,
)
from ax.benchmark.runners.botorch_test import (
    ParamBasedTestProblem,
    ParamBasedTestProblemRunner,
)
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:  # We don't require TorchVision by default.
    from torchvision import datasets, transforms

    _REGISTRY = {
        "MNIST": datasets.MNIST,
        "FashionMNIST": datasets.FashionMNIST,
    }


except ModuleNotFoundError:
    transforms = None
    datasets = None
    _REGISTRY = {}


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8 * 8 * 20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 8 * 8 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


@lru_cache(maxsize=64)
def train_and_evaluate(
    lr: float,
    momentum: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    device: torch.device,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> torch.Tensor:
    """Return the fraction of correctly classified test examples."""
    net = CNN()
    net.to(device=device)

    # Train
    net.train()
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for inputs, labels in train_loader:
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluate
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return correct / total


@dataclass(kw_only=True)
class PyTorchCNNTorchvisionParamBasedProblem(ParamBasedTestProblem):
    name: str  # The name of the dataset to load -- MNIST or FashionMNIST
    num_objectives: int = 1
    optimal_value: float = 1.0
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    negate: bool = False
    # Using `InitVar` prevents the DataLoaders from being serialized; instead
    # they are reconstructed upon deserialization.
    # Pyre doesn't understand InitVars.
    # pyre-ignore: Undefined attribute [16]: `typing.Type` has no attribute
    # `train_loader`
    train_loader: InitVar[DataLoader | None] = None
    # pyre-ignore
    test_loader: InitVar[DataLoader | None] = None

    def __post_init__(self, train_loader: None, test_loader: None) -> None:
        if self.name not in _REGISTRY:
            raise UserInputError(
                f"Unrecognized torchvision dataset {self.name}. Please ensure it "
                "is listed in ax/benchmark/problems/hop/torchvision.py registry."
            )
        dataset_fn = _REGISTRY[self.name]

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
        # pyre-fixme: Undefined attribute [16]:
        # `PyTorchCNNTorchvisionParamBasedProblem` has no attribute
        # `train_loader`.
        self.train_loader = DataLoader(train_set, num_workers=1)
        # pyre-fixme
        self.test_loader = DataLoader(test_set, num_workers=1)

    # pyre-fixme[14]: Inconsistent override (super class takes a more general
    # type, TParameterization)
    def evaluate_true(self, params: Mapping[str, int | float]) -> Tensor:
        return train_and_evaluate(
            **params,
            device=self.device,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
        )


def get_pytorch_cnn_torchvision_benchmark_problem(
    name: str,
    num_trials: int,
) -> BenchmarkProblem:
    base_problem = PyTorchCNNTorchvisionParamBasedProblem(name=name)

    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name="lr", parameter_type=ParameterType.FLOAT, lower=1e-6, upper=0.4
            ),
            RangeParameter(
                name="momentum",
                parameter_type=ParameterType.FLOAT,
                lower=0,
                upper=1,
            ),
            RangeParameter(
                name="weight_decay",
                parameter_type=ParameterType.FLOAT,
                lower=0,
                upper=1,
            ),
            RangeParameter(
                name="step_size",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=100,
            ),
            RangeParameter(
                name="gamma",
                parameter_type=ParameterType.FLOAT,
                lower=0,
                upper=1,
            ),
        ]
    )
    optimization_config, outcome_names = get_soo_config_and_outcome_names(
        num_constraints=0,
        lower_is_better=False,
        observe_noise_sd=False,
        objective_name="accuracy",
    )
    runner = ParamBasedTestProblemRunner(
        test_problem_class=PyTorchCNNTorchvisionParamBasedProblem,
        test_problem_kwargs={"name": name},
        outcome_names=outcome_names,
    )
    return BenchmarkProblem(
        name=f"HPO_PyTorchCNN_Torchvision::{name}",
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        observe_noise_stds=False,
        optimal_value=base_problem.optimal_value,
        runner=runner,
    )
