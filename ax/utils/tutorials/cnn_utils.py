#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset


class CNN(nn.Module):
    """
    Convolutional Neural Network.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8 * 8 * 20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 8 * 8 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def load_mnist(
    downsample_pct: float = 0.5,
    train_pct: float = 0.8,
    data_path: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset (download if necessary) and split data into training,
        validation, and test sets.

    Args:
        downsample_pct: the proportion of the dataset to use for training,
            validation, and test
        train_pct: the proportion of the downsampled data to use for training
        data_path: Root directory of dataset where `MNIST/processed/training.pt`
            and `MNIST/processed/test.pt` exist.
        batch_size: how many samples per batch to load
        num_workers: number of workers (subprocesses) for loading data
        deterministic_partitions: whether to partition data in a deterministic
            fashion
        downsample_pct_test: the proportion of the dataset to use for test, default
            to be equal to downsample_pct

    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    # Specify transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # Load training set
    train_valid_set = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    # Load test set
    test_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )
    return get_partition_data_loaders(
        train_valid_set=train_valid_set,
        test_set=test_set,
        downsample_pct=downsample_pct,
        train_pct=train_pct,
        batch_size=batch_size,
        num_workers=num_workers,
        deterministic_partitions=deterministic_partitions,
        downsample_pct_test=downsample_pct_test,
    )


def get_partition_data_loaders(
    train_valid_set: Dataset,
    test_set: Dataset,
    downsample_pct: float = 0.5,
    train_pct: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 0,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper function for partitioning training data into training and validation sets,
        downsampling data, and initializing DataLoaders for each partition.

    Args:
        train_valid_set: torch.dataset
        downsample_pct: the proportion of the dataset to use for training, and
            validation
        train_pct: the proportion of the downsampled data to use for training
        batch_size: how many samples per batch to load
        num_workers: number of workers (subprocesses) for loading data
        deterministic_partitions: whether to partition data in a deterministic
            fashion
        downsample_pct_test: the proportion of the dataset to use for test, default
            to be equal to downsample_pct

    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    # Partition into training/validation
    # pyre-ignore [6]
    downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_valid_examples = downsampled_num_examples - n_train_examples
    train_set, valid_set, _ = split_dataset(
        dataset=train_valid_set,
        lengths=[
            n_train_examples,
            n_valid_examples,
            len(train_valid_set) - downsampled_num_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    if downsample_pct_test is None:
        downsample_pct_test = downsample_pct
    # pyre-ignore [6]
    downsampled_num_test_examples = int(downsample_pct_test * len(test_set))
    test_set, _ = split_dataset(
        test_set,
        lengths=[
            downsampled_num_test_examples,
            len(test_set) - downsampled_num_test_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader


def split_dataset(
    dataset: Dataset, lengths: List[int], deterministic_partitions: bool = False
) -> List[Dataset]:
    """
    Split a dataset either randomly or deterministically.

    Args:
        dataset: the dataset to split
        lengths: the lengths of each partition
        deterministic_partitions: deterministic_partitions: whether to partition
            data in a deterministic fashion

    Returns:
        List[Dataset]: split datasets
    """
    if deterministic_partitions:
        indices = list(range(sum(lengths)))
    else:
        indices = torch.randperm(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


def train(
    net: torch.nn.Module,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(
        net.parameters(),
        lr=parameters.get("lr", 0.001),
        momentum=parameters.get("momentum", 0.0),
        weight_decay=parameters.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(parameters.get("step_size", 30)),
        gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
    )
    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net


def evaluate(
    net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
