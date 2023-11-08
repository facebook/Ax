# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Set

import pandas as pd
import torch
from ax.benchmark.benchmark_problem import SingleObjectiveBenchmarkProblem
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.result import Err, Ok
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class PyTorchCNNBenchmarkProblem(SingleObjectiveBenchmarkProblem):
    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, PyTorchCNNBenchmarkProblem):
            return False

        # Checking the whole datasets' equality here would be too expensive to be
        # worth it; just check names instead
        return self.name == other.name

    @classmethod
    def from_datasets(
        cls,
        name: str,
        num_trials: int,
        train_set: Dataset,
        test_set: Dataset,
        infer_noise: bool = True,
    ) -> "PyTorchCNNBenchmarkProblem":
        optimal_value = 1

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
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=PyTorchCNNMetric(infer_noise=infer_noise),
                minimize=False,
            )
        )

        runner = PyTorchCNNRunner(name=name, train_set=train_set, test_set=test_set)

        return cls(
            name=f"HPO_PyTorchCNN_{name}",
            optimal_value=optimal_value,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=runner,
            num_trials=num_trials,
            infer_noise=infer_noise,
        )


class PyTorchCNNMetric(Metric):
    def __init__(self, infer_noise: bool = True) -> None:
        super().__init__(name="accuracy")
        self.infer_noise = infer_noise

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        try:
            accuracy = [
                trial.run_metadata["accuracy"][name]
                for name, arm in trial.arms_by_name.items()
            ]
            df = pd.DataFrame(
                {
                    "arm_name": [name for name, _ in trial.arms_by_name.items()],
                    "metric_name": self.name,
                    "mean": accuracy,
                    "sem": None if self.infer_noise else 0,
                    "trial_index": trial.index,
                }
            )

            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                value=MetricFetchE(
                    message=f"Failed to fetch {self.name} for trial {trial}",
                    exception=e,
                )
            )


class PyTorchCNNRunner(Runner):
    def __init__(self, name: str, train_set: Dataset, test_set: Dataset) -> None:
        self.name = name

        # pyre-fixme[4]: Attribute must be annotated.
        self.train_loader = DataLoader(train_set)
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_loader = DataLoader(test_set)

        self.results: Dict[int, float] = {}
        self.statuses: Dict[int, TrialStatus] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CNN(nn.Module):
        # pyre-fixme[3]: Return type must be annotated.
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
            self.fc1 = nn.Linear(8 * 8 * 20, 64)
            self.fc2 = nn.Linear(64, 10)

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 3, 3)
            x = x.view(-1, 8 * 8 * 20)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=-1)

    def train_and_evaluate(
        self,
        lr: float,
        momentum: float,
        weight_decay: float,
        step_size: int,
        gamma: float,
    ) -> float:
        net = self.CNN()
        net.to(device=self.device)

        # Train
        net.train()
        criterion = nn.NLLLoss(reduction="sum")
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        for inputs, labels in self.train_loader:
            inputs = inputs.to(device=self.device)
            labels = labels.to(device=self.device)

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
            for inputs, labels in self.test_loader:
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        self.statuses[trial.index] = TrialStatus.RUNNING

        self.statuses[trial.index] = TrialStatus.COMPLETED
        return {
            "accuracy": {
                arm.name: self.train_and_evaluate(
                    lr=arm.parameters["lr"],  # pyre-ignore[6]
                    momentum=arm.parameters["momentum"],  # pyre-ignore[6]
                    weight_decay=arm.parameters["weight_decay"],  # pyre-ignore[6]
                    step_size=arm.parameters["step_size"],  # pyre-ignore[6]
                    gamma=arm.parameters["gamma"],  # pyre-ignore[6]
                )
                for arm in trial.arms
            }
        }

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}
