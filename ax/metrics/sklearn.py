#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from functools import lru_cache
from math import sqrt
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric
from ax.utils.common.typeutils import checked_cast
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor


class SklearnModelType(Enum):
    RF: str = "rf"
    NN: str = "nn"


class SklearnDataset(Enum):
    DIGITS: str = "digits"
    BOSTON: str = "boston"
    CANCER: str = "cancer"


@lru_cache(maxsize=8)
def _get_data(dataset) -> Dict[str, np.ndarray]:
    """Return sklearn dataset, loading and caching if necessary."""
    if dataset is SklearnDataset.DIGITS:
        return datasets.load_digits()
    elif dataset is SklearnDataset.BOSTON:
        return datasets.load_boston()
    elif dataset is SklearnDataset.CANCER:
        return datasets.load_breast_cancer()
    else:
        raise NotImplementedError(
            f"{dataset.value} is not a currently supported {dataset.name}."
        )


class SklearnMetric(Metric):
    """A metric that trains and evaluates an sklearn model.

    The evaluation metric is the k-fold "score". The scoring function
    depends on the model type and task type (e.g. classification/regression),
    but higher scores are better.

    See sklearn documentation for supported parameters.

    In addition, this metric supports tuning the hidden_layer_size
    and the number of hidden layers (num_hidden_layers) of a NN model.
    """

    def __init__(
        self,
        name: str,
        lower_is_better: bool = False,
        model_type: SklearnModelType = SklearnModelType.RF,
        dataset: SklearnDataset = SklearnDataset.DIGITS,
        observed_noise: bool = False,
        num_folds: int = 5,
    ) -> None:
        """Initialize metric.

        Args:
            name: Name of the metric.
            lower_is_better: Flag for metrics which should be minimized.
            model_type: Sklearn model type
            dataset: Sklearn Dataset for training/evaluating the model
            observed_noise: A boolean indicating whether to return the SE
                of the mean k-fold cross-validation score.
            num_folds: The number of cross-validation folds.
        """
        super().__init__(name=name, lower_is_better=lower_is_better)
        self.dataset = dataset
        self.model_type = model_type
        self.num_folds = num_folds
        self.observed_noise = observed_noise
        if self.dataset is SklearnDataset.DIGITS:
            regression = False
        elif self.dataset is SklearnDataset.BOSTON:
            regression = True
        elif self.dataset is SklearnDataset.CANCER:
            regression = False
        else:
            raise NotImplementedError(
                f"{self.dataset.value} is not a currently supported {dataset.name}."
            )
        if model_type is SklearnModelType.NN:
            if regression:
                self._model_cls = MLPRegressor
            else:
                self._model_cls = MLPClassifier
        elif model_type is SklearnModelType.RF:
            if regression:
                self._model_cls = RandomForestRegressor
            else:
                self._model_cls = RandomForestClassifier
        else:
            raise NotImplementedError(
                f"{model_type.value} is not a currently supported {model_type.name}."
            )

    def clone(self) -> SklearnMetric:
        return self.__class__(
            name=self._name,
            lower_is_better=checked_cast(bool, self.lower_is_better),
            model_type=self.model_type,
            dataset=self.dataset,
        )

    def fetch_trial_data(
        self, trial: BaseTrial, noisy: bool = True, **kwargs: Any
    ) -> Data:
        arm_names = []
        means = []
        sems = []
        for name, arm in trial.arms_by_name.items():
            arm_names.append(name)
            # TODO: Consider parallelizing evaluation of large batches
            # (e.g. via ProcessPoolExecutor)
            mean, sem = self.train_eval(arm=arm)
            means.append(mean)
            sems.append(sem)
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self._name,
                "mean": means,
                "sem": sems,
                "trial_index": trial.index,
            }
        )
        return Data(df=df)

    def train_eval(self, arm: Arm) -> Tuple[float, float]:
        """Train and evaluate model.

        Args:
            arm: An arm specifying the parameters to evaluate.

        Returns:
           A two-element tuple containing:
            - The average k-fold CV score
            - The SE of the mean k-fold CV score if observed_noise is True
                and 'nan' otherwise
        """
        data = _get_data(self.dataset)  # cached
        X, y = data["data"], data["target"]
        params: Dict[str, Any] = deepcopy(arm.parameters)
        if self.model_type == SklearnModelType.NN:
            hidden_layer_size = params.pop("hidden_layer_size", None)
            if hidden_layer_size is not None:
                hidden_layer_size = checked_cast(int, hidden_layer_size)
                num_hidden_layers = checked_cast(
                    int, params.pop("num_hidden_layers", 1)
                )
                params["hidden_layer_sizes"] = [hidden_layer_size] * num_hidden_layers
        model = self._model_cls(**params)
        cv_scores = cross_val_score(model, X, y, cv=self.num_folds)
        mean = cv_scores.mean()
        sem = (
            cv_scores.std() / sqrt(cv_scores.shape[0])
            if self.observed_noise
            else float("nan")
        )
        return mean, sem
