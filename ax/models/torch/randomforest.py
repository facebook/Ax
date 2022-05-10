#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.models.torch.utils import _datasets_to_legacy_inputs
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from botorch.utils.datasets import SupervisedDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from torch import Tensor


class RandomForest(TorchModel):
    """A Random Forest model.

    Uses a parametric bootstrap to handle uncertainty in Y.

    Can be used to fit data, make predictions, and do cross validation; however
    gen is not implemented and so this model cannot generate new points.

    Args:
        max_features: Maximum number of features at each split. With one-hot
            encoding, this should be set to None. Defaults to "sqrt", which is
            Breiman's version of Random Forest.
        num_trees: Number of trees.
    """

    def __init__(
        self, max_features: Optional[str] = "sqrt", num_trees: int = 500
    ) -> None:
        self.max_features = max_features
        self.num_trees = num_trees
        self.models: List[RandomForestRegressor] = []

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        for X, Y, Yvar in zip(Xs, Ys, Yvars):
            self.models.append(
                _get_rf(
                    X=X.numpy(),
                    Y=Y.numpy(),
                    Yvar=Yvar.numpy(),
                    num_trees=self.num_trees,
                    max_features=self.max_features,
                )
            )

    @copy_doc(TorchModel.predict)
    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        return _rf_predict(self.models, X)

    @copy_doc(TorchModel.cross_validate)
    def cross_validate(  # pyre-ignore [14]: not using metric_names or ssd
        self,
        datasets: List[SupervisedDataset],
        X_test: Tensor,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        Xs, Ys, Yvars = _datasets_to_legacy_inputs(datasets=datasets)
        cv_models: List[RandomForestRegressor] = []
        for X, Y, Yvar in zip(Xs, Ys, Yvars):
            cv_models.append(
                _get_rf(
                    X=X.numpy(),
                    Y=Y.numpy(),
                    Yvar=Yvar.numpy(),
                    num_trees=self.num_trees,
                    max_features=self.max_features,
                )
            )
        return _rf_predict(cv_models, X_test)


def _get_rf(
    X: np.ndarray,
    Y: np.ndarray,
    Yvar: np.ndarray,
    num_trees: int,
    max_features: Optional[str],
) -> RandomForestRegressor:
    """Fit a Random Forest model.

    Args:
        X: X
        Y: Y
        Yvar: Variance for Y
        num_trees: Number of trees
        max_features: Max features specifier

    Returns: Fitted Random Forest.
    """
    r = RandomForestRegressor(
        n_estimators=num_trees, max_features=max_features, bootstrap=True
    )
    # pyre-fixme[16]: `RandomForestRegressor` has no attribute `estimators_`.
    r.estimators_ = [DecisionTreeRegressor() for i in range(r.n_estimators)]
    for estimator in r.estimators_:
        # Parametric bootstrap
        y = np.random.normal(loc=Y[:, 0], scale=np.sqrt(Yvar[:, 0]))
        estimator.fit(X, y)
    return r


def _rf_predict(
    models: List[RandomForestRegressor], X: Tensor
) -> Tuple[Tensor, Tensor]:
    """Make predictions with Random Forest models.

    Args:
        models: List of models for each outcome
        X: X to predict

    Returns:
        mean and covariance estimates
    """
    f = np.zeros((X.shape[0], len(models)))
    cov = np.zeros((X.shape[0], len(models), len(models)))
    for i, m in enumerate(models):
        # pyre-fixme[16]: `RandomForestRegressor` has no attribute `estimators_`.
        preds = np.vstack([tree.predict(X.numpy()) for tree in m.estimators_])
        f[:, i] = preds.mean(0)
        cov[:, i, i] = preds.var(0)
    return torch.from_numpy(f), torch.from_numpy(cov)
