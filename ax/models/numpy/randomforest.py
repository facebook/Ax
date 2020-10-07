#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import numpy as np
from ax.core.types import TCandidateMetadata
from ax.models.numpy_base import NumpyModel
from ax.utils.common.docutils import copy_doc
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


class RandomForest(NumpyModel):
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

    @copy_doc(NumpyModel.fit)
    def fit(
        self,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        for i, X in enumerate(Xs):
            self.models.append(
                _get_rf(
                    X=X,
                    Y=Ys[i],
                    Yvar=Yvars[i],
                    num_trees=self.num_trees,
                    max_features=self.max_features,
                )
            )

    @copy_doc(NumpyModel.predict)
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _rf_predict(self.models, X)

    @copy_doc(NumpyModel.cross_validate)
    def cross_validate(
        self,
        Xs_train: List[np.ndarray],
        Ys_train: List[np.ndarray],
        Yvars_train: List[np.ndarray],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cv_models: List[RandomForestRegressor] = []
        for i, X in enumerate(Xs_train):
            cv_models.append(
                _get_rf(
                    X=X,
                    Y=Ys_train[i],
                    Yvar=Yvars_train[i],
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
    models: List[RandomForestRegressor], X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions with Random Forest models.

    Args:
        models: List of models for each outcome
        X: X to predict

    Returns: mean and covariance estimates
    """
    f = np.zeros((X.shape[0], len(models)))
    cov = np.zeros((X.shape[0], len(models), len(models)))
    for i, m in enumerate(models):
        preds = np.vstack([tree.predict(X) for tree in m.estimators_])  # pyre-ignore
        f[:, i] = preds.mean(0)
        cov[:, i, i] = preds.var(0)
    return f, cov
