#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import numpy as np
from ax.core.types import TConfig, TGenMetadata, TParamValue, TParamValueList
from ax.models.base import Model


class DiscreteModel(Model):
    """This class specifies the interface for a model based on discrete parameters.

    These methods should be implemented to have access to all of the features
    of Ax.
    """

    def fit(
        self,
        Xs: List[List[TParamValueList]],
        Ys: List[List[float]],
        Yvars: List[List[float]],
        parameter_values: List[TParamValueList],
        outcome_names: List[str],
    ) -> None:
        """Fit model to m outcomes.

        Args:
            Xs: A list of m lists X of parameterizations (each parameterization
                is a list of parameter values of length d), each of length k_i,
                for each outcome.
            Ys: The corresponding list of m lists Y, each of length k_i, for
                each outcome.
            Yvars: The variances of each entry in Ys, same shape.
            parameter_values: A list of possible values for each parameter.
            outcome_names: A list of m outcome names.
        """
        pass

    def predict(self, X: List[TParamValueList]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict

        Args:
            X: List of the j parameterizations at which to make predictions.

        Returns:
            2-element tuple containing

            - (j x m) array of outcome predictions at X.
            - (j x m x m) array of predictive covariances at X.
              cov[j, m1, m2] is Cov[m1@j, m2@j].
        """
        raise NotImplementedError

    def gen(
        self,
        n: int,
        parameter_values: List[TParamValueList],
        objective_weights: Optional[np.ndarray],
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, TParamValue]] = None,
        pending_observations: Optional[List[List[TParamValueList]]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Tuple[List[TParamValueList], List[float], TGenMetadata]:
        """
        Generate new candidates.

        Args:
            n: Number of candidates to generate.
            parameter_values: A list of possible values for each parameter.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m lists of parameterizations
                (each parameterization is a list of parameter values of length d),
                each of length k_i, for each outcome i.
            model_gen_options: A config dictionary that can contain
                model-specific options.

        Returns:
            2-element tuple containing

            - List of n generated points, where each point is represented
              by a list of parameter values.
            - List of weights for each of the n points.
        """
        raise NotImplementedError

    def cross_validate(
        self,
        Xs_train: List[List[TParamValueList]],
        Ys_train: List[List[float]],
        Yvars_train: List[List[float]],
        X_test: List[TParamValueList],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do cross validation with the given training and test sets.

        Training set is given in the same format as to fit. Test set is given
        in the same format as to predict.

        Args:
            Xs_train: A list of m lists X of parameterizations (each parameterization
                is a list of parameter values of length d), each of length k_i,
                for each outcome.
            Ys_train: The corresponding list of m lists Y, each of length k_i, for
                each outcome.
            Yvars_train: The variances of each entry in Ys, same shape.
            X_test: List of the j parameterizations at which to make predictions.

        Returns:
            2-element tuple containing

            - (j x m) array of outcome predictions at X.
            - (j x m x m) array of predictive covariances at X.
              `cov[j, m1, m2]` is `Cov[m1@j, m2@j]`.
        """
        raise NotImplementedError

    def best_point(
        self,
        n: int,
        parameter_values: List[TParamValueList],
        objective_weights: Optional[np.ndarray],
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, TParamValue]] = None,
        pending_observations: Optional[List[List[TParamValueList]]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[TParamValueList]:
        """Obtains the point that has the best value according to the model
        prediction and its model predictions.

        Returns:
            (1 x d) parameter value list representing the point with the best
            value according to the model prediction. None if this function
            is not implemented for the given model.
        """
        return None  # pragma: no cover TODO[bletham, drfreund]
