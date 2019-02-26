#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.models.model_utils import (
    add_fixed_features,
    rejection_sample,
    tunable_feature_indices,
    validate_bounds,
)


class RandomModel:
    """This class specifies the basic skeleton for a random model.

    As random generators do not make use of models, they do not implement
    the fit or predict methods.

    These models do not need data, or optimization configs.

    To satisfy search space parameter constraints, these models can use
    rejection sampling. To enable rejection sampling for a subclass, only
    only `_gen_samples` needs to be implemented, or alternatively,
    `_gen_unconstrained`/`gen` can be directly implemented
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self.seed = seed

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
                Defined on [0, 1]^d.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                    A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            model_gen_options: A config dictionary that is passed along to the
                model.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations).

        Returns:
            X: An (n x d) array of generated points.
            w: Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=bounds, fixed_features=fixed_features
        )
        if fixed_features:
            fixed_feature_indices = np.array(list(fixed_features.keys()))
        else:
            fixed_feature_indices = np.array([])

        validate_bounds(bounds=bounds, fixed_feature_indices=fixed_feature_indices)
        attempted_draws = 0
        if linear_constraints is None or len(linear_constraints) == 0:
            points = self._gen_unconstrained(
                n=n,
                d=len(bounds),
                fixed_features=fixed_features,
                tunable_feature_indices=tf_indices,
            )
            attempted_draws = n
        else:
            max_draws = None
            if model_gen_options:
                max_draws = model_gen_options.get("max_rs_draws")
                if max_draws is not None:
                    max_draws: int = int(max_draws)
            points, attempted_draws = rejection_sample(
                model=self,
                n=n,
                d=len(bounds),
                tunable_feature_indices=tf_indices,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                max_draws=max_draws,
            )
        self.attempted_draws = attempted_draws
        return (points, np.ones(len(points)))

    def _gen_unconstrained(
        self,
        n: int,
        d: int,
        tunable_feature_indices: np.ndarray,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Generate n points, from an unconstrained parameter space, using _gen_samples.

        Args:
            n: Number of points to generate.
            d: Dimension of parameter space.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            tunable_feature_indices: Parameter indices (in d) which are tunable.

        Returns:
            X: An (n x d) array of generated points.

        """
        tunable_points = self._gen_samples(n=n, tunable_d=len(tunable_feature_indices))
        points = add_fixed_features(
            tunable_points=tunable_points,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )
        return points

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate n samples on [0, 1]^d.

        Args:
            n: Number of points to generate.

        Returns:
            X: An (n x d) array of generated points.

        """
        raise NotImplementedError("Base RandomModel can't generate samples.")
