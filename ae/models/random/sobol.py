#!/usr/bin/env python3

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ae.lazarus.ae.core.types.types import TConfig
from ae.lazarus.ae.models.model_utils import tunable_feature_indices
from ae.lazarus.ae.models.random.base import RandomModel
from ae.lazarus.ae.utils.stats.sobol import (  # pyre-ignore: Not handling .pyx properly
    SobolEngine,
)


class SobolGenerator(RandomModel):
    """This class specifies the generation algorithm for a Sobol generator.

    As Sobol does not make use of a model, it does not implement
    the fit or predict methods.

    Attributes:
        init_position: The initial state of the Sobol generator.
            Starts at 0 by default.
        scramble: If True, permutes the parameter values among
            the elements of the Sobol sequence. Default is True.
        seed: An optional seed value for scrambling.

    """

    engine: Optional[SobolEngine] = None

    def __init__(
        self, init_position: int = 0, scramble: bool = True, seed: Optional[int] = None
    ) -> None:
        super().__init__(seed)
        self.init_position = init_position
        self.scramble = scramble
        # Initialize engine on gen.
        self._engine = None

    # pyre-fixme[11]: Type `SobolEngine` is not defined.
    def init_engine(self, n_tunable_features: int) -> SobolEngine:
        """Initialize singleton SobolEngine, only on gen.

        Args:
            n_tunable_features: The number of features which can be
                searched over.

        Returns:
            engine: SobolEngine, which can generate Sobol points.

        """
        if not self._engine:
            self._engine = SobolEngine(  # pyre-ignore: .pyx not parsed properly.
                dimen=n_tunable_features, scramble=self.scramble, seed=self.seed
            ).fast_forward(self.init_position)
        return self._engine

    @property
    # pyre-fixme[11]: Type `SobolEngine` is not defined.
    def engine(self) -> SobolEngine:
        """Return a singleton SobolEngine."""
        return self._engine

    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: Optional[np.ndarray],
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[np.ndarray]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                    A f(x) <= b.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                    A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature arrays X
                for m outcomes and k_i pending observations for outcome i.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations)
                but *unused here*.

        Returns:
            X: An (n x d) array of generated points.
            w: Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=bounds, fixed_features=fixed_features
        )
        self.init_engine(len(tf_indices))
        points, weights = super().gen(
            n=n,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            model_gen_options=model_gen_options,
        )
        if self.engine:
            self.init_position = self.engine.num_generated
        return (points, weights)

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate n samples.

        tunable_d is ignored; as it is specified at engine initialization.

        Args:
            bounds: A list of d (lower, upper) tuples for each column of X.
            fixed_feature_indices: Indices of features which are fixed at a
                particular value.
        """
        if self.engine is None:
            raise ValueError(
                "Sobol Engine must be initialized before "
                "candidate generation."  # pragma nocover
            )
        return self.engine.draw(n)
