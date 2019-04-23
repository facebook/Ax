#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ax.core.types import TConfig
from ax.models.model_utils import tunable_feature_indices
from ax.models.random.base import RandomModel
from ax.utils.stats.sobol import SobolEngine  # pyre-ignore: Not handling .pyx properly


class SobolGenerator(RandomModel):
    """This class specifies the generation algorithm for a Sobol generator.

    As Sobol does not make use of a model, it does not implement
    the fit or predict methods.

    Attributes:
        deduplicate: If true, a single instantiation of the generator will not
            return the same point twice.
        init_position: The initial state of the Sobol generator.
            Starts at 0 by default.
        scramble: If True, permutes the parameter values among
            the elements of the Sobol sequence. Default is True.
        seed: An optional seed value for scrambling.

    """

    engine: Optional[SobolEngine] = None

    def __init__(
        self,
        seed: Optional[int] = None,
        deduplicate: bool = False,
        init_position: int = 0,
        scramble: bool = True,
    ) -> None:
        super().__init__(deduplicate=deduplicate, seed=seed)
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
            SobolEngine, which can generate Sobol points.

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
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations)
                but *unused here*.

        Returns:
            2-element tuple containing

            - (n x d) array of generated points.
            - Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=bounds, fixed_features=fixed_features
        )
        self.init_engine(len(tf_indices))
        points, weights = super().gen(
            n=n,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            rounding_func=rounding_func,
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
