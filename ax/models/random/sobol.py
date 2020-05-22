#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from ax.core.types import TConfig
from ax.models.base import Model
from ax.models.model_utils import tunable_feature_indices
from ax.models.random.base import RandomModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none
from torch.quasirandom import SobolEngine


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
        generated_points: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            deduplicate=deduplicate, seed=seed, generated_points=generated_points
        )
        self.init_position = init_position
        self.scramble = scramble
        # Initialize engine on gen.
        self._engine = None

    def init_engine(self, n_tunable_features: int) -> SobolEngine:
        """Initialize singleton SobolEngine, only on gen.

        Args:
            n_tunable_features: The number of features which can be
                searched over.

        Returns:
            SobolEngine, which can generate Sobol points.

        """
        if not self._engine:
            self._engine = SobolEngine(
                dimension=n_tunable_features, scramble=self.scramble, seed=self.seed
            ).fast_forward(self.init_position)
        return self._engine

    @property
    def engine(self) -> Optional[SobolEngine]:
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
        if len(tf_indices) > 0:
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
            self.init_position = not_none(self.engine).num_generated
        return (points, weights)

    @copy_doc(Model._get_state)
    def _get_state(self) -> Dict[str, Any]:
        state = super()._get_state()
        state.update({"init_position": self.init_position})
        return state

    @copy_doc(RandomModel._gen_unconstrained)
    def _gen_unconstrained(
        self,
        n: int,
        d: int,
        tunable_feature_indices: np.ndarray,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        if len(tunable_feature_indices) == 0:
            # Search space is entirely fixed, should return the only avail. point.
            fixed_features = fixed_features or {}
            # pyre-fixme[7]: Expected `ndarray` but got `Tuple[typing.Any, typing.Any]`.
            return (
                np.tile(np.array([list(not_none(fixed_features).values())]), (n, 1)),
                np.ones(n),
            )
        return super()._gen_unconstrained(
            n=n,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate n samples.

        tunable_d is ignored; as it is specified at engine initialization.

        Args:
            bounds: A list of d (lower, upper) tuples for each column of X.
            fixed_feature_indices: Indices of features which are fixed at a
                particular value.
        """
        if self.engine is None:
            raise ValueError(  # pragma: no cover
                "Sobol Engine must be initialized before candidate generation."
            )
        return not_none(self.engine).draw(n, dtype=torch.double).numpy()
