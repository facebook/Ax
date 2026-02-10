#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.generators.random.base import RandomGenerator
from ax.generators.types import TConfig
from ax.generators.utils import tunable_feature_indices
from pyre_extensions import none_throws
from torch.quasirandom import SobolEngine


class SobolGenerator(RandomGenerator):
    """This class specifies the generation algorithm for a Sobol generator.

    As Sobol does not make use of a model, it does not implement
    the fit or predict methods.

    Attributes:
        scramble: If True, permutes the parameter values among
            the elements of the Sobol sequence. Default is True.
        See base `RandomGenerator` for a description of remaining attributes.
    """

    def __init__(
        self,
        seed: int | None = None,
        init_position: int = 0,
        scramble: bool = True,
        generated_points: npt.NDArray | None = None,
        fallback_to_sample_polytope: bool = False,
        polytope_sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            seed=seed,
            init_position=init_position,
            generated_points=generated_points,
            fallback_to_sample_polytope=fallback_to_sample_polytope,
            polytope_sampler_kwargs=polytope_sampler_kwargs,
        )
        self.scramble = scramble
        # Initialize engine on gen.
        self._engine: SobolEngine | None = None

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
    def engine(self) -> SobolEngine | None:
        """Return a singleton SobolEngine."""
        return self._engine

    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        linear_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
        fixed_features: dict[int, float] | None = None,
        model_gen_options: TConfig | None = None,
        rounding_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Generate new candidates.

        Args:
            n: Number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations).

        Returns:
            2-element tuple containing

            - (n x d) array of generated points.
            - Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=search_space_digest.bounds, fixed_features=fixed_features
        )
        if len(tf_indices) > 0:
            self.init_engine(len(tf_indices))
        points, weights = super().gen(
            n=n,
            search_space_digest=search_space_digest,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            rounding_func=rounding_func,
        )
        if self.engine:
            self.init_position = none_throws(self.engine).num_generated
        return points, weights

    def _gen_samples(self, n: int, tunable_d: int) -> npt.NDArray:
        """Generate n samples.

        Args:
            n: Number of samples to generate.
            tunable_d: The dimension of the generated samples. This must
                match the tunable parameters used while initializing the
                Sobol engine.

        Returns:
            A numpy array of samples of shape `(n x tunable_d)`.
        """
        if tunable_d == 0:
            return np.zeros((n, 0))
        if self.engine is None:
            raise ValueError(
                "Sobol Engine must be initialized before candidate generation."
            )
        return none_throws(self.engine).draw(n, dtype=torch.double).numpy()
