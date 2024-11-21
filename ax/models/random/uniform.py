#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
import numpy.typing as npt
from ax.models.random.base import RandomModel


class UniformGenerator(RandomModel):
    """This class specifies a uniform random generation algorithm.

    As a uniform generator does not make use of a model, it does not implement
    the fit or predict methods.

    See base `RandomModel` for a description of model attributes.
    """

    def __init__(
        self,
        deduplicate: bool = True,
        seed: int | None = None,
        init_position: int = 0,
        generated_points: npt.NDArray | None = None,
        fallback_to_sample_polytope: bool = False,
    ) -> None:
        super().__init__(
            deduplicate=deduplicate,
            seed=seed,
            init_position=init_position,
            generated_points=generated_points,
            fallback_to_sample_polytope=fallback_to_sample_polytope,
        )
        self._rs = np.random.RandomState(seed=self.seed)
        if self.init_position > 0:
            # Fast-forward the random state by generating & discarding samples.
            self._rs.uniform(size=(self.init_position))

    def _gen_samples(self, n: int, tunable_d: int) -> npt.NDArray:
        """Generate samples from the scipy uniform distribution.

        Args:
            n: Number of samples to generate.
            tunable_d: Dimension of samples to generate.

        Returns:
            samples: An (n x d) array of random points.

        """
        self.init_position += n * tunable_d
        return self._rs.uniform(size=(n, tunable_d))
