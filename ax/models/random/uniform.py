#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import numpy as np
from ax.models.random.base import RandomModel
from scipy.stats import uniform


class UniformGenerator(RandomModel):
    """This class specifies a uniform random generation algorithm.

    As a uniform generator does not make use of a model, it does not implement
    the fit or predict methods.

    See base `RandomModel` for a description of model attributes.
    """

    def __init__(
        self,
        deduplicate: bool = True,
        seed: Optional[int] = None,
        generated_points: Optional[np.ndarray] = None,
        fallback_to_sample_polytope: bool = False,
    ) -> None:
        super().__init__(
            deduplicate=deduplicate,
            seed=seed,
            generated_points=generated_points,
            fallback_to_sample_polytope=fallback_to_sample_polytope,
        )
        self._rs = np.random.RandomState(seed=self.seed)

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate samples from the scipy uniform distribution.

        Args:
            n: Number of samples to generate.
            tunable_d: Dimension of samples to generate.

        Returns:
            samples: An (n x d) array of random points.

        """
        return uniform.rvs(size=(n, tunable_d), random_state=self._rs)  # pyre-ignore
