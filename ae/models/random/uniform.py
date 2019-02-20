#!/usr/bin/env python3

from typing import Optional

import numpy as np
from ae.lazarus.ae.models.random.base import RandomModel
from scipy.stats import uniform


class UniformGenerator(RandomModel):
    """This class specifies a uniform random generation algorithm.

    As a uniform generator does not make use of a model, it does not implement
    the fit or predict methods.

    Attributes:
        seed: An optional seed value for the underlying PRNG.

    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)
        self._rs = np.random.RandomState(seed=seed)

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate samples from the scipy uniform distribution.

        Args:
            n: Number of samples to generate.
            tunable_d: Dimension of samples to generate.

        Returns:
            samples: An (n x d) array of random points.

        """
        return uniform.rvs(size=(n, tunable_d), random_state=self._rs)  # pyre-ignore
