#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from ax.core.types import TConfig
from ax.models.random.uniform import UniformGenerator
from ax.utils.common.docutils import copy_doc


class ALEBOInitializer(UniformGenerator):
    """Sample in a low-dimensional linear embedding, to initialize ALEBO.

    Generates points on a linear subspace of [-1, 1]^D by generating points in
    [-b, b]^D, projecting them down with a matrix B, and then projecting them
    back up with the pseudoinverse of B. Thus points thus all lie in a linear
    subspace defined by B. Points whose up-projection falls outside of [-1, 1]^D
    are thrown out, via rejection sampling.

    To generate n points, we start with nsamp points in [-b, b]^D, which are
    mapped down to the embedding and back up as described above. If >=n points
    fall within [-1, 1]^D after being mapped up, then the first n are returned.
    If there are less than n points in [-1, 1]^D, then b is constricted
    (halved) and the process is repeated until there are at least n points in
    [-1, 1]^D. There exists a b small enough that all points will project to
    [-1, 1]^D, so this is guaranteed to terminate, typically after few rounds.

    Args:
        B: A (dxD) projection down.
        nsamp: Number of samples to use for rejection sampling.
        init_bound: b for the initial sampling space described above.
        kwargs: kwargs for UniformGenerator
    """

    def __init__(
        self, B: np.ndarray, nsamp: int = 10000, init_bound: int = 16, **kwargs: Any
    ) -> None:
        self.Q = np.linalg.pinv(B) @ B  # Projects down to B and then back up
        self.nsamp = nsamp
        self.init_bound = init_bound
        super().__init__(**kwargs)

    @copy_doc(UniformGenerator.gen)
    def gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if n > self.nsamp:
            raise ValueError("n > nsamp")
        # The projection is from [-1, 1]^D.
        for b in bounds:
            assert b == (-1.0, 1.0)
        # The following can be easily handled in the future when needed
        assert linear_constraints is None
        assert fixed_features is None
        # Do gen in the high-dimensional space.
        X01, w = super().gen(
            n=self.nsamp,
            bounds=[(0.0, 1.0)] * self.Q.shape[0],
            model_gen_options={"max_rs_draws": self.nsamp},
        )
        finished = False
        b = float(self.init_bound)
        while not finished:
            # Map to [-b, b]
            X_b = 2 * b * X01 - b
            # Project down to B and back up
            X = X_b @ np.transpose(self.Q)
            # Filter out to points in [-1, 1]^D
            X = X[(X >= -1.0).all(axis=1) & (X <= 1.0).all(axis=1)]
            if X.shape[0] >= n:
                finished = True
            else:
                b = b / 2.0  # Constrict the space
        X = X[:n, :]
        return X, np.ones(n)
