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


class REMBOInitializer(UniformGenerator):
    """Sample in a low-dimensional linear embedding.

    Generates points in [-1, 1]^D by generating points in a d-dimensional
    embedding, with box bounds as specified. When points are projected up, if
    they fall outside [-1, 1]^D they are clamped to those bounds.

    Args:
        A: A (Dxd) linear embedding
        bounds_d: Box bounds in the low-d space
        kwargs: kwargs for UniformGenerator
    """

    def __init__(
        self, A: np.ndarray, bounds_d: List[Tuple[float, float]], **kwargs: Any
    ) -> None:
        self.bounds_d = bounds_d
        self.A = A
        self.X_d_gen = []  # Store points in low-d space generated here
        super().__init__(**kwargs)

    def project_up(self, X: np.ndarray) -> np.ndarray:
        """Project to high-dimensional space.
        """
        Z = np.transpose(self.A @ np.transpose(X))
        return np.clip(Z, a_min=-1, a_max=1)

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
        # The projection is from [-1, 1]^D.
        for b in bounds:
            assert b == (-1, 1)
        # The following can be easily handled in the future when needed
        assert linear_constraints is None
        assert fixed_features is None
        # Do gen in the low-dimensional space. First on [0, 1]^d,
        X_01, w = super().gen(n=n, bounds=[(0.0, 1.0)] * len(self.bounds_d))
        # Then map to bounds_d
        lw, up = zip(*self.bounds_d)
        lw = np.array(lw)
        up = np.array(up)
        X_d = X_01 * (up - lw) + lw
        # Store
        self.X_d_gen.extend(list(X_d))
        # And finally project up
        return self.project_up(X_d), w
