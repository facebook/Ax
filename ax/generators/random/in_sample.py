#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from ax.core.search_space import SearchSpaceDigest
from ax.generators.random.base import RandomGenerator
from ax.generators.types import TConfig


class InSampleUniformGenerator(RandomGenerator):
    """Randomly select candidates from existing experiment arms.

    Selects n arms uniformly at random without replacement from the
    ``generated_points`` array passed by the adapter. This array contains
    the in-design, non-failed arms on the experiment (deduplicated).

    Used for model-free candidate selection in use cases like LILO
    (Language-in-the-Loop Optimization), where a labeling node needs
    to randomly select previously observed configurations without
    fitting any surrogate model.

    See base ``RandomGenerator`` for a description of model attributes.
    """

    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        linear_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
        fixed_features: dict[int, float] | None = None,
        model_gen_options: TConfig | None = None,
        rounding_func: Callable[[npt.NDArray], npt.NDArray] | None = None,
        generated_points: npt.NDArray | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Select n candidates from ``generated_points``.

        Args:
            n: Number of candidates to select.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            linear_constraints: Not used. Accepted for interface compatibility.
            fixed_features: Not used. Accepted for interface compatibility.
            model_gen_options: Not used. Accepted for interface compatibility.
            rounding_func: Not used. Accepted for interface compatibility.
            generated_points: A numpy array of shape ``(num_arms, d)`` containing
                the existing experiment arms to select from. Constructed by the
                adapter from in-design, non-failed arms (deduplicated).

        Returns:
            2-element tuple containing

            - ``(n, d)`` array of selected points.
            - Uniform weights, an n-array of ones.

        Raises:
            ValueError: If ``generated_points`` is None or has fewer than
                ``n`` rows.
        """
        available = 0 if generated_points is None else len(generated_points)
        if generated_points is None or available < n:
            raise ValueError(
                f"Cannot select {n} arms: only {available} eligible "
                f"arms available on the experiment."
            )

        rng = np.random.default_rng(seed=self.seed + self.init_position)
        indices = rng.choice(len(generated_points), size=n, replace=False)
        self.init_position += n
        return generated_points[indices], np.ones(n)

    def _gen_samples(self, n: int, tunable_d: int, bounds: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError(
            "InSampleUniformGenerator selects from existing points "
            "and does not generate new samples."
        )
