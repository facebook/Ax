#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from ax.exceptions.core import SearchSpaceExhausted
from ax.models.base import Model
from ax.models.model_utils import (
    add_fixed_features,
    rejection_sample,
    tunable_feature_indices,
    validate_bounds,
)
from ax.models.types import TConfig
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.utils.sampling import HitAndRunPolytopeSampler
from torch import Tensor


logger: Logger = get_logger(__name__)


class RandomModel(Model):
    """This class specifies the basic skeleton for a random model.

    As random generators do not make use of models, they do not implement
    the fit or predict methods.

    These models do not need data, or optimization configs.

    To satisfy search space parameter constraints, these models can use
    rejection sampling. To enable rejection sampling for a subclass, only
    only `_gen_samples` needs to be implemented, or alternatively,
    `_gen_unconstrained`/`gen` can be directly implemented.

    Attributes:
        deduplicate: If True (defaults to True), a single instantiation
            of the model will not return the same point twice. This flag
            is used in rejection sampling.
        scramble: If True, permutes the parameter values among
            the elements of the Sobol sequence. Default is True.
        seed: An optional seed value for scrambling.
    """

    def __init__(
        self,
        deduplicate: bool = True,
        seed: Optional[int] = None,
        generated_points: Optional[np.ndarray] = None,
        fallback_to_sample_polytope: bool = False,
    ) -> None:
        super().__init__()
        self.deduplicate = deduplicate
        self.seed = seed
        # Used for deduplication.
        self.generated_points = generated_points
        self.fallback_to_sample_polytope = fallback_to_sample_polytope

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
                Defined on [0, 1]^d.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            model_gen_options: A config dictionary that is passed along to the
                model.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations).

        Returns:
            2-element tuple containing

            - (n x d) array of generated points.
            - Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=bounds, fixed_features=fixed_features
        )
        if fixed_features:
            fixed_feature_indices = np.array(list(fixed_features.keys()))
        else:
            fixed_feature_indices = np.array([])

        validate_bounds(bounds=bounds, fixed_feature_indices=fixed_feature_indices)
        attempted_draws = 0
        max_draws = None
        if model_gen_options:
            max_draws = model_gen_options.get("max_rs_draws")
            if max_draws is not None:
                # pyre-fixme[6]: Expected `Union[bytes, str, typing.SupportsInt]`
                #  for 1st param but got
                #  `Union[botorch.acquisition.acquisition.AcquisitionFunction, float,
                #  int, str]`.
                # pyre-fixme[35]: Target cannot be annotated.
                max_draws: int = int(max_draws)
        try:
            # Always rejection sample, but this only rejects if there are
            # constraints or actual duplicates and deduplicate is specified.
            # If rejection sampling fails, fall back to polytope sampling
            points, attempted_draws = rejection_sample(
                gen_unconstrained=self._gen_unconstrained,
                n=n,
                d=len(bounds),
                tunable_feature_indices=tf_indices,
                linear_constraints=linear_constraints,
                deduplicate=self.deduplicate,
                max_draws=max_draws,
                fixed_features=fixed_features,
                rounding_func=rounding_func,
                existing_points=self.generated_points,
            )
        except SearchSpaceExhausted as e:
            if self.fallback_to_sample_polytope:
                logger.info(
                    "Rejection sampling exceeded specified maximum draws."
                    "Falling back on polytope sampler"
                )
                # If rejection sampling fails, try polytope sampler.
                polytope_sampler = HitAndRunPolytopeSampler(
                    inequality_constraints=self._convert_inequality_constraints(
                        linear_constraints
                    ),
                    equality_constraints=self._convert_equality_constraints(
                        len(bounds), fixed_features
                    ),
                    interior_point=self._get_last_point(),
                    bounds=self._convert_bounds(bounds),
                )
                points = polytope_sampler.draw(n).numpy()
            else:
                raise e

        # pyre-fixme[16]: `RandomModel` has no attribute `attempted_draws`.
        self.attempted_draws = attempted_draws
        if self.deduplicate:
            if self.generated_points is None:
                self.generated_points = points
            else:
                self.generated_points = np.vstack([self.generated_points, points])
        return points, np.ones(len(points))

    @copy_doc(Model._get_state)
    def _get_state(self) -> Dict[str, Any]:
        state = super()._get_state()
        if not self.deduplicate:
            return state
        state.update({"generated_points": self.generated_points})
        return state

    def _gen_unconstrained(
        self,
        n: int,
        d: int,
        tunable_feature_indices: np.ndarray,
        fixed_features: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Generate n points, from an unconstrained parameter space, using _gen_samples.

        Args:
            n: Number of points to generate.
            d: Dimension of parameter space.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            tunable_feature_indices: Parameter indices (in d) which are tunable.

        Returns:
            An (n x d) array of generated points.

        """
        tunable_points = self._gen_samples(n=n, tunable_d=len(tunable_feature_indices))
        points = add_fixed_features(
            tunable_points=tunable_points,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )
        return points

    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate n samples on [0, 1]^d.

        Args:
            n: Number of points to generate.

        Returns:
            (n x d) array of generated points.

        """
        raise NotImplementedError("Base RandomModel can't generate samples.")

    def _convert_inequality_constraints(
        self, linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Helper method to convert inequality constraints used by the rejection
        sampler to the format required for the polytope sampler.

            Args:
                linear_constraints: A tuple of (A, b). For k linear constraints on
                    d-dimensional x, A is (k x d) and b is (k x 1) such that
                    A x <= b.

            Returns:
                Optional 2-element tuple containing A and b as tensors
        """
        if linear_constraints is None:
            return None
        else:
            A = torch.tensor(linear_constraints[0], dtype=torch.double)
            b = torch.tensor(linear_constraints[1], dtype=torch.double)
            return A, b

    def _convert_equality_constraints(
        self, d: int, fixed_features: Optional[Dict[int, float]]
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Helper method to convert the fixed feature dictionary used by the rejection
        sampler to the corresponding matrix representation required for the polytope
        sampler.

            Args:
                d: dimension of samples
                fixed_features: A map {feature_index: value} for features that
                    should be fixed to a particular value during generation.

            Returns:
                Optional 2-element tuple containing C and c such that the equality
                constraints are defined by Cx = c
        """
        if fixed_features is None:
            return None
        else:
            n = len(fixed_features)
            fixed_indices = sorted(fixed_features.keys())
            fixed_vals = torch.tensor(
                [fixed_features[i] for i in fixed_indices], dtype=torch.double
            )
            constraint_matrix = torch.zeros((n, d), dtype=torch.double)
            for index in range(0, len(fixed_vals)):
                constraint_matrix[index, fixed_indices[index]] = 1.0
            return constraint_matrix, fixed_vals

    def _convert_bounds(self, bounds: List[Tuple[float, float]]) -> Optional[Tensor]:
        """Helper method to convert bounds list used by the rejectionsampler to the
        tensor format required for the polytope sampler.

            Args:
                bounds: A list of (lower, upper) tuples for each column of X.
                    Defined on [0, 1]^d.

            Returns:
                Optional 2 x d tensor representing the bounds
        """
        if bounds is None:
            return None
        else:
            return torch.tensor(bounds, dtype=torch.double).transpose(-1, -2)

    def _get_last_point(self) -> Optional[Tensor]:
        # Return the last sampled point when points have been sampled
        if self.generated_points is None:
            return None
        else:
            last_point = self.generated_points[-1, :].reshape((-1, 1))
            return torch.from_numpy(last_point).double()
