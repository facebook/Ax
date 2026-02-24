#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections.abc import Callable
from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import SearchSpaceExhausted
from ax.generators.base import Generator
from ax.generators.types import TConfig
from ax.generators.utils import (
    add_fixed_features,
    DEFAULT_MAX_RS_DRAWS,
    rejection_sample,
    tunable_feature_indices,
)
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import assert_is_instance_of_tuple
from botorch.utils.sampling import HitAndRunPolytopeSampler
from pyre_extensions import assert_is_instance
from torch import Tensor


logger: Logger = get_logger(__name__)


class RandomGenerator(Generator):
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
        seed: An optional seed value for scrambling.
        init_position: The initial state of the generator. This is the number
            of samples to fast-forward before generating new samples.
            Used to ensure that the re-loaded generator will continue generating
            from the same sequence rather than starting from scratch.
        generated_points: A set of previously generated points to use
            for deduplication. These should be provided in the raw transformed
            space the model operates in.
        fallback_to_sample_polytope: If True, when rejection sampling fails,
            we fall back to the HitAndRunPolytopeSampler.
    """

    def __init__(
        self,
        deduplicate: bool = True,
        seed: int | None = None,
        init_position: int = 0,
        generated_points: npt.NDArray | None = None,
        fallback_to_sample_polytope: bool = False,
        polytope_sampler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.deduplicate = deduplicate
        self.seed: int = (
            seed
            if seed is not None
            else assert_is_instance(torch.randint(high=100_000, size=(1,)).item(), int)
        )
        self.init_position = init_position
        # Used for deduplication.
        self.fallback_to_sample_polytope = fallback_to_sample_polytope
        self.polytope_sampler_kwargs: dict[str, Any] = polytope_sampler_kwargs or {}
        self._bounds: npt.NDArray = np.empty((0, 2))
        self.attempted_draws: int = 0
        if generated_points is not None:
            # generated_points was deprecated in Ax 1.0.0, so it can now be reaped.
            warnings.warn(
                "The `generated_points` argument is deprecated and will be removed "
                "in a future version of Ax. It is being ignored in favor of "
                "extracting the generated points directly from the experiment.",
                DeprecationWarning,
                stacklevel=2,
            )

    @property
    def can_predict(self) -> bool:
        """Whether this generator can predict outcomes for new parameterizations."""
        return False

    @property
    def can_model_in_sample(self) -> bool:
        """Whether this generator can model (e.g. apply shrinkage) on observed
        parameterizations (in this case, it needs to support calling `predict`()
        on points in the training data / provided during `fit()`)."""
        return False

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
            model_gen_options: A config dictionary that is passed along to the
                model.
            rounding_func: A function that rounds an optimization result
                appropriately (e.g., according to `round-trip` transformations).
            generated_points: A numpy array of shape `n x d` containing the
                previously generated points to deduplicate against.

        Returns:
            2-element tuple containing

            - (n x d) array of generated points.
            - Uniform weights, an n-array of ones for each point.

        """
        tf_indices = tunable_feature_indices(
            bounds=search_space_digest.bounds, fixed_features=fixed_features
        )
        self._bounds = np.array(search_space_digest.bounds)  # (d, 2)

        max_draws = DEFAULT_MAX_RS_DRAWS
        discrete_indices = set(search_space_digest.discrete_choices.keys())
        continuous_indices = {
            i
            for i in range(len(search_space_digest.feature_names))
            if i not in discrete_indices
        }
        if fixed_features is not None:
            for i in fixed_features.keys():
                if i in continuous_indices:
                    continuous_indices.remove(i)
        has_continuous_parameters = len(continuous_indices) > 0
        if model_gen_options:
            max_draws = model_gen_options.get("max_rs_draws", DEFAULT_MAX_RS_DRAWS)
            max_draws = int(assert_is_instance_of_tuple(max_draws, (int, float)))
        try:
            # Always rejection sample, but this only rejects if there are
            # constraints or actual duplicates and deduplicate is specified.
            # If rejection sampling fails, fall back to polytope sampling.
            points, attempted_draws = rejection_sample(
                gen_unconstrained=self._gen_unconstrained,
                n=n,
                d=len(search_space_digest.bounds),
                tunable_feature_indices=tf_indices,
                linear_constraints=linear_constraints,
                deduplicate=self.deduplicate,
                max_draws=max_draws,
                fixed_features=fixed_features,
                rounding_func=rounding_func,
                existing_points=generated_points,
            )
        except SearchSpaceExhausted as e:
            if has_continuous_parameters or self.fallback_to_sample_polytope:
                logger.warning(
                    "Parameter constraints are very restrictive, this makes "
                    "candidate generation difficult. "
                    "(Rejection sampling exceeded specified maximum draws). "
                )
                logger.debug(
                    "Falling back on HitAndRunPolytopeSampler instead of "
                    f"{self.__class__.__name__}."
                )
                # If rejection sampling fails, try polytope sampler.
                num_generated = (
                    len(generated_points) if generated_points is not None else 0
                )
                interior_point = (  # A feasible point of shape `d x 1`.
                    torch.from_numpy(generated_points[-1].reshape((-1, 1))).double()
                    if generated_points is not None
                    else None
                )
                kwargs = {"n_burnin": 100, "n_thinning": 20}
                kwargs.update(self.polytope_sampler_kwargs)
                polytope_sampler: HitAndRunPolytopeSampler = HitAndRunPolytopeSampler(
                    inequality_constraints=self._convert_inequality_constraints(
                        linear_constraints,
                    ),
                    equality_constraints=self._convert_equality_constraints(
                        d=len(search_space_digest.bounds),
                        fixed_features=fixed_features,
                    ),
                    bounds=self._convert_bounds(bounds=search_space_digest.bounds),
                    interior_point=interior_point,
                    seed=self.seed + num_generated,
                    **kwargs,
                )

                def gen_polytope_sampler(
                    n: int,
                    d: int,
                    tunable_feature_indices: npt.NDArray,
                    fixed_features: dict[int, float] | None = None,
                ) -> npt.NDArray:
                    # Note: the fixed features are applied as equality constraints
                    # in the polytope sampler, so we don't need to apply them here.
                    return polytope_sampler.draw(n=n).numpy()

                # we call rejection_sample here to reuse all the deduplication
                # logic
                points, _ = rejection_sample(
                    gen_unconstrained=gen_polytope_sampler,
                    n=n,
                    d=len(search_space_digest.bounds),
                    tunable_feature_indices=tf_indices,
                    linear_constraints=linear_constraints,
                    deduplicate=self.deduplicate,
                    max_draws=max_draws,
                    fixed_features=fixed_features,
                    rounding_func=rounding_func,
                    existing_points=generated_points,
                )
            else:
                raise e

        return points, np.ones(len(points))

    @copy_doc(Generator._get_state)
    def _get_state(self) -> dict[str, Any]:
        state = super()._get_state()
        state.update(
            {
                "seed": self.seed,
                "init_position": self.init_position,
            }
        )
        return state

    def _gen_unconstrained(
        self,
        n: int,
        d: int,
        tunable_feature_indices: npt.NDArray,
        fixed_features: dict[int, float] | None = None,
    ) -> npt.NDArray:
        """Generate n points, from an unconstrained parameter space, using _gen_samples.

        Args:
            n: Number of points to generate.
            d: Dimension of parameter space.
            tunable_feature_indices: Parameter indices (in d) which are tunable.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.

        Returns:
            An (n x d) array of generated points.

        """
        tunable_bounds = self._bounds[tunable_feature_indices]
        tunable_points = self._gen_samples(
            n=n,
            tunable_d=len(tunable_feature_indices),
            bounds=tunable_bounds,
        )
        return add_fixed_features(
            tunable_points=tunable_points,
            d=d,
            tunable_feature_indices=tunable_feature_indices,
            fixed_features=fixed_features,
        )

    def _gen_samples(self, n: int, tunable_d: int, bounds: npt.NDArray) -> npt.NDArray:
        """Generate n samples within the given bounds.

        Args:
            n: Number of points to generate.
            tunable_d: Number of tunable dimensions.
            bounds: A (tunable_d x 2) array of (lower, upper) bounds.

        Returns:
            (n x tunable_d) array of generated points.

        """
        raise NotImplementedError("Base RandomGenerator can't generate samples.")

    def _convert_inequality_constraints(
        self,
        linear_constraints: tuple[npt.NDArray, npt.NDArray] | None,
    ) -> tuple[Tensor, Tensor] | None:
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
            A = torch.as_tensor(linear_constraints[0], dtype=torch.double)
            b = torch.as_tensor(linear_constraints[1], dtype=torch.double)
            return A, b

    def _convert_equality_constraints(
        self, d: int, fixed_features: dict[int, float] | None
    ) -> tuple[Tensor, Tensor] | None:
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

        n = len(fixed_features)
        fixed_indices = sorted(fixed_features.keys())
        fixed_vals = torch.tensor(
            [fixed_features[i] for i in fixed_indices], dtype=torch.double
        )
        constraint_matrix = torch.zeros((n, d), dtype=torch.double)
        for index in range(0, len(fixed_vals)):
            constraint_matrix[index, fixed_indices[index]] = 1.0
        return constraint_matrix, fixed_vals

    def _convert_bounds(self, bounds: list[tuple[float, float]]) -> Tensor | None:
        """Helper method to convert bounds list used by the rejectionsampler to the
        tensor format required for the polytope sampler.

            Args:
                bounds: A list of (lower, upper) tuples for each column of X.

            Returns:
                Optional 2 x d tensor representing the bounds
        """
        if bounds is None:
            return None
        else:
            return torch.tensor(bounds, dtype=torch.double).transpose(-1, -2)
