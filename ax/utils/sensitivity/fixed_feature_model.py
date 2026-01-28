# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from typing import Any

import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.posterior_list import PosteriorList
from torch import Tensor


class FixedFeatureModel:
    """Wrapper that fixes certain input dimensions at specified values for sensitivity.

    This wrapper is used to compute sensitivity analysis conditioned on fixed values
    for certain features (e.g., fixing the "step" feature at a target value).

    Note: This class does not inherit from Model to avoid torch.nn.Module's
    attribute handling complexity. It implements the posterior() interface needed
    by sensitivity analysis.
    """

    _model: Model
    fixed_indices: list[int]
    fixed_values: Tensor
    original_dim: int
    free_indices: list[int]

    def __init__(
        self,
        model: Model,
        fixed_feature_indices: list[int],
        fixed_feature_values: list[float],
        original_dim: int,
    ) -> None:
        """Initialize the fixed feature model wrapper.

        Args:
            model: The underlying Model.
            fixed_feature_indices: Indices of features to fix (in original space).
            fixed_feature_values: Values to fix those features at.
            original_dim: The original input dimensionality of the model.
        """
        self._model = model
        self.fixed_indices = fixed_feature_indices
        self.fixed_values = torch.tensor(fixed_feature_values)
        self.original_dim = original_dim
        self.free_indices = [
            i for i in range(original_dim) if i not in fixed_feature_indices
        ]

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: Any = None,
        **kwargs: Any,
    ) -> Posterior | PosteriorList:
        """Compute posterior by expanding X to include fixed features.

        Args:
            X: Input tensor of shape (n, d_free) where d_free is the number of
                non-fixed features.
            output_indices: Optional list of output indices to compute posterior for.
            observation_noise: If True, add observation noise to posterior.
            posterior_transform: Optional posterior transform to apply.
            **kwargs: Additional arguments passed to the underlying model.

        Returns:
            Posterior from the underlying model.
        """
        full_X = self._expand_X(X)
        return self._model.posterior(
            full_X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
            **kwargs,
        )

    def _expand_X(self, X: Tensor) -> Tensor:
        """Expand X to include fixed feature values.

        Args:
            X: Input tensor of shape (..., d_free).

        Returns:
            Tensor of shape (..., original_dim) with fixed values inserted.
        """
        # Create output tensor with the original dimensionality
        shape = list(X.shape[:-1]) + [self.original_dim]
        full_X = torch.empty(shape, dtype=X.dtype, device=X.device)

        # Move fixed values to the correct device and dtype
        fixed_vals = self.fixed_values.to(X)

        # Fill in free features from X (vectorized assignment)
        full_X[..., self.free_indices] = X

        # Fill in fixed features with their fixed values
        for idx, val in zip(self.fixed_indices, fixed_vals):
            full_X[..., idx] = val

        return full_X


def prepare_fixed_feature_inputs(
    model_list: Sequence[Model],
    bounds: Tensor,
    discrete_features: list[int] | None,
    fixed_features: dict[int, float],
) -> tuple[list[FixedFeatureModel], Tensor, list[int] | None]:
    """Prepare wrapped models and reduced bounds for fixed feature sensitivity analysis.

    This helper function handles the common logic for fixing features during
    sensitivity analysis. It wraps models with FixedFeatureModel, reduces the
    bounds to only include free features, and remaps discrete feature indices.

    Args:
        model_list: A list of models to wrap.
        bounds: A 2 x d Tensor of lower and upper bounds.
        discrete_features: Indices of discrete features (will be remapped).
        fixed_features: A dictionary mapping feature indices to fixed values.

    Returns:
        A tuple of (wrapped_models, reduced_bounds, reduced_discrete_features).
    """
    original_dim = bounds.shape[1]
    fixed_indices = sorted(fixed_features.keys())
    fixed_values = [fixed_features[i] for i in fixed_indices]
    free_indices = [i for i in range(original_dim) if i not in fixed_indices]

    # Reduce bounds to only include free features
    reduced_bounds = bounds[:, free_indices]

    # Adjust discrete_features indices to account for removed dimensions
    reduced_discrete_features: list[int] | None = None
    if discrete_features is not None:
        old_to_new = {old: new for new, old in enumerate(free_indices)}
        reduced_discrete_features = [
            old_to_new[i] for i in discrete_features if i in old_to_new
        ]

    # Wrap models to fix the specified features
    wrapped_models = [
        FixedFeatureModel(
            model=model,
            fixed_feature_indices=fixed_indices,
            fixed_feature_values=fixed_values,
            original_dim=original_dim,
        )
        for model in model_list
    ]

    return wrapped_models, reduced_bounds, reduced_discrete_features
