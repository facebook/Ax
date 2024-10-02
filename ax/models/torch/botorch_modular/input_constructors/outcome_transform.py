# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from __future__ import annotations

from typing import Any

from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.dispatcher import Dispatcher

outcome_transform_argparse = Dispatcher(
    name="outcome_transform_argparse", encoder=_argparse_type_encoder
)


@outcome_transform_argparse.register(OutcomeTransform)
def _outcome_transform_argparse_base(
    outcome_transform_class: type[OutcomeTransform],
    dataset: SupervisedDataset | None = None,
    outcome_transform_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract the outcome transform kwargs from the given arguments.

    Args:
        outcome_transform_class: Outcome transform class.
        dataset: Dataset containing feature matrix and the response.
        outcome_transform_options: An optional dictionary of outcome transform options.
            This may include overrides for the above options. For example, when
            `outcome_transform_class` is Standardize this dictionary might include
            {
                "m": 1, # the output dimension
            }
            See `botorch/models/transforms/outcome.py` for more options.

    Returns:
        A dictionary with outcome transform kwargs.
    """
    return outcome_transform_options or {}


@outcome_transform_argparse.register(Standardize)
def _outcome_transform_argparse_standardize(
    outcome_transform_class: type[Standardize],
    dataset: SupervisedDataset,
    outcome_transform_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the outcome transform kwargs form the given arguments.

    Args:
        outcome_transform_class: Outcome transform class, which is Standardize in this
            case.
        dataset: Dataset containing feature matrix and the response.
        outcome_transform_options: Outcome transform kwargs.
            See botorch.models.transforms.outcome.Standardize for all available options

    Returns:
        A dictionary with outcome transform kwargs.
    """

    outcome_transform_options = outcome_transform_options or {}
    m = dataset.Y.shape[-1]
    outcome_transform_options.setdefault("m", m)

    return outcome_transform_options
