#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import TYPE_CHECKING

from ax.adapter.transforms.choice_encode import ChoiceToNumericChoice
from ax.core.parameter import ChoiceParameter, Parameter

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class TaskChoiceToIntTaskChoice(ChoiceToNumericChoice):
    """Convert task ChoiceParameters to integer-valued ChoiceParameters.

    Parameters will be transformed to an integer ChoiceParameter with
    property `is_task=True`, mapping values from the original choice domain to a
    contiguous range integers `0, 1, ..., n_choices-1`.

    In the inverse transform, parameters will be mapped back onto the original domain.

    Transform is done in-place.
    """

    def _should_encode(self, p: Parameter) -> bool:
        """Check if a parameter should be encoded.
        Encodes task choice parameters.
        Raises an error if the task parameter is also a fidelity parameter.
        """
        if isinstance(p, ChoiceParameter) and p.is_task:
            if p.is_fidelity:
                raise ValueError(
                    f"Task parameter {p.name} cannot simultaneously be "
                    "a fidelity parameter."
                )
            return True
        return False
