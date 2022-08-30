#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Union

import torch
from ax.utils.common.typeutils import checked_cast


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def torch_type_to_str(value: Any) -> str:
    """Converts torch types, commonly used in Ax, to string representations."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return checked_cast(str, value.type)
    raise ValueError(f"Object {value} was of unexpected torch type.")


def torch_type_from_str(
    identifier: str, type_name: str
) -> Union[torch.dtype, torch.device]:
    if type_name == "device":
        return torch.device(identifier)
    if type_name == "dtype":
        return getattr(torch, identifier[6:])
    raise ValueError(f"Unexpected type: {type_name} for identifier: {identifier}.")
