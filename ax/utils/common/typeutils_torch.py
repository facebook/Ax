#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Union

import torch
from ax.utils.common.typeutils import checked_cast


def torch_type_to_str(value: Union[torch.dtype, torch.device, torch.Size]) -> str:
    """Converts torch types, commonly used in Ax, to string representations."""
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return checked_cast(str, value.type)
    if isinstance(value, torch.Size):
        return json.dumps(list(value))
    raise ValueError(f"Object {value} was of unexpected torch type.")


def torch_type_from_str(
    identifier: str, type_name: str
) -> Union[torch.dtype, torch.device, torch.Size]:
    if type_name == "device":
        return torch.device(identifier)
    if type_name == "dtype":
        return getattr(torch, identifier[6:])
    if type_name == "Size":
        return torch.Size(json.loads(identifier))
    raise ValueError(f"Unexpected type: {type_name} for identifier: {identifier}.")
