# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ax.api.types import TParameterValue
from ax.storage.registry_bundle import RegistryBundleBase


@dataclass
class RangeParameterConfig:
    """
    Allows specifying a continuous dimension of an experiment's search space
    and internally validates the inputs.
    """

    name: str

    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None


@dataclass
class ChoiceParameterConfig:
    """
    Allows specifying a discrete dimension of an experiment's search space and
    internally validates the inputs. Choice parameters can be either ordinal or
    categorical; this is controlled via the ``is_ordered`` flag.
    """

    name: str
    values: list[float] | list[int] | list[str] | list[bool]
    parameter_type: Literal["float", "int", "str", "bool"]
    is_ordered: bool | None = None
    dependent_parameters: Mapping[TParameterValue, Sequence[str]] | None = None


@dataclass
class StorageConfig:
    """
    Allows the user to configure how Ax should connect to a SQL database to store the
    experiment and its data.
    """

    creator: Callable[..., Any] | None = None  # pyre-fixme[4]
    url: str | None = None
    registry_bundle: RegistryBundleBase | None = None
