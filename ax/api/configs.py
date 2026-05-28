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

    ``step_size`` and ``digits`` are two distinct ways to express limited
    resolution and are mutually exclusive:

    - ``step_size`` materializes an ordered ``ChoiceParameter`` over the
      explicit grid ``[lower, lower + step, ..., upper]``. The optimizer
      treats the parameter as discrete (with nearest-neighbor acquisition
      optimization for low-cardinality grids). Use this when the domain
      *is* the grid -- e.g. a small set of hardware settings. Requires
      linear scaling and ``(upper - lower) % step_size == 0``. Capped at
      1000 grid points; use ``digits`` for finer resolutions.
    - ``digits`` keeps the parameter as a continuous ``RangeParameter``
      and rounds suggested values to ``digits`` decimal places at output
      time. The optimizer treats the parameter as continuous. Use this
      when the underlying quantity is continuous but the
      measurement/control resolution is limited -- e.g. a laser intensity
      dialed in 0.1 W increments. Float-only; works with log/logit
      scaling and has no cardinality cap.
    """

    name: str

    bounds: tuple[float, float]
    parameter_type: Literal["float", "int"]
    step_size: float | None = None
    scaling: Literal["linear", "log"] | None = None
    digits: int | None = None


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
class DerivedParameterConfig:
    """
    Allows specifying a dimension of an experiment's search space and that is
    derived from other parameters and internally validates the inputs.
    """

    name: str
    expression_str: str
    parameter_type: Literal["float", "int", "str", "bool"]


@dataclass
class StorageConfig:
    """
    Allows the user to configure how Ax should connect to a SQL database to store the
    experiment and its data.
    """

    creator: Callable[..., Any] | None = None
    url: str | None = None
    registry_bundle: RegistryBundleBase | None = None
