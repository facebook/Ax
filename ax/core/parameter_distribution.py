# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase

if TYPE_CHECKING:
    from ax.core.search_space import RobustSearchSpace
    from scipy.stats.distributions import rv_frozen  # pyre-ignore [21]

TDistribution = str
TParamName = str


class ParameterDistribution(SortableBase):
    """A class for defining parameter distributions.

    Intended for robust optimization use cases. This could be used to specify the
    distribution of an environmental variable or the distribution of the input noise.
    """

    def __init__(
        self,
        parameters: List[TParamName],
        distribution_class: TDistribution,
        distribution_parameters: Optional[Dict[str, Any]],
        multiplicative: bool = False,
    ) -> None:
        """Initialize a parameter distribution.

        Args:
            parameters: A list of parameters, which the distribution belongs to. If
                this represents the joint input noise distribution of the parameters
                `x1` and `x2`, pass in `parameters = ["x1", "x2"]`, etc.
            distribution_class: The name of the scipy distribution class. This must be
                importable as `from scipy.stats import <distribution_class>`.
            distribution_parameters: A dictionary of keyword arguments for initializing
                the distribution class. The distribution will be initialized as
                `distribution = distribution_class(**distribution_parameters)`.
            multiplicative: A boolean denoting whether the distribution will be used as
                a multiplicative input perturbation. Should be `False` for the
                distributions of environmental variables.
        """
        super().__init__()
        self.parameters = parameters
        self._distribution_class = distribution_class
        self._distribution_parameters: Dict[str, Any] = distribution_parameters or {}
        self.multiplicative = multiplicative
        self._distribution: Optional[rv_frozen] = None  # pyre-ignore [11]

    @property
    def distribution_class(self) -> TDistribution:
        r"""The name of the scipy distribution class."""
        return self._distribution_class

    @distribution_class.setter
    def distribution_class(self, new_class: TDistribution) -> None:
        r"""Update the distribution class and delete the cached distribution object."""
        self._distribution = None
        self._distribution_class = new_class

    @property
    def distribution_parameters(self) -> Dict[str, Any]:
        r"""The parameters of the distribution."""
        return self._distribution_parameters

    @distribution_parameters.setter
    def distribution_parameters(self, new_parameters: Dict[str, Any]) -> None:
        r"""Update the distribution parameters and delete the cached
        distribution object.
        """
        self._distribution = None
        self._distribution_parameters = new_parameters

    def _construct_distribution_object(self) -> None:
        r"""Constructs the scipy distribution object."""
        stats = import_module("scipy.stats")
        try:
            dist_class = getattr(stats, self.distribution_class)
        except AttributeError:
            raise UserInputError(
                "Got an error while importing the distribution "
                f"{self.distribution_class}. Make sure that the "
                "`distribution_class` is importable from `scipy.stats`."
            )
        self._distribution = dist_class(**self.distribution_parameters)

    @property
    def distribution(self) -> rv_frozen:
        """Get the distribution object."""
        if self._distribution is None:
            self._construct_distribution_object()
        return self._distribution

    def is_environmental(self, search_space: RobustSearchSpace) -> bool:
        r"""Check if the parameters are environmental variables of the given
        search space.

        Args:
            search_space: The search space to check.

        Returns:
            A boolean denoting whether the parameters are environmental variables.
        """
        return any(search_space.is_environmental_variable(p) for p in self.parameters)

    def clone(self) -> ParameterDistribution:
        """Clone."""
        return ParameterDistribution(
            parameters=self.parameters.copy(),
            distribution_class=self.distribution_class,
            distribution_parameters=deepcopy(self.distribution_parameters),
            multiplicative=self.multiplicative,
        )

    def __hash__(self) -> int:
        """Make the class hashable to support the use of `lru_cache` above.

        NOTE: The hash of two `ParameterDistribution`s with identical attributes
        will be the same. This is compatible with the use in `lru_cache` above,
        since the resulting distributions will be the same.
        """
        return hash(repr(self))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "parameters=" + repr(self.parameters) + ", "
            "distribution_class=" + self.distribution_class + ", "
            "distribution_parameters=" + repr(self.distribution_parameters) + ", "
            "multiplicative=" + repr(self.multiplicative) + ")"
        )

    @property
    def _unique_id(self) -> str:
        return str(self)
