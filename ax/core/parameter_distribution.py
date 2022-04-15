# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, List, Optional

from ax.exceptions.core import UserInputError
from ax.utils.common.base import Base
from scipy.stats._distn_infrastructure import rv_generic

TDistribution = str
TParamName = str


class ParameterDistribution(Base):
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
        self.distribution_class = distribution_class
        self.distribution_parameters = distribution_parameters or {}
        self.multiplicative = multiplicative

    @property
    @functools.lru_cache()
    def distribution(self) -> rv_generic:
        """Get the distribution object."""
        stats = import_module("scipy.stats")
        try:
            dist_class = getattr(stats, self.distribution_class)
        except AttributeError:
            raise UserInputError(
                "Got an error while importing the distribution "
                f"{self.distribution_class}. Make sure that the "
                "`distribution_class` is importable from `scipy.stats`."
            )
        return dist_class(**self.distribution_parameters)

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
