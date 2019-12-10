#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
import json
from typing import Optional

from ax.core.base import Base
from ax.core.types import TParameterization
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.typeutils import numpy_type_to_python_type


class Arm(Base):
    """Base class for defining arms.

    Randomization in experiments assigns units to a given arm. Thus, the arm
    encapsulates the parametrization needed by the unit.

    """

    def __init__(
        self, parameters: TParameterization, name: Optional[str] = None
    ) -> None:
        """Inits Arm.

        Args:
            parameters: Mapping from parameter names to values.
            name: Defaults to None; will be set when arm is attached to a trial
        """
        self._parameters: TParameterization = _numpy_types_to_python_types(parameters)
        self._name = name

    @property
    def parameters(self) -> TParameterization:
        """Get mapping from parameter names to values."""
        # Make a copy before returning so it cannot be accidentally mutated
        return dict(self._parameters)

    @property
    def has_name(self) -> bool:
        """Return true if arm's name is not None."""
        return self._name is not None

    @property
    def name(self) -> str:
        """Get arm name. Throws if name is None."""
        if self._name is None:
            raise ValueError("Arm's name is None.")
        # pyre-fixme[7]: Expected `str` but got `Optional[str]`.
        return self._name

    @property
    def name_or_short_signature(self) -> str:
        """Returns arm name if exists; else last 4 characters of the hash.

        Used for presentation of candidates (e.g. plotting and tables),
        where the candidates do not yet have names (since names are
        automatically set upon addition to a trial).

        """
        return self._name or self.signature[-4:]

    @name.setter
    def name(self, name: str) -> None:
        if self._name is not None:
            raise ValueError("Arm name is not mutable once set.")
        self._name = name

    @property
    def signature(self) -> str:
        """Get unique representation of a arm."""
        return self.md5hash(self.parameters)

    @staticmethod
    def md5hash(parameters: TParameterization) -> str:
        """Return unique identifier for arm's parameters.

        Args:
            parameters: Parameterization; mapping of param name
                to value.

        Returns:
            Hash of arm's parameters.

        """
        for k, v in parameters.items():
            parameters[k] = numpy_type_to_python_type(v)
        parameters_str = json.dumps(parameters, sort_keys=True)
        return hashlib.md5(parameters_str.encode("utf-8")).hexdigest()

    def clone(self, clear_name: bool = False) -> "Arm":
        """Create a copy of this arm.

        Args:
            clear_name: whether this cloned copy should set its
                name to None instead of the name of the arm being cloned.
                Defaults to False.
        """
        clear_name = clear_name or not self.has_name
        return Arm(
            parameters=self.parameters.copy(), name=None if clear_name else self.name
        )

    def __repr__(self) -> str:
        parameters_str = f"parameters={self._parameters}"
        if self.has_name:
            name_str = f"name='{self.name}'"
            return f"Arm({name_str}, {parameters_str})"
        return f"Arm({parameters_str})"

    @equality_typechecker
    def __eq__(self, other: "Arm") -> bool:
        """Need to overwrite the default __eq__ method of Base,
        because accessing the "name" attribute of Arm
        can result in an error.
        """
        parameters_equal = self.parameters == other.parameters
        names_equal = self.has_name == other.has_name
        if names_equal and self.has_name:
            names_equal = self.name == other.name
        return parameters_equal and names_equal

    def __hash__(self) -> int:
        return int(self.signature, 16)


def _numpy_types_to_python_types(
    parameterization: TParameterization,
) -> TParameterization:
    """If applicable, coerce values of the parameterization from Numpy int/float to
    Python int/float.
    """
    return {
        name: numpy_type_to_python_type(value)
        for name, value in parameterization.items()
    }
