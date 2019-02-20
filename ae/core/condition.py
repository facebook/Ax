#!/usr/bin/env python3
# pyre-strict

import hashlib
import json
from typing import Optional

import numpy as np
from ae.lazarus.ae.core.base import Base
from ae.lazarus.ae.core.types.types import TParameterization
from ae.lazarus.ae.utils.common.equality import equality_typechecker


class Condition(Base):
    """Base class for defining conditions.

    Randomization in experiments assigns units to a given condition. Thus, the condition
    encapsulates the parametrization needed by the unit.

    """

    def __init__(self, params: TParameterization, name: Optional[str] = None) -> None:
        """Inits Condition.

        Args:
            params: Mapping from parameter names to values.
            name: Defaults to None; will be set when condition is attached to a trial
        """
        self._params = params
        self._name = name

    @property
    def params(self) -> TParameterization:
        """Get mapping from parameter names to values."""
        return self._params

    @property
    def has_name(self) -> bool:
        """Return true if condition's name is not None."""
        return self._name is not None

    @property
    def name(self) -> str:
        """Get condition name. Throws if name is None."""
        if self._name is None:
            raise ValueError("Condition's name is None.")
        return self._name

    @property
    def name_or_short_signature(self) -> str:
        """Returns condition name if exists; else last 4 characters of the hash.

        Used for presentation of candidates (e.g. plotting and tables),
        where the candidates do not yet have names (since names are
        automatically set upon addition to a trial).

        """
        return self._name or self.signature[-4:]

    @name.setter
    def name(self, name: str) -> None:
        if self._name is not None:
            raise ValueError("Condition name is not mutable once set.")
        self._name = name

    @property
    def signature(self) -> str:
        """Get unique representation of a condition."""
        return self.md5hash(self.params)

    @staticmethod
    def md5hash(params: TParameterization) -> str:
        """Return unique identifier for condition's parameters.

        Args:
            params: Parameterization; mapping of param name
                to value.

        Returns:
            Hash of condition's parameters.

        """
        # TODO [drfreund]: remove!
        for k, v in params.items():
            if type(v) is np.int64:
                params[k] = int(v)  # pragma: no cover
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()

    def clone(self, clear_name: bool = False) -> "Condition":
        """Create a copy of this condition.

        Args:
            clear_name (bool, optional): whether this cloned copy should set its
                name to None instead of the name of the condition being cloned.
                Defaults to False.
        """
        clear_name = clear_name or not self.has_name
        return Condition(
            params=self.params.copy(), name=None if clear_name else self.name
        )

    def __repr__(self) -> str:
        params_str = f"params={self._params}"
        if self.has_name:
            name_str = f"name={self.name}"
            return f"Condition({name_str}, {params_str})"
        return f"Condition({params_str})"

    @equality_typechecker
    def __eq__(self, other: "Condition") -> bool:
        """Need to overwrite the default __eq__ method of Base,
        because accessing the "name" attribute of Condition
        can result in an error.
        """
        params_equal = self.params == other.params
        names_equal = self.has_name == other.has_name
        if names_equal and self.has_name:
            names_equal = self.name == other.name
        return params_equal and names_equal

    def __hash__(self) -> int:
        return int(self.signature, 16)
