#!/usr/bin/env python3

from typing import List

from ae.lazarus.ae.generator.array import ArrayGenerator
from ae.lazarus.ae.models.numpy_base import NumpyModel


class NumpyGenerator(ArrayGenerator):
    """A model generator for using numpy array-based models.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.
    """

    model: NumpyModel
    outcomes: List[str]
    params: List[str]
