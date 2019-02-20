#!/usr/bin/env python3

import enum
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np


TParamCounter = DefaultDict[int, int]
TParamValue = Optional[Union[str, bool, float]]
TParameterization = Dict[str, TParamValue]
TParamValueList = List[TParamValue]  # a parameterization without the keys
TContextStratum = Optional[Dict[str, Union[str, float, int]]]

TBounds = Optional[Tuple[np.ndarray, np.ndarray]]
TModelMean = Dict[str, List[float]]
TModelCov = Dict[str, Dict[str, List[float]]]
TModelPredict = Tuple[TModelMean, TModelCov]
# Model predictions for a single condition:
# ( { metric -> mean }, { metric -> { other_metric -> covariance } } ).
TModelPredictCondition = Tuple[Dict[str, float], Dict[str, Dict[str, float]]]

TConfig = Dict[str, Union[int, float, str]]
TBucket = List[Dict[str, List[str]]]


class ComparisonOp(enum.Enum):
    """Class for enumerating comparison operations."""

    GEQ: int = 0
    LEQ: int = 1
