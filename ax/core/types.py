#!/usr/bin/env python3

import enum
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from botorch.acquisition import AcquisitionFunction


TParamCounter = DefaultDict[int, int]
TParamValue = Optional[Union[str, bool, float, int]]
TParameterization = Dict[str, TParamValue]
TParamValueList = List[TParamValue]  # a parameterization without the keys
TContextStratum = Optional[Dict[str, Union[str, float, int]]]

TBounds = Optional[Tuple[np.ndarray, np.ndarray]]
TModelMean = Dict[str, List[float]]
TModelCov = Dict[str, Dict[str, List[float]]]
TModelPredict = Tuple[TModelMean, TModelCov]
# Model predictions for a single arm:
# ( { metric -> mean }, { metric -> { other_metric -> covariance } } ).
TModelPredictArm = Tuple[Dict[str, float], Dict[str, Dict[str, float]]]

# Format for trasmitting externally evaluated data to Ax
# {metric_name -> (mean, standard error)}
TEvaluationOutcome = Dict[str, Tuple[float, float]]

TConfig = Dict[str, Union[int, float, str, AcquisitionFunction]]
TBucket = List[Dict[str, List[str]]]


class ComparisonOp(enum.Enum):
    """Class for enumerating comparison operations."""

    GEQ: int = 0
    LEQ: int = 1
