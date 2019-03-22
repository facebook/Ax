#!/usr/bin/env python3

import enum
from typing import Any, Dict, List, NamedTuple, Optional, Union

import simplejson
from ax.core.types import TParameterization
from plotly import utils


# Constants used for numerous plots
CI_OPACITY = 0.4
DECIMALS = 3
Z = 1.96


class AEPlotTypes(enum.Enum):
    """Enum of AE plot types."""

    DATA_AVAILABILITY = 0
    CONTOUR = 1
    EXPOSURE = 2
    GENERIC = 3
    SLICE = 4
    INTERACT_CONTOUR = 5
    BANDIT_ROLLOUT = 6


# Configuration for all plots
class _AEPlotConfigBase(NamedTuple):
    data: Dict[str, Any]
    plot_type: AEPlotTypes


class AEPlotConfig(_AEPlotConfigBase):
    """Config for plots"""

    def __new__(cls, data: Dict[str, Any], plot_type: AEPlotTypes) -> "AEPlotConfig":
        # Convert data to json-encodable form (strips out NamedTuple and numpy
        # array). This is a lossy conversion.
        dict_data = simplejson.loads(
            simplejson.dumps(
                data,
                cls=utils.PlotlyJSONEncoder,
                namedtuple_as_object=True,  # uses NamesTuple's `_asdict()`
            )
        )
        # pyre-fixme[7]: Expected `AEPlotConfig` but got `_AEPlotConfigBase`.
        return super(AEPlotConfig, cls).__new__(cls, dict_data, plot_type)


# Structs for plot data
class PlotInSampleArm(NamedTuple):
    """Struct for in-sample arms (both observed and predicted data)"""

    name: str
    params: TParameterization
    y: Dict[str, float]
    y_hat: Dict[str, float]
    se: Dict[str, float]
    se_hat: Dict[str, float]
    context_stratum: Optional[Dict[str, Union[str, float]]]


class PlotOutOfSampleArm(NamedTuple):
    """Struct for out-of-sample arms (only predicted data)"""

    name: str
    params: TParameterization
    y_hat: Dict[str, float]
    se_hat: Dict[str, float]
    context_stratum: Optional[Dict[str, Union[str, float]]]


class PlotData(NamedTuple):
    """Struct for plot data, including both in-sample and out-of-sample arms"""

    metrics: List[str]
    in_sample: Dict[str, PlotInSampleArm]
    out_of_sample: Optional[Dict[str, Dict[str, PlotOutOfSampleArm]]]
    status_quo_name: Optional[str]


class PlotMetric(NamedTuple):
    """Struct for metric"""

    # @TODO T40555279: metric --> metric_name everywhere in plotting
    metric: str
    pred: bool
