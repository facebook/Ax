#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
import json
import os
import pkgutil
import uuid
from typing import Dict

import plotly.offline as plotly_offline
from ax import plot as PLOT_MODULE
from ax.plot.base import AxPlotConfig, AxPlotTypes
from jinja2 import Template


# Rendering constants
DEFAULT_WIDTH = "100%"
DEFAULT_HEIGHT = 550
CSS_FILE = "ax/ax/plot/css/base.css"


# Common resources used in plotting (load with _load_js_resource)
class _AxPlotJSResources(enum.Enum):
    """Enum of common JS resources for plotting."""

    CSS_INJECTION = "css"
    HELPER_FXNS = "helpers"
    PLOTLY_OFFLINE = "plotly_offline"
    PLOTLY_ONLINE = "plotly_online"
    PLOTLY_REQUIRES = "plotly_requires"


# JS-based plots that are supported in Ax should be registered here
Ax_PLOT_REGISTRY: Dict[enum.Enum, str] = {AxPlotTypes.GENERIC: "generic_plotly.js"}


def _load_js_resource(resource_type: _AxPlotJSResources) -> str:
    """Convert plot config to corresponding JS code."""
    resource = pkgutil.get_data(
        PLOT_MODULE.__name__, os.path.join("js", "common", resource_type.value + ".js")
    )
    if resource is None:
        raise ValueError(f"Cannot find JS resource {resource_type.value}.")
    return resource.decode("utf8")


def _load_css_resource() -> str:
    resource = pkgutil.get_data(PLOT_MODULE.__name__, os.path.join("css", "base.css"))
    assert resource is not None
    return resource.decode("utf8")


def _js_requires(offline: bool = False) -> str:
    """Format JS requires for Plotly dependency.

    Args:
        offline: if True, inject entire Plotly library for offline use.

    Returns:
        str: <script> block with Plotly dependency.

    """
    helper_fxns = _load_js_resource(_AxPlotJSResources.HELPER_FXNS)
    if offline:
        script = Template(_load_js_resource(_AxPlotJSResources.PLOTLY_OFFLINE)).render(
            library=plotly_offline.offline.get_plotlyjs()
        )
    else:
        script = _load_js_resource(_AxPlotJSResources.PLOTLY_ONLINE)
    return script + helper_fxns


def _get_plot_js(
    config: AxPlotConfig,
    plot_module_name: str,
    plot_resources: Dict[enum.Enum, str],
    plotdivid: str,
) -> str:
    """Convert plot config to corresponding JS code."""
    if not isinstance(config, AxPlotConfig):
        raise ValueError("Config must be instance of AxPlotConfig.")
    js_template = pkgutil.get_data(
        plot_module_name, os.path.join("js", plot_resources[config.plot_type])
    )
    if js_template is None:
        raise ValueError(f"Cannot find JS template {plot_resources[config.plot_type]}.")
    return Template(js_template.decode("utf8")).render(
        id=json.dumps(plotdivid), **{k: json.dumps(v) for k, v in config.data.items()}
    )


def _wrap_js(script: str) -> str:
    """Wrap JS in <script></script> tag for injection into HTML."""
    return "<script type='text/javascript'>{script}</script>".format(script=script)


def _plot_js_to_html(js_script: str, plotdivid: str) -> str:
    """Embed JS script for Plotly plot in HTML.

    Result is a <div> block with a unique id set in `plotdivid` and a
    <script> block that contains the JS needed to render the plot inside the
    div.

    Args:
        js_script: JS for rendering plot.
        plotdivid: unique string ID for div.

    """
    plot_div = (
        '<div id="{id}" style="width: {width};" class="plotly-graph-div">' "</div>"
    ).format(id=plotdivid, width=DEFAULT_WIDTH)
    plot_js = Template(_load_js_resource(_AxPlotJSResources.PLOTLY_REQUIRES)).render(
        script=js_script
    )
    return plot_div + _wrap_js(plot_js)


def plot_config_to_html(
    plot_config: AxPlotConfig,
    plot_module_name: str = PLOT_MODULE.__name__,
    plot_resources: Dict[enum.Enum, str] = Ax_PLOT_REGISTRY,
    inject_helpers: bool = False,
) -> str:
    """Generate HTML + JS corresponding from a plot config."""
    plotdivid = uuid.uuid4().hex
    plot_js = _get_plot_js(plot_config, plot_module_name, plot_resources, plotdivid)
    if inject_helpers:
        helper_fxns = _load_js_resource(_AxPlotJSResources.HELPER_FXNS)
        plot_js = helper_fxns + plot_js
    return _plot_js_to_html(plot_js, plotdivid)
