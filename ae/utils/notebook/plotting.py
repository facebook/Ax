#!/usr/bin/env python3

from ae.lazarus.ae.plot.base import AEPlotConfig
from ae.lazarus.ae.plot.render import _js_requires, _wrap_js, plot_config_to_html
from ae.lazarus.ae.utils.common.logger import get_logger
from IPython.display import display


logger = get_logger("ipy_plotting")


def init_notebook_plotting(offline=False):
    """Initialize plotting in notebooks, either in online or offline mode."""
    display_bundle = {"text/html": _wrap_js(_js_requires(offline=True))}
    display(display_bundle, raw=True)
    logger.info(
        "Injecting Plotly library into cell. " "Do not overwrite or delete cell."
    )


def render(plot_config: AEPlotConfig, inject_helpers=False) -> None:
    """Render plot config."""
    display_bundle = {"text/html": plot_config_to_html(plot_config, inject_helpers)}
    display(display_bundle, raw=True)
