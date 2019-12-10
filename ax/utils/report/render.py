#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pkgutil
from typing import List, Optional

import ax.utils.report as report_module
from ax.plot.render import _js_requires, _load_css_resource as _load_plot_css_resource
from jinja2 import Environment, FunctionLoader


def _load_css_resource() -> str:
    resource = pkgutil.get_data(
        report_module.__name__, os.path.join("resources", "report.css")
    )
    assert resource is not None
    return resource.decode("utf8")


REPORT_ELEMENT_TEMPLATE = "simple_template.html"


def p_html(text: str) -> str:
    """Embed text in paragraph tag."""
    return "<p>{}</p>".format(text)


def h2_html(text: str) -> str:
    """Embed text in subheading tag."""
    return "<h2>{}</h2>".format(text)


def h3_html(text: str) -> str:
    """Embed text in subsubheading tag."""
    return "<h3>{}</h3>".format(text)


def list_item_html(text: str) -> str:
    """Embed text in list element tag."""
    return "<li>{}</li>".format(text)


def unordered_list_html(list_items: List[str]) -> str:
    """Embed list of html elements into an unordered list tag."""
    return "<ul>{}</ul>".format("".join(list_items))


def link_html(text: str, href: str) -> str:
    """Embed text and reference address into link tag."""
    return '<a href="{}">{}</a>'.format(href, text)


def table_cell_html(text: str, width: Optional[str] = None) -> str:
    """Embed text or an HTML element into table cell tag."""
    if width:
        return f"<td width={width}>{text}</td>"
    else:
        return f"<td>{text}</td>"


def table_heading_cell_html(text: str) -> str:
    """Embed text or an HTML element into table heading cell tag."""
    return "<th>{}</th>".format(text)


def table_row_html(table_cells: List[str]) -> str:
    """Embed list of HTML elements into table row tag."""
    return "<tr>{}</tr>".format("".join(table_cells))


def table_html(table_rows: List[str]) -> str:
    """Embed list of HTML elements into table tag."""
    return "<table>{}</table>".format("".join(table_rows))


def render_report_elements(
    experiment_name: str,
    html_elements: List[str],
    header: bool = True,
    offline: bool = False,
    notebook_env: bool = False,
) -> str:
    """Generate Ax HTML report for a given experiment from HTML elements.

    Uses Jinja2 for template. Injects Plotly JS for graph rendering.

    Example:

    ::

        html_elements = [
            h2_html("Subsection with plot"),
            p_html("This is an example paragraph."),
            plot_html(plot_fitted(gp_model, 'perf_metric')),
            h2_html("Subsection with table"),
            pandas_html(data.df),
        ]
        html = render_report_elements('My experiment', html_elements)

    Args:
        experiment_name: the name of the experiment to use for title.
        html_elements: list of HTML strings to render in report
            body.
        header: if True, render experiment title as a header.
            Meant to be used for standalone reports (e.g. via email), as opposed
            to served on the front-end.
        offline: if True, entire Plotly library is bundled
            with report.
        notebook_env: if True, caps the report width to 700px for viewing in a
            notebook environment.

    Returns:
        str: HTML string.

    """
    # combine CSS for report and plots
    css = _load_css_resource() + _load_plot_css_resource()
    return (
        _get_jinja_environment()
        .get_template(REPORT_ELEMENT_TEMPLATE)
        .render(
            experiment_name=experiment_name,
            css=css,
            js_requires=_js_requires(),
            html_elements=html_elements,
            headfoot=header,
            notebook_env=notebook_env,
        )
    )


def _load_html_template(name: str) -> str:
    resource = pkgutil.get_data(report_module.__name__, os.path.join("resources", name))
    assert resource is not None
    return resource.decode("utf8")


def _get_jinja_environment() -> Environment:
    return Environment(loader=FunctionLoader(_load_html_template))
