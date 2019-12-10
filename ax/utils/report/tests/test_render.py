#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.utils.common.testutils import TestCase
from ax.utils.report.render import (
    h2_html,
    h3_html,
    link_html,
    list_item_html,
    p_html,
    render_report_elements,
    table_cell_html,
    table_heading_cell_html,
    table_html,
    table_row_html,
    unordered_list_html,
)


class RenderTest(TestCase):
    def testRenderReportElements(self):
        elements = [
            p_html("foobar"),
            h2_html("foobar"),
            h3_html("foobar"),
            list_item_html("foobar"),
            unordered_list_html(["foo", "bar"]),
            link_html("foo", "bar"),
            table_cell_html("foobar"),
            table_cell_html("foobar", width="100px"),
            table_heading_cell_html("foobar"),
            table_row_html(["foo", "bar"]),
            table_html(["foo", "bar"]),
        ]
        render_report_elements("test", elements)
