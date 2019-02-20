#!/usr/bin/env python3

from ae.lazarus.ae.utils.common.testutils import TestCase
from ae.lazarus.ae.utils.render.render import (
    h2_html,
    h3_html,
    list_item_html,
    p_html,
    render_report_elements,
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
        ]
        render_report_elements("test", elements)
