#!/usr/bin/env python3

import argparse
import json
import os

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, ScriptExporter


TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${{CWD}}/core/Tutorial.js`);

function renderTutorial(props) {{
  return <Tutorial tutorialID="{}"/>;
}}

module.exports = renderTutorial;

"""

SRCS = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js",
]

JS_SCRIPT_TAGS = """<script src="{}"></script>"""


def gen_tutorials(repo_dir: str) -> None:
    """Generate HTML tutorials for Docusaurus Ax site from Jupyter notebooks.

    Also create ipynb and py versions of tutorial in Docusaurus site for
    download.
    """
    with open(os.path.join(repo_dir, "website", "tutorials.json"), "r") as infile:
        tutorial_config = json.loads(infile.read())

    tutorial_ids = {x["id"] for v in tutorial_config.values() for x in v}

    for tid in tutorial_ids:
        print("Generating {} tutorial".format(tid))

        # convert notebook to HTML
        with open(
            os.path.join(repo_dir, "tutorials", "{}.ipynb".format(tid)), "r"
        ) as infile:
            nb_str = infile.read()
            nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

        exporter = HTMLExporter()
        html, meta = exporter.from_notebook_node(nb)

        # pull out html div for notebook
        soup = BeautifulSoup(html, "html.parser")
        nb_meat = soup.find("div", {"id": "notebook-container"})
        del nb_meat.attrs["id"]
        nb_meat.attrs["class"] = ["notebook"]

        # when output html, iframe it (useful for Ax reports)
        for html_div in nb_meat.findAll("div", {"class": "output_html"}):
            if html_div.html is not None:
                iframe = soup.new_tag("iframe")
                iframe.attrs["src"] = "data:text/html;charset=utf-8," + str(
                    html_div.html
                )
                # replace `#` in CSS
                iframe.attrs["src"] = iframe.attrs["src"].replace("#", "%23")
                html_div.contents = [iframe]

        html_out = "".join([JS_SCRIPT_TAGS.format(src) for src in SRCS]) + str(nb_meat)

        with open(
            os.path.join(repo_dir, "website", "_tutorials", "{}.html".format(tid)), "w"
        ) as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        script = TEMPLATE.format(tid)
        with open(
            os.path.join(
                repo_dir, "website", "pages", "tutorials", "{}.js".format(tid)
            ),
            "w",
        ) as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        with open(
            os.path.join(
                repo_dir, "website", "static", "files", "{}.ipynb".format(tid)
            ),
            "w",
        ) as ipynb_outfile:
            ipynb_outfile.write(nb_str)
        exporter = ScriptExporter()
        script, meta = exporter.from_notebook_node(nb)
        with open(
            os.path.join(repo_dir, "website", "static", "files", "{}.py".format(tid)),
            "w",
        ) as py_outfile:
            py_outfile.write(script)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, ipynb, and py files for tutorials."
    )
    parser.add_argument(
        "-w", "--repo_dir", metavar="path", required=True, help="Ax repo directory."
    )
    args = parser.parse_args()
    gen_tutorials(args.repo_dir)
