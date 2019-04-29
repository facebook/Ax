#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import json
import os
import tarfile

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, ScriptExporter


TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${{CWD}}/core/Tutorial.js`);

class TutorialPage extends React.Component {{
  render() {{
      const {{config: siteConfig}} = this.props;
      const {{baseUrl}} = siteConfig;
      return <Tutorial baseUrl={{baseUrl}} tutorialDir="{t_dir}" tutorialID="{tid}"/>;
  }}
}}

module.exports = TutorialPage;

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

    tutorial_ids = [x["id"] for v in tutorial_config.values() for x in v]
    tutorial_dirs = [x.get("dir") for v in tutorial_config.values() for x in v]

    for tid, t_dir in zip(tutorial_ids, tutorial_dirs):
        print("Generating {} tutorial".format(tid))

        if t_dir is not None:
            tutorial_dir = os.path.join(repo_dir, "tutorials", t_dir)
            html_dir = os.path.join(repo_dir, "website", "_tutorials", t_dir)
            js_dir = os.path.join(repo_dir, "website", "pages", "tutorials", t_dir)
            py_dir = os.path.join(repo_dir, "website", "static", "files", t_dir)

            for d in [tutorial_dir, html_dir, js_dir, py_dir]:
                os.makedirs(d, exist_ok=True)

            tutorial_path = os.path.join(tutorial_dir, "{}.ipynb".format(tid))
            html_path = os.path.join(html_dir, "{}.html".format(tid))
            js_path = os.path.join(js_dir, "{}.js".format(tid))
            ipynb_path = os.path.join(py_dir, "{}.ipynb".format(tid))
            py_path = os.path.join(py_dir, "{}.py".format(tid))
            tar_path = os.path.join(py_dir, "{}.tar.gz".format(tid))
        else:
            tutorial_path = os.path.join(repo_dir, "tutorials", "{}.ipynb".format(tid))
            html_path = os.path.join(repo_dir, "website", "_tutorials", "{}.html".format(tid))
            js_path = os.path.join(repo_dir, "website", "pages", "tutorials", "{}.js".format(tid))
            ipynb_path = os.path.join(repo_dir, "website", "static", "files", "{}.ipynb".format(tid))
            py_path = os.path.join(repo_dir, "website", "static", "files", "{}.py".format(tid))

        # convert notebook to HTML
        with open(tutorial_path, "r") as infile:
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

        # generate HTML file
        with open(html_path, 'w') as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        t_dir_js = t_dir if t_dir else ''
        script = TEMPLATE.format(t_dir=t_dir_js, tid=tid)
        with open(js_path, "w") as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        with open(ipynb_path, "w") as ipynb_outfile:
            ipynb_outfile.write(nb_str)
        exporter = ScriptExporter()
        script, meta = exporter.from_notebook_node(nb)
        with open(py_path, "w") as py_outfile:
            py_outfile.write(script)

        # create .tar archive (if necessary)
        if t_dir is not None:
	        with tarfile.open(tar_path, "w:gz") as tar:
	            tar.add(tutorial_dir, arcname=os.path.basename(tutorial_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, ipynb, and py files for tutorials."
    )
    parser.add_argument(
        "-w", "--repo_dir", metavar="path", required=True, help="Ax repo directory."
    )
    args = parser.parse_args()
    gen_tutorials(args.repo_dir)
