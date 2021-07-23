#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import tarfile
import time
from typing import Optional

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, ScriptExporter
from nbconvert.preprocessors import ExecutePreprocessor


TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${{CWD}}/core/Tutorial.js`);

class TutorialPage extends React.Component {{
  render() {{
    const {{config: siteConfig}} = this.props;
    const {{baseUrl}} = siteConfig;
    return (
      <Tutorial
        baseUrl={{baseUrl}}
        tutorialDir="{t_dir}"
        tutorialID="{tid}"
        totalExecTime={{{total_time}}}
      />
    );
  }}
}}

module.exports = TutorialPage;

"""

# we already load Plotly within html head on the site (just using <script>); this causes
# problems when trying to use requires within the notebook body, so we basically mock
# require to return already-loaded Plotly library
MOCK_JS_REQUIRES = """
<script>
const requirejs = Object();
requirejs.config = function() { };

const dependencyMap = {
    'plotly': Plotly,
}

function require(deps, fxn) {
    return fxn(...deps.map(dep => dependencyMap[dep]));
};
</script>
"""


def gen_tutorials(
    repo_dir: str, exec_tutorials: bool, kernel_name: Optional[str] = None
) -> None:
    """Generate HTML tutorials for Docusaurus Ax site from Jupyter notebooks.

    Also create ipynb and py versions of tutorial in Docusaurus site for
    download.
    """
    has_errors = False
    with open(os.path.join(repo_dir, "website", "tutorials.json"), "r") as infile:
        tutorial_config = json.loads(infile.read())

    # flatten config dict
    tutorial_configs = [
        config for category in tutorial_config.values() for config in category
    ]

    # prepare paths for converted tutorials & files
    os.makedirs(os.path.join(repo_dir, "website", "_tutorials"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "website", "static", "files"), exist_ok=True)

    for config in tutorial_configs:
        tid = config["id"]
        t_dir = config.get("dir")
        exec_on_build = config.get("exec_on_build", True)

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
            tutorial_dir = os.path.join(repo_dir, "tutorials")
            tutorial_path = os.path.join(repo_dir, "tutorials", "{}.ipynb".format(tid))
            html_path = os.path.join(
                repo_dir, "website", "_tutorials", "{}.html".format(tid)
            )
            js_path = os.path.join(
                repo_dir, "website", "pages", "tutorials", "{}.js".format(tid)
            )
            ipynb_path = os.path.join(
                repo_dir, "website", "static", "files", "{}.ipynb".format(tid)
            )
            py_path = os.path.join(
                repo_dir, "website", "static", "files", "{}.py".format(tid)
            )

        # load notebook
        with open(tutorial_path, "r") as infile:
            nb_str = infile.read()
            nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)

        # track total exec time (non-None if exec_on_build=True)
        total_time = None

        if exec_tutorials and exec_on_build:
            print("Executing tutorial {}".format(tid))
            kwargs = {"kernel_name": kernel_name} if kernel_name is not None else {}
            # 2.5 hours, in seconds
            timeout = int(60 * 60 * 2.5)
            ep = ExecutePreprocessor(timeout=timeout, **kwargs)
            start_time = time.time()

            # try / catch failures for now
            # will re-raise at the end
            try:
                # execute notebook, using `tutorial_dir` as working directory
                ep.preprocess(nb, {"metadata": {"path": tutorial_dir}})
                total_time = time.time() - start_time
                print(
                    "Done executing tutorial {}. Took {:.2f} seconds.".format(
                        tid, total_time
                    )
                )
            except Exception as exc:
                has_errors = True
                print("Couldn't execute tutorial {}!".format(tid))
                print(exc)
                total_time = None

        # convert notebook to HTML
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

        html_out = MOCK_JS_REQUIRES + str(nb_meat)

        # generate HTML file
        with open(html_path, "w") as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        t_dir_js = t_dir if t_dir else ""
        script = TEMPLATE.format(
            t_dir=t_dir_js,
            tid=tid,
            total_time=total_time if total_time is not None else "null",
        )
        with open(js_path, "w") as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        nbformat.write(nb, ipynb_path)
        exporter = ScriptExporter()
        script, meta = exporter.from_notebook_node(nb)
        with open(py_path, "w") as py_outfile:
            py_outfile.write(script)

        # create .tar archive (if necessary)
        if t_dir is not None:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(tutorial_dir, arcname=os.path.basename(tutorial_dir))

    if has_errors:
        raise Exception("There are errors in tutorials, will not continue to publish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, ipynb, and py files for tutorials."
    )
    parser.add_argument(
        "-w", "--repo_dir", metavar="path", required=True, help="Ax repo directory."
    )
    parser.add_argument(
        "-e",
        "--exec_tutorials",
        action="store_true",
        default=False,
        help="Execute tutorials (instead of just converting).",
    )
    parser.add_argument(
        "-k",
        "--kernel_name",
        required=False,
        default=None,
        type=str,
        help="Name of IPython / Jupyter kernel to use for executing notebooks.",
    )
    args = parser.parse_args()
    gen_tutorials(args.repo_dir, args.exec_tutorials, args.kernel_name)
