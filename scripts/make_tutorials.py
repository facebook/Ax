#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, Optional

import nbformat
import papermill
from bs4 import BeautifulSoup
from nbclient.exceptions import CellTimeoutError
from nbconvert import HTMLExporter, ScriptExporter

TUTORIALS_TO_SKIP = [
    "raytune_pytorch_cnn",  # TODO: Times out CI but passes locally. Investigate.
]


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


def _get_paths(repo_dir: str, t_dir: Optional[str], tid: str) -> Dict[str, str]:
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

    paths = {
        "tutorial_dir": tutorial_dir,
        "tutorial_path": tutorial_path,
        "html_path": html_path,
        "js_path": js_path,
        "ipynb_path": ipynb_path,
        "py_path": py_path,
    }
    if t_dir is not None:
        paths["tar_path"] = os.path.join(py_dir, "{}.tar.gz".format(tid))
    return paths


def run_script(
    tutorial: Path, timeout_minutes: int, env: Optional[Dict[str, str]] = None
) -> None:
    if env is not None:
        os.environ.update(env)
    papermill.execute_notebook(
        tutorial,
        tutorial,
        # This timeout is on cell-execution time, not on total runtime.
        execution_timeout=timeout_minutes * 60,
    )


def gen_tutorials(
    repo_dir: str,
    exec_tutorials: bool,
    name: Optional[str] = None,
    smoke_test: bool = False,
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
    # Running only the tutorial described by "name"
    if name is not None:
        tutorial_configs = [d for d in tutorial_configs if d["id"] == name]
        if len(tutorial_configs) == 0:
            raise RuntimeError(f"No tutorial found with name {name}.")
    # prepare paths for converted tutorials & files
    os.makedirs(os.path.join(repo_dir, "website", "_tutorials"), exist_ok=True)
    os.makedirs(os.path.join(repo_dir, "website", "static", "files"), exist_ok=True)
    env = {"SMOKE_TEST": "True"} if smoke_test else None

    for config in tutorial_configs:
        tid = config["id"]
        t_dir = config.get("dir")
        exec_on_build = config.get("exec_on_build", True)
        print("Generating {} tutorial".format(tid))
        paths = _get_paths(repo_dir=repo_dir, t_dir=t_dir, tid=tid)

        total_time = None

        if tid in TUTORIALS_TO_SKIP:
            print(f"Skipping execution of {tid}")
            continue
        elif exec_tutorials and exec_on_build:
            tutorial_path = Path(paths["tutorial_path"])
            print("Executing tutorial {}".format(tid))
            start_time = time.monotonic()

            # Try / catch failures for now. We will re-raise at the end.
            timeout_minutes = 15 if smoke_test else 150
            try:
                # Execute notebook.
                run_script(
                    tutorial=tutorial_path,
                    timeout_minutes=timeout_minutes,
                    env=env,
                )
                total_time = time.monotonic() - start_time
                print(
                    f"Finished executing tutorial {tid} in {total_time:.2f} seconds. "
                )
            except CellTimeoutError:
                has_errors = True
                print(
                    f"Tutorial {tid} exceeded the maximum runtime of "
                    f"{timeout_minutes} minutes."
                )
            except Exception as e:
                has_errors = True
                print(f"Encountered error running tutorial {tid}: \n {e}")

        # load notebook
        with open(paths["tutorial_path"], "r") as infile:
            nb_str = infile.read()
            nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)
        # convert notebook to HTML
        exporter = HTMLExporter(template_name="classic")
        html, _ = exporter.from_notebook_node(nb)

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
        with open(paths["html_path"], "w") as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        t_dir_js = t_dir if t_dir else ""
        script = TEMPLATE.format(
            t_dir=t_dir_js,
            tid=tid,
            total_time=total_time if total_time is not None else "null",
        )
        with open(paths["js_path"], "w") as js_outfile:
            js_outfile.write(script)

        # output tutorial in both ipynb & py form
        nbformat.write(nb, paths["ipynb_path"])
        exporter = ScriptExporter()
        script, _ = exporter.from_notebook_node(nb)
        with open(paths["py_path"], "w") as py_outfile:
            py_outfile.write(script)

        # create .tar archive (if necessary)
        if t_dir is not None:
            with tarfile.open(paths["tar_path"], "w:gz") as tar:
                tar.add(
                    paths["tutorial_dir"],
                    arcname=os.path.basename(paths["tutorial_dir"]),
                )

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
        "-s", "--smoke", action="store_true", help="Run in smoke test mode."
    )
    parser.add_argument(
        "-e",
        "--exec_tutorials",
        action="store_true",
        default=False,
        help="Execute tutorials (instead of just converting).",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Run a specific tutorial by name. The name should not include the "
        ".ipynb extension.",
    )
    args = parser.parse_args()
    gen_tutorials(
        args.repo_dir,
        args.exec_tutorials,
        smoke_test=args.smoke,
        name=args.name,
    )
