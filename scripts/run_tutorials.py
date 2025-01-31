#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import time
from pathlib import Path

import papermill
from nbclient.exceptions import CellTimeoutError

TUTORIALS_TO_SKIP = [
    "raytune_pytorch_cnn",  # TODO: Times out CI but passes locally. Investigate.
]


def run_script(
    tutorial: Path, timeout_minutes: int, env: dict[str, str] | None = None
) -> None:
    if env is not None:
        os.environ.update(env)
    papermill.execute_notebook(
        tutorial,
        tutorial,
        # This timeout is on cell-execution time, not on total runtime.
        execution_timeout=timeout_minutes * 60,
    )


def run_tutorials(
    repo_dir: str,
    name: str | None = None,
    smoke_test: bool = False,
) -> None:
    """Run Jupyter notebooks.

    We check in the tutorial notebook un-run, and run them in CI as integration tests.
    """
    has_errors = False

    with open(os.path.join(repo_dir, "website", "tutorials.json")) as infile:
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
    env = {"SMOKE_TEST": "True"} if smoke_test else None

    for config in tutorial_configs:
        tid = config["id"]
        tutorial_path = os.path.join(repo_dir, "tutorials", tid, f"{tid}.ipynb")

        total_time = None

        if tid in TUTORIALS_TO_SKIP:
            print(f"Skipping execution of {tid}")
            continue
        else:
            print(f"Executing tutorial {tid}")
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
        "-n",
        "--name",
        help="Run a specific tutorial by name. The name should not include the "
        ".ipynb extension.",
    )
    args = parser.parse_args()
    run_tutorials(
        args.repo_dir,
        smoke_test=args.smoke,
        name=args.name,
    )
