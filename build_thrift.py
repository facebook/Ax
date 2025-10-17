# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import subprocess

from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        subprocess.check_call(["thrift", "-r", "--gen", "py", "ax/ax_types.thrift"])

        for root, _dirs, files in os.walk("gen-py"):
            for filename in files:
                if filename == "ttypes.py":
                    old_path = os.path.join(root, filename)
                    new_path = os.path.join(root, "thrift_types.py")
                    os.rename(old_path, new_path)
                    print(f"Renamed {old_path} to {new_path}")

        if os.path.exists("gen-py"):
            for item in os.listdir("gen-py"):
                src = os.path.join("gen-py", item)
                dst = os.path.join("ax", item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    print(f"Copied {src} to {dst}")
                elif os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"Copied {src} to {dst}")

        super().run()
