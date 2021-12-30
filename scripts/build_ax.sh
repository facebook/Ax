#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# ManyLinux build
# Mount top-level Ax directory as Ax-main.
docker container run --mount type=bind,source="$(pwd)/../",target=/Ax-main -it quay.io/pypa/manylinux2010_x86_64

# MANUAL STEP FOR NOW
# Now, in Docker container, cd Ax-main and MANUALLY RUN ./docker_install.sh

# LOCAL BUILD
# Requires Python 3.7 installed locally, and on path
cd ..
pip3.7 install numpy
python3.7 setup.py bdist_wheel

# Final PyPI Upload
twine upload dist/*
