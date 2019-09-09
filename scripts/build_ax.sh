#!/bin/bash
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# ManyLinux build
# Mount top-level Ax directory as Ax-master.
docker container run --mount type=bind,source="$(pwd)/../",target=/Ax-master -it quay.io/pypa/manylinux2010_x86_64

# MANUAL STEP FOR NOW
# Now, in Docker container, cd Ax-master and MANUALLY RUN ./docker_install.sh

# LOCAL BUILD
# Requires Python 3.6+3.7 installed locally, and on path
cd .. 
pip3.6 install numpy
python3.6 setup.py bdist_wheel

# Build Linux Python3.7
pip3.7 install numpy
python3.7 setup.py bdist_wheel

# Final PyPI Upload
twine upload dist/*

