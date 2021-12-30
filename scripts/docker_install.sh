#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

VERSION=$1

# Build Linux Python3.7
# Execute from Ax root directory.
cd ../ || exit
/opt/python/cp37-cp37m/bin/pip3.7 install numpy
/opt/python/cp37-cp37m/bin/python3.7 setup.py bdist_wheel

# Convert to manylinux
cd dist || exit
auditwheel repair ax_platform-"$VERSION"-cp37-cp37m-linux_x86_64.whl
rm ./*
mv wheelhouse/* .
rm -rf wheelhouse
