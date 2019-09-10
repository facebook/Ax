#!/bin/bash
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

VERSION=$1

# Build Linux Python3.6
# Execute from Ax root directory.
cd ../ || exit
/opt/python/cp36-cp36m/bin/pip3.6 install numpy
/opt/python/cp36-cp36m/bin/python3.6 setup.py bdist_wheel

# Build Linux Python3.7
/opt/python/cp37-cp37m/bin/pip3.7 install numpy
/opt/python/cp37-cp37m/bin/python3.7 setup.py bdist_wheel

# Convert to manylinux
cd dist || exit
auditwheel repair ax_platform-"$VERSION"-cp36-cp36m-linux_x86_64.whl
auditwheel repair ax_platform-"$VERSION"-cp37-cp37m-linux_x86_64.whl
rm ./*
mv wheelhouse/* .
rm -rf wheelhouse
