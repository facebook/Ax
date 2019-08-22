# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# Powershell script, used for building the Python wheel distribution for Ax
#
# Notes:
# * Requires VS C++ Build Tools
# * This script should be run from the repo's parent dir
#
# Usage: ./scripts/wheel_build.ps1 -pypath [PATH_TO_PYTHON] [-upload]

# Python path to use for build, flag to upload file
param(
  [string]$pypath = $("python3"),
  [switch]$upload = $false
)

Set-Alias -Name mypy -Value $pypath
Write-Host "Py Version:"
mypy --version

# Jump into the Ax repo folder
pushd $PSScriptRoot\..

# Install or upgrade all the dependecies
mypy -m pip install botorch jinja2 pandas scipy simplejson sklearn plotly cython numpy searchtools twine wheel
mypy -m pip install --upgrade botorch jinja2 pandas scipy simplejson sklearn plotly cython numpy searchtools twine wheel

# Let's build
mypy ./setup.py bdist_wheel
# Validate the build
twine check dist/*

# Final PyPI Upload
If ($upload) {
  echo "Uploading"
  twine upload dist/*
}

# Done!
popd
