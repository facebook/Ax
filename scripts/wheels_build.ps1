# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
mypy -m pip install botorch jinja2 pandas scipy simplejson sklearn plotly numpy twine wheel
mypy -m pip install --upgrade botorch jinja2 pandas scipy simplejson sklearn plotly numpy twine wheel

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
