# Used for building the Python 3.6.3 wheel dist for Ax
# Note: Requires VS C++ Build Tools

# Usage: ./wheel_build.ps1 -pypath [PATH_TO_PYTHON] [-upload]
# Note: This script should be run from the repo's parent dir

# Python path to use for build, flag to upload file
param(
  [string]$pypath = $("C:\Users\scubasteve\AppData\Local\Programs\Python\Python36\python.exe"),
  [switch]$upload = $false
)

# Jump into the Ax repo folder
pushd Ax
Set-Alias -Name mypy -Value $pypath
Write-Host "Py Version:"
mypy --version

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

