#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Exit if any error occurs
set -e

usage() {
  echo "Usage: $0 [-b] [-o] [-r] [-t]"
  echo ""
  echo "Build Ax documentation. Must be executed from root of Ax repository."
  echo ""
  echo "  -b   Build static version of documentation (otherwise start server)."
  echo "  -o   Only Docusaurus (skip tutorials). Useful when just make change to Docusaurus settings."
  echo "  -t   Execute tutorials (instead of just converting)."
  echo "  -r   Convert backtick-quoted class or function names in .md files into links to API documentation."
  echo ""
  exit 1
}

BUILD_STATIC=false
ONLY_DOCUSAURUS=false
BUILD_TUTORIALS=false
INSERT_API_REFS=false

while getopts 'hbotrk:' flag; do
  case "${flag}" in
    h)
      usage
      ;;
    b)
      BUILD_STATIC=true
      ;;
    o)
      ONLY_DOCUSAURUS=true
      ;;
    t)
      BUILD_TUTORIALS=true
      ;;
    r)
      INSERT_API_REFS=true
      ;;
    *)
      usage
      ;;
  esac
done

if [[ $ONLY_DOCUSAURUS == false ]]; then
  echo "-----------------------------------"
  echo "Generating tutorials"
  echo "-----------------------------------"
  python3 scripts/convert_ipynb_to_mdx.py --clean

fi  # end of not only Docusaurus block

# init Docusaurus deps
echo "-----------------------------------"
echo "Getting Docusaurus deps"
echo "-----------------------------------"
cd website || exit
yarn


if [[ $INSERT_API_REFS == true ]]; then
  echo "-----------------------------------"
  echo "Inserting API reference links in Markdown files"
  echo "-----------------------------------"
  cd ..
  cwd=$(pwd)
  python3 scripts/insert_api_refs.py --source_path "${cwd}/ax" --docs_path "${cwd}/docs"
  cd - || exit
fi

if [[ $BUILD_STATIC == true ]]; then
  echo "-----------------------------------"
  echo "Building static Docusaurus site"
  echo "-----------------------------------"
  yarn build
else
  echo "-----------------------------------"
  echo "Starting local server"
  echo "-----------------------------------"
  yarn start
fi
cd .. || exit
