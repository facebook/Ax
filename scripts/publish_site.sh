#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Exit if any error occurs
set -e
set -x

usage() {
  echo "Usage: $0 [-d] [-k KERNEL_NAME] [-v VERSION]"
  echo ""
  echo "Build and push updated Ax site. Will either update latest or bump stable version."
  echo ""
  echo "  -d                Use Docusaurus bot GitHub credentials. If not specified, will use default GitHub credentials."
  echo "  -k=KERNEL_NAME    Kernel name to use for executing tutorials. Use Jupyter default if not set."
  echo "  -v=VERSION        Build site for new library version. If not specified, will update latest."
  echo ""
  exit 1
}

VERSION=false
DOCUSAURUS_BOT=false
KERNEL_NAME=false

while getopts 'dhk:v:' option; do
  case "${option}" in
    d)
      DOCUSAURUS_BOT=true
      ;;
    h)
      usage
      ;;
    k)
      KERNEL_NAME=${OPTARG}
      ;;
    v)
      VERSION=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done

# Function to get absolute filename
fullpath() {
  echo "$(cd "$(dirname "$1")" || exit; pwd -P)/$(basename "$1")"
}

# Current directory (needed for cleanup later)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit

if [[ $DOCUSAURUS_BOT == true ]]; then
  # Setup git credentials
  git config --global user.name "Ax Website Deployment Script"
  git config --global user.email "docusaurus-bot@users.noreply.github.com"
  echo "machine github.com login docusaurus-bot password ${DOCUSAURUS_PUBLISH_TOKEN}" > ~/.netrc

  # Clone both main & gh-pages branches
  git clone https://docusaurus-bot@github.com/facebook/Ax.git Ax-main
  git clone --branch gh-pages https://docusaurus-bot@github.com/facebook/Ax.git Ax-gh-pages
else
  git clone git@github.com:facebook/Ax.git Ax-main
  git clone --branch gh-pages git@github.com:facebook/Ax.git Ax-gh-pages
fi

# A few notes about the script below:
# * Docusaurus versioning was designed to *only* version the markdown
#   files in the docs/ subdirectory. We are repurposing parts of Docusaurus
#   versioning, but snapshotting the entire site. Versions of the site are
#   stored in the versions/ subdirectory on gh-pages:
#
#   --gh-pages/
#     |-- api/
#     |-- css/
#     |-- docs/
#     |   ...
#     |-- versions/
#     |   |-- 1.0.1/
#     |   |-- 1.0.2/
#     |   |   ...
#     |   |-- latest/
#     |   ..
#     |-- versions.html
#
# * The stable version is in the top-level directory. It is also
#   placed into the versions/ subdirectory so that it does not need to
#   be built again when the version is augmented.
# * We want to support serving / building the Docusaurus site locally
#   without any versions. This means that we have to keep versions.js
#   outside of the website/ subdirectory.
# * We do not want to have a tracked file that contains all versions of
#   the site or the latest version. Instead, we determine this at runtime.
#   We use what's on gh-pages in the versions subdirectory as the
#   source of truth for available versions and use the latest tag on
#   the main branch as the source of truth for the latest version.

if [[ $VERSION == false ]]; then
  echo "-----------------------------------------"
  echo "Updating latest (main) version of site "
  echo "-----------------------------------------"

  # Populate _versions.json from existing versions; this is used
  # by versions.js & needed to build the site (note that we don't actually
  # use versions.js for latest build, but we do need versions.js
  # in website/pages in order to use docusaurus-versions)
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('Ax-gh-pages/versions') if v != 'latest' and not v.startswith('.')]; "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > Ax-main/website/_versions.json

  # Move versions.js to website subdirectory.
  # This is the page you see when click on version in navbar.
  cp Ax-main/scripts/versions.js Ax-main/website/pages/en/versions.js
  cd Ax-main/website || exit

  # Replace baseUrl (set to /versions/latest/) & disable Algolia
  CONFIG_FILE=$(fullpath "siteConfig.js")
  python3 ../scripts/patch_site_config.py -f "${CONFIG_FILE}" -b "/versions/latest/" --disable_algolia

  # Tag site with "latest" version
  yarn
  yarn run version latest

  # Build site
  cd .. || exit
  ./scripts/make_docs.sh -b -t -k "${KERNEL_NAME}"
  rm -rf ../website/build/Ax/docs/next  # don't need this

  # Move built site to gh-pages (but keep old versions.js)
  cd "${WORK_DIR}" || exit
  cp Ax-gh-pages/versions/latest/versions.html versions.html
  rm -rf Ax-gh-pages/versions/latest
  mv Ax-main/website/build/Ax Ax-gh-pages/versions/latest
  # versions.html goes both in top-level and under en/ (default language)
  cp versions.html Ax-gh-pages/versions/latest/versions.html
  cp versions.html Ax-gh-pages/versions/latest/en/versions.html

  # erase git history then force push to overwrite
  cd Ax-gh-pages || exit
  rm -rf .git
  git init -b main
  git add --all
  git commit -m 'Update latest version of site'
  git push --force "https://github.com/facebook/Ax" main:gh-pages

else
  echo "-----------------------------------------"
  echo "Building new version ($VERSION) of site "
  echo "-----------------------------------------"

  # Checkout main branch with specified tag
  cd Ax-main || exit
  git fetch --tags
  git checkout "${VERSION}"

  # Populate _versions.json from existing versions; this contains a list
  # of versions present in gh-pages (excluding latest). This is then used
  # to populate versions.js (which forms the page that people see when they
  # click on version number in navbar).
  # Note that this script doesn't allow building a version of the site that
  # is already on gh-pages.
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('../Ax-gh-pages/versions') if v != 'latest' and not v.startswith('.')]; "
  CMD+="assert '${VERSION}' not in vs, '${VERSION} is already on gh-pages.'; "
  CMD+="vs.append('${VERSION}'); "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > website/_versions.json

  cp scripts/versions.js website/pages/en/versions.js

  # Set Docusaurus version as 'stable'
  cd website || exit
  yarn
  yarn run version stable

  # Build new version of site (this will be stable, default version)
  # Execute tutorials
  cd .. || exit
  ./scripts/make_docs.sh -b -t -k "${KERNEL_NAME}"

  # Move built site to new folder (new-site) & carry over old versions
  # from existing gh-pages
  cd "${WORK_DIR}" || exit
  rm -rf Ax-main/website/build/Ax/docs/next  # don't need this
  mv Ax-main/website/build/Ax new-site
  mv Ax-gh-pages/versions new-site/versions

  # Build new version of site (to be placed in versions/$VERSION/)
  # the only thing that changes here is the baseUrl (for nav purposes)
  # we build this now so that in the future, we can just bump version and not move
  # previous stable to versions
  cd Ax-main/website || exit

  # Replace baseUrl & disable Algolia
  CONFIG_FILE=$(fullpath "siteConfig.js")
  python3 ../scripts/patch_site_config.py -f "${CONFIG_FILE}" -b "/versions/${VERSION}/" --disable_algolia

  # Set Docusaurus version with exact version & build
  yarn run version "${VERSION}"
  cd .. || exit
  # Only run Docusaurus (skip tutorial build & Sphinx)
  ./scripts/make_docs.sh -b -o
  rm -rf website/build/Ax/docs/next  # don't need this
  rm -rf website/build/Ax/docs/stable  # or this
  mv website/build/Ax "../new-site/versions/${VERSION}"

  # Need to run script to update versions.js for previous versions in
  # new-site/versions with the newly built versions.js. Otherwise,
  # the versions.js for older versions in versions subdirectory
  # won't be up-to-date and will not have a way to navigate back to
  # newer versions. This is the only part of the old versions that
  # needs to be updated when a new version is built.
  cd "${WORK_DIR}" || exit
  python3 Ax-main/scripts/update_versions_html.py -p "${WORK_DIR}"

  # Init as Git repo and push to gh-pages
  cd new-site || exit
  git init -b main
  git add --all
  git commit -m "Publish version ${VERSION} of site"
  git push --force "https://github.com/facebook/Ax" main:gh-pages

fi

# Clean up
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
