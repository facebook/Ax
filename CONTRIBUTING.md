# Contributing to Ax
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
Coming soon.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style  
We use the [`Black` code formatter](https://github.com/ambv/black) for all Python files. You can install the latest release via `pip install black` and run it over the library via `black ax`.

## Documentation

The Ax website was created with [Docusaurus](https://docusaurus.io/).
FontAwesome icons were used under the [Creative Commons Attribution 4.0 International](https://fontawesome.com/license).

### Building

You will need [Node](https://nodejs.org/en/) >= 8.x and [Yarn](https://yarnpkg.com/en/) >= 1.5
to build the Sphinx docs and Docusaurus site (which embeds the Sphinx docs inside). The
following command will both build the docs and serve the site locally:
```
cd scripts
./make_docs.sh
```

Open http://localhost:3000 (if doesn't automatically open).

Anytime you change the contents of the page, the page should auto-update.

### Publishing
The site is hosted as a GitHub page. Once Ax is live, we will generate a static
site and automatically push the output to the `gh-pages` branch via CircleCI.

## License
By contributing to Ax, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
