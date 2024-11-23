# Contributing to Ax
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
  For the most seamless developer experience, we recommend that you install
  all optional dependencies for Ax.
2. Please see the [Code Requirements](#code-requirements) section for requirements on unit testing, code style, documentation, etc. These requirements are enforced via Travis for each PR.
3. If you haven't already, complete the Contributor License Agreement ("CLA").

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

## Code Requirements

The Ax codebase has a high standard for code quality, which we enforce via Travis.

### Coding Style
We use the [`Ruff` code formatter](https://docs.astral.sh/ruff/formatter/) for all Python files. You can install the latest release via `pip install ruff` and run it over the library via `ruff format ax`.

### Unit Tests
The majority of our code is covered by unit tests and we are working to get to 100% code coverage. Please ensure that new code is covered by unit tests. To run all unit tests, we recommend installing pytest using `pip install pytest` and running `pytest -ra` from the root of the Ax repo. To get coverage, `pip install pytest-cov` and run `pytest -ra --cov=ax`.

#### Code Style

Ax uses [ufmt](https://github.com/omnilib/ufmt) to enforce consistent code
formatting (based on [black](https://github.com/ambv/black)) and import sorting
(based on [µsort](https://github.com/facebook/usort)) across the code base.
Install via `pip install ufmt`, and auto-format and auto-sort by running

```bash
ufmt format .
```

from the repository root.

#### Flake8 linting

Ax uses `flake8` for linting. To run the linter locally, install `flake8`
via `pip install flake8`, and then run

```bash
flake8 .
```

from the repository root.

#### Pre-commit hooks

Contributors can use [pre-commit](https://pre-commit.com/) to run `ufmt` and
`flake8` as part of the commit process. To install the hooks, install `pre-commit`
via `pip install pre-commit` and run `pre-commit install` from the repository
root.

### Static Type Checking
We use [Pyre](https://pyre-check.org/) for static type checking and require code to be fully type annotated. At the moment, static type checking is not supported within Travis.

### Documentation
* We require docstrings on all public functions and classes (those not prepended with `_`).
* We use the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) & use Sphinx to compile the complete API documentation.
* To rule out parsing errors, you can run Sphinx from the Ax root via `sphinx-build -WT sphinx/source sphinx/build`. The `-W` flag will cause Sphinx to break on the first error.
* If adding a new module to Ax, please add it to the appropriate Sphinx reStructuredText (rst) file (see [here](https://github.com/facebook/Ax/tree/main/sphinx/source)). We have a check in Travis for un-documented modules.

## Website

The Ax website was created with [Docusaurus](https://docusaurus.io/), with heavy customization to support tutorials, complete API documentation via Sphinx, and versioning.

FontAwesome icons were used under the [Creative Commons Attribution 4.0 International](https://fontawesome.com/license).

### Local Building

You will need [Node](https://nodejs.org/en/) >= 8.x and [Yarn](https://yarnpkg.com/en/) >= 1.5
to build the Sphinx docs and Docusaurus site (which embeds the Sphinx docs inside). The
following command will both build the docs and serve the (unversioned) site locally:
```
./scripts/make_docs.sh
```

Open http://localhost:3000 (if doesn't automatically open). Anytime you change the contents of the page, the page should auto-update.

To build a static version, add the `-b` flag.

Additional details:

* This is the unversioned site built off of the current repository. Versioning is much more complex, and is generally not necessary for testing the site. Versioning is automatically handled when publishing the site to `gh-pages`.
* Skipping Sphinx API & tutorials: to build the site without running Sphinx or compiling the tutorials, pass the `-o` flag to the `make_docs.sh` script. This is especially useful when you want to iterate quickly on making changes to the Docusaurus config and you've already run Sphinx and tutorial generation (the outputs are still in `website` subdirectory and will be picked up).
* Tutorials: we embed tutorials written in Jupyter notebooks into the site. By default, these tutorials are converted to HTML without execution. However, you can execute all tutorials via `./scripts/make_docs.sh -t`. If you do execute the tutorials, please keep in mind that the version of Ax you have installed should match the version of Ax you're trying to build tutorials for.

### Publishing
The site is hosted as a GitHub page (on the `gh-pages` branch). We build the [latest version](https://ax.dev/versions/latest/index.html) of the site with every commit to the `main` branch via GitHub Actions. The latest version of the site can be manually updated using `./scripts/publish_site.sh` (assuming proper credentials).

When new version of Ax rolls out, we add a new version to the site via `./scripts/publish_site.sh -v [version]`.

## License
By contributing to Ax, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
