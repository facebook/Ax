---
id: installation
title: Installation
---

### Requirements
You need Python 3.6 or later to run Ax.

The required Python dependencies are:

* [botorch](https://www.botorch.org)
* jinja2
* pandas
* scipy
* simplejson
* sklearn
* plotly >=2.2.1, <3.0

### Installation via pip
We recommend installing Ax via pip.
To do so, run:

```
pip3 install ax-platform
```

*Recommendation for MacOS users*: PyTorch is a required dependency of BoTorch, and can be automatically installed via pip.
However, **we recommend you [install PyTorch manually](https://pytorch.org/get-started/locally/#anaconda-1) before installing Ax, using the Anaconda package manager**.
Installing from Anaconda will link against MKL (a library that optimizes mathematical computation for Intel processors).
This will result in up to an order-of-magnitude speed-up for Bayesian optimization, as at the moment, installing PyTorch from pip does not link against MKL.

### Installing from source
To install from source:
1. Make sure you have [installed the botorch dependency](https://www.botorch.org/docs/getting_started/#installing-botorch).
1. Download Ax from the [Git repository](https://github.com/facebook/Ax).
1. `cd` into the `ax` project and run:

```
pip3 install -e .
```

*Note:* When installing from source, Ax requires a compiler for Cython code.

### Optional Dependencies
Depending on your intended use of Ax, you may want to install Ax with optional dependencies.

If using Ax in Jupyter notebooks:
```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[notebook]
```
If storing Ax experiments via SQLAlchemy in MySQL or SQLite:

```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[mysql]
```
Note that instead of installation from Git, you can also clone a local version of the repo and then pip install with desired flags from the root of the local repo, e.g.:

`pip3 install -e .[mysql]`
