---
id: installation
title: Installation
---
## Requirements

You need Python 3.10 or later to run Ax.

The required Python dependencies are:

-   [botorch][def]
-   jinja2
-   pandas
-   scipy
-   sklearn
-   plotly &gt;=2.2.1

## Stable Version

### Installing via pip

We recommend installing Ax via pip (even if using Conda environment):

```shell
conda install pytorch torchvision -c pytorch  # OSX only (details below)
pip install ax-platform
```

Installation will use Python wheels from PyPI, available for [OSX, Linux, and Windows](https://pypi.org/project/ax-platform/#files).

_Note_: Make sure the `pip` being used to install `ax-platform` is actually the one from the newly created Conda environment.
If you're using a Unix-based OS, you can use `which pip` to check.

_Recommendation for MacOS users_: PyTorch is a required dependency of BoTorch, and can be automatically installed via pip.
However, **we recommend you [install PyTorch manually](https://pytorch.org/get-started/locally/#anaconda-1) before installing Ax, using the Anaconda package manager**.
Installing from Anaconda will link against MKL (a library that optimizes mathematical computation for Intel processors).
This will result in up to an order-of-magnitude speed-up for Bayesian optimization, whereas installing PyTorch from pip does not link against MKL.

If you need CUDA on MacOS, you will need to build PyTorch from source. Please consult the PyTorch installation instructions above.

### Optional Dependencies

To use Ax with a notebook environment, you will need Jupyter. Install it first:

```shell
pip install jupyter
```

If you want to store the experiments in MySQL, you will need SQLAlchemy:

```shell
pip install SQLAlchemy
```

## Latest Version

### Installing from Git

You can install the latest (bleeding edge) version from GitHub:

```shell
pip install 'git+https://github.com/facebook/Ax.git#egg=ax-platform'
```

See also the recommendation for installing PyTorch for MacOS users above.

At times, the bleeding edge for Ax can depend on bleeding edge versions of BoTorch (or GPyTorch). We therefore recommend installing those from Git as well:

```shell
pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git
```

### Optional Dependencies

To use Ax with a notebook environment, you will need Jupyter. Install it first:

```shell
pip install 'git+https://github.com/facebook/Ax.git#egg=ax-platform[notebook]'
```

If storing Ax experiments via SQLAlchemy in MySQL or SQLite:

```shell
pip install 'git+https://github.com/facebook/Ax.git#egg=ax-platform[mysql]'
```

## Development

When contributing to Ax, we recommend cloning the [repository](https://github.com/facebook/Ax) and installing all optional dependencies:

```shell
# bleeding edge versions of GPyTorch + BoTorch are recommended
pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git

git clone https://github.com/facebook/ax.git --depth 1
cd ax
pip install -e .[notebook,mysql,dev]
```

See recommendation for installing PyTorch for MacOS users above.

The above example limits the cloned directory size via the
[`--depth`](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt)
argument to `git clone`. If you require the entire commit history you may remove this
argument.

[def]: https://www.botorch.org
