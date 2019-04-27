<img width="300" src="./ax_logo_lockup.svg" alt="Ax Logo" />

<hr/>

[![Build Status](https://img.shields.io/pypi/v/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/pyversions/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/wheel/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://travis-ci.com/facebook/Ax.svg?token=m8nxq4QpA9U383aZWDyF&branch=master)](https://travis-ci.com/facebook/Ax)
[![Build Status](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

Ax is an accessible, general-purpose platform for understanding, managing,
deploying, and automating adaptive experiments.

Adaptive experimentation is the machine-learning guided process of iteratively
exploring a (possibly infinite) parameter space in order to identify optimal
configurations in a resource-efficient manner. Ax currently supports Bayesian
optimization and bandit optimization as exploration strategies. Bayesian
optimization in Ax is powered by [BoTorch](https://github.com/facebookexternal/botorch),
a modern library for Bayesian optimization research built on PyTorch.

For full documentation and tutorials, see the Ax website [TODO: add link].

## Installation

### Requirements

You need **Python 3.6 or later** to run Ax.

The required Python dependencies are:

* botorch
* jinja2
* pandas
* scipy
* simplejson
* sklearn
* plotly==2.4.1  **# TODO!**

### pip [PRIOR TO LAUNCH]

NOTE: Both BoTorch and Ax are currently private repositories.
This means that to download them, using `pip`, you need to make sure that
you have an [SSH key is registered with GitHub](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/).

1) Install numpy and cython, if you don't already have them, since they are
needed for parsing the `setup.py` files for Ax:

```
pip3 install cython numpy
```

2) Install PyTorch nightly build

See installation instructions via pip or conda [here](https://pytorch.org/get-started/locally/).

BoTorch uses advanced PyTorch features and thus currently requires PyTorch's
nightly build (the requirement upon beta release will be PyTorch 1.1).

*Recommendation for MacOS users*: install PyTorch before installing BoTorch
using the [Anaconda package manager](https://pytorch.org/get-started/locally/#anaconda-1)
to get up to an order-of-magnitude speed-up for Bayesian optimization. At the
moment, installing PyTorch from pip does not link against [MKL](https://github.com/intel/mkl-dnn),
a library that optimizes mathematical computation for Intel processors.


3) Download and install BoTorch from the [GitHub repo](https://github.com/pytorch/botorch):
```
pip3 install git+ssh://git@github.com/pytorch/botorch.git
```


4) Download and install Ax from the [GitHub repo](https://github.com/facebook/Ax):
```
pip3 install git+ssh://git@github.com/facebook/Ax.git
```


### pip (post-launch; TODO: set up wheels)

Install Ax via pip:
```
pip3 install ax-platform
```

*Recommendation for MacOS users*: install PyTorch before Ax by using the
[Anaconda package manager](https://pytorch.org/get-started/locally/#anaconda-1)
to get up to an order-of-magnitude speed-up for Bayesian optimization. At the
moment, installing PyTorch from pip does not link against
[MKL](https://software.intel.com/en-us/mkl), a library that optimizes
mathematical computation for Intel processors.

### Optional Dependencies

Depending on your intended use of Ax, you may want to install Ax with optional
dependencies.

If using Ax in Jupyter notebooks:
```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[notebook]
```

If storing Ax experiments via SQLAlchemy in MySQL or SQLite:
```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[mysql]
```

Note that instead of installation from Git, you can also clone
a local version of the repo and then pip install with desired
flags from the root of the local repo, e.g.:
```
pip3 install -e .[mysql]
```

## Getting Started

To run a simple optimization loop in Ax (using the
[Booth response surface](https://www.sfu.ca/~ssurjano/booth.html) as the
artificial evaluation function):

```
>>> from ax import optimize
>>> optimize(
        parameters=[
          {
            "name": "x1",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
          {
            "name": "x2",
            "type": "range",
            "bounds": [-10.0, 10.0],
          },
        ],
        # Booth function
        evaluation_function=lambda p: p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5,
        minimize=True,
    )

{'x1': 1.02, 'x2': 2.97}  # global min is (1, 3)
```

## Join the Ax community

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out. You will
also need to install the dependencies needed for development, which are listed
in `DEV_REQUIRES` in `setup.py`, as follows:

```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[dev]
```

## License

Ax is licensed under the [MIT license](LICENSE.md).
