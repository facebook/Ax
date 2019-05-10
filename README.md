<img width="300" src="website/static/img/ax_logo_lockup.svg" alt="Ax Logo" />

<hr/>

[![Build Status](https://img.shields.io/pypi/v/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/pyversions/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/wheel/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://travis-ci.com/facebook/Ax.svg?token=m8nxq4QpA9U383aZWDyF&branch=master)](https://travis-ci.com/facebook/Ax)
[![codecov](https://codecov.io/gh/facebook/Ax/branch/master/graph/badge.svg)](https://codecov.io/gh/facebook/Ax)
[![Build Status](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

Ax is an accessible, general-purpose platform for understanding, managing,
deploying, and automating adaptive experiments.

Adaptive experimentation is the machine-learning guided process of iteratively
exploring a (possibly infinite) parameter space in order to identify optimal
configurations in a resource-efficient manner. Ax currently supports Bayesian
optimization and bandit optimization as exploration strategies. Bayesian
optimization in Ax is powered by [BoTorch](https://github.com/facebookexternal/botorch),
a modern library for Bayesian optimization research built on PyTorch.

For full documentation and tutorials, see the [Ax website](https://ax.dev)

## Why Ax?

* **Versatility**: Ax supports different kinds of experiments, from dynamic ML-assisted A/B testing, to hyperparameter optimization in machine learning.
* **Customization**: Ax makes it easy to add new modeling and decision algorithms, enabling research and development with minimal overhead.
* **Production-completeness**: Ax comes with storage integration and ability to fully save and reload experiments.
* **Support for multi-modal and constrained experimentation**: Ax allows for running and combining multiple experiments (e.g. simulation with a real-world "online" A/B test) and for constrained optimization (e.g. improving classification accuracy without signifant increase in resource-utilization).
* **Efficiency in high-noise setting**: Ax offers state-of-the-art algorithms specifically geared to noisy experiments, such as simulations with reinforcement-learning agents.
* **Ease of use**: Ax includes 3 different APIs that strike different balances between lightweight structure and flexibility. Using the most concise Loop API, a whole optimization can be done in just one function call. The Service API integrates easily with external schedulers. The most elaborate Developer API affords full algorithm customization and experiment introspection.

## Getting Started

To run a simple optimization loop in Ax (using the
[Booth response surface](https://www.sfu.ca/~ssurjano/booth.html) as the
artificial evaluation function):

```python
>>> from ax import optimize
>>> best_parameters, best_values, experiment, model = optimize(
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
        evaluation_function=lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
        minimize=True,
    )

# best_parameters contains {'x1': 1.02, 'x2': 2.97}; the global min is (1, 3)
```

## Installation

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
conda install pytorch torchvision -c pytorch  # OSX only
pip3 install ax-platform  # all systems
```

*Recommendation for MacOS users*: PyTorch is a required dependency of BoTorch, and can be automatically installed via pip.
However, we recommend you install PyTorch manually before installing Ax, using the Anaconda package manager.
Installing from Anaconda will link against MKL (a library that optimizes mathematical computation for Intel processors).
This will result in up to an order-of-magnitude speed-up for Bayesian optimization, as at the moment, installing PyTorch from pip does not link against MKL. **Currently, installation through Anaconda is temporarily required for OSX, as the pip installation of PyTorch is broken.**

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

## Join the Ax community

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out. You will
also need to install the dependencies needed for development, which are listed
in `DEV_REQUIRES` in `setup.py`, as follows:

```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[dev]
```

## License

Ax is licensed under the [MIT license](LICENSE.md).
