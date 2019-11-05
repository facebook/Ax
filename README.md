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
* sklearn
* plotly >=2.2.1

### Stable Version

#### Installing via pip
We recommend installing Ax via pip (even if using Conda environment):

```
conda install pytorch torchvision -c pytorch  # OSX only (details below)
pip3 install ax-platform
```

Installation will use Python wheels from PyPI, available for [OSX, Linux, and Windows](https://pypi.org/project/ax-platform/#files).

*Recommendation for MacOS users*: PyTorch is a required dependency of BoTorch, and can be automatically installed via pip.
However, **we recommend you [install PyTorch manually](https://pytorch.org/get-started/locally/#anaconda-1) before installing Ax, using the Anaconda package manager**.
Installing from Anaconda will link against MKL (a library that optimizes mathematical computation for Intel processors).
This will result in up to an order-of-magnitude speed-up for Bayesian optimization, as at the moment, installing PyTorch from pip does not link against MKL.

If you need CUDA on MacOS, you will need to build PyTorch from source. Please consult the PyTorch installation instructions above.

#### Optional Dependencies

To use Ax with a notebook environment, you will need Jupyter. Install it first:
```
pip3 install jupyter
```

If you want to store the experiments in MySQL, you will need SQLAlchemy:
```
pip3 install SQLAlchemy
```

### Latest Version

#### Installing from Git

You can install the latest (bleeding edge) version from Git:

```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax
```

See recommendation for installing PyTorch for MacOS users above.

At times, the bleeding edge for Ax can depend on bleeding edge versions of BoTorch (or GPyTorch). We therefore recommend installing those from Git as well:
```
pip3 install git+https://github.com/cornellius-gp/gpytorch.git
pip3 install git+https://github.com/pytorch/botorch.git
```

#### Optional Dependencies

If using Ax in Jupyter notebooks:

```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[notebook]
```

To support plotly-based plotting in newer Jupyter notebook versions 

```
pip install "notebook>=5.3" "ipywidgets==7.5"
```

[See Plotly repo's README](https://github.com/plotly/plotly.py#jupyter-notebook-support) for details and JupyterLab instructions.

If storing Ax experiments via SQLAlchemy in MySQL or SQLite:
```
pip3 install git+ssh://git@github.com/facebook/Ax.git#egg=Ax[mysql]
```

## Join the Ax Community
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

When contributing to Ax, we recommend cloning the [repository](https://github.com/facebook/Ax) and installing all optional dependencies:

```
# bleeding edge versions of GPyTorch + BoTorch are recommended
pip3 install git+https://github.com/cornellius-gp/gpytorch.git
pip3 install git+https://github.com/pytorch/botorch.git

git clone https://github.com/facebook/ax.git
cd ax
pip3 install -e .[notebook,mysql,dev]
```

See recommendation for installing PyTorch for MacOS users above.

## License

Ax is licensed under the [MIT license](LICENSE.md).
