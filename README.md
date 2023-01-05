<img width="300" src="https://ax.dev/img/ax_logo_lockup.svg" alt="Ax Logo" />

<hr/>

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)
[![Build Status](https://img.shields.io/pypi/v/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/pyversions/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://img.shields.io/pypi/wheel/ax-platform.svg)](https://pypi.org/project/ax-platform/)
[![Build Status](https://github.com/facebook/Ax/workflows/Build%20and%20Test%20Workflow/badge.svg)](https://github.com/facebook/Ax/actions?query=workflow%3A%22Build+and+Test+Workflow%22)
[![codecov](https://codecov.io/gh/facebook/Ax/branch/main/graph/badge.svg)](https://codecov.io/gh/facebook/Ax)
[![Build Status](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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
* **Support for multi-modal and constrained experimentation**: Ax allows for running and combining multiple experiments (e.g. simulation with a real-world "online" A/B test) and for constrained optimization (e.g. improving classification accuracy without significant increase in resource-utilization).
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
You need Python 3.8 or later to run Ax.

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
pip install ax-platform
```

Installation will use Python wheels from PyPI, available for [OSX, Linux, and Windows](https://pypi.org/project/ax-platform/#files).

*Note*: Make sure the `pip` being used to install `ax-platform` is actually the one from the newly created Conda environment.
If you're using a Unix-based OS, you can use `which pip` to check.

*Recommendation for MacOS users*: PyTorch is a required dependency of BoTorch, and can be automatically installed via pip.
However, **we recommend you [install PyTorch manually](https://pytorch.org/get-started/locally/#anaconda-1) before installing Ax, using the Anaconda package manager**.
Installing from Anaconda will link against MKL (a library that optimizes mathematical computation for Intel processors).
This will result in up to an order-of-magnitude speed-up for Bayesian optimization, as at the moment, installing PyTorch from pip does not link against MKL.

If you need CUDA on MacOS, you will need to build PyTorch from source. Please consult the PyTorch installation instructions above.

#### Optional Dependencies

To use Ax with a notebook environment, you will need Jupyter. Install it first:
```
pip install jupyter
```

If you want to store the experiments in MySQL, you will need SQLAlchemy:
```
pip install SQLAlchemy
```

### Latest Version

#### Installing from Git

You can install the latest (bleeding edge) version from Git.

First, see recommendation for installing PyTorch for MacOS users above.

At times, the bleeding edge for Ax can depend on bleeding edge versions of BoTorch (or GPyTorch). We therefore recommend installing those from Git as well:

```
pip install git+https://github.com/cornellius-gp/linear_operator.git
pip install git+https://github.com/cornellius-gp/gpytorch.git
export ALLOW_LATEST_GPYTORCH_LINOP=true
pip install git+https://github.com/pytorch/botorch.git
export ALLOW_BOTORCH_LATEST=true
pip install git+https://github.com/facebook/Ax.git#egg=ax-platform
```

#### Optional Dependencies

If using Ax in Jupyter notebooks:

```
pip install git+https://github.com/facebook/Ax.git#egg=ax-platform[notebook]
```

To support plotly-based plotting in newer Jupyter notebook versions

```
pip install "notebook>=5.3" "ipywidgets==7.5"
```

[See Plotly repo's README](https://github.com/plotly/plotly.py#jupyter-notebook-support) for details and JupyterLab instructions.

If storing Ax experiments via SQLAlchemy in MySQL or SQLite:
```
pip install git+https://github.com/facebook/Ax.git#egg=ax-platform[mysql]
```

## Join the Ax Community

### Getting help

Please open an issue on our [issues page](https://github.com/facebook/Ax/issues) with any questions, feature requests or bug reports! If posting a bug report, please include a minimal reproducible example (as a code snippet) that we can use to reproduce and debug the problem you encountered.

### Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

When contributing to Ax, we recommend cloning the [repository](https://github.com/facebook/Ax) and installing all optional dependencies:

```
pip install git+https://github.com/cornellius-gp/linear_operator.git
pip install git+https://github.com/cornellius-gp/gpytorch.git
export ALLOW_LATEST_GPYTORCH_LINOP=true
pip install git+https://github.com/pytorch/botorch.git
export ALLOW_BOTORCH_LATEST=true
git clone https://github.com/facebook/ax.git --depth 1
cd ax
pip install -e .[notebook,mysql,dev]
```

See recommendation for installing PyTorch for MacOS users above.

The above example limits the cloned directory size via the
[`--depth`](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt)
argument to `git clone`. If you require the entire commit history you may remove this
argument.

## License

Ax is licensed under the [MIT license](./LICENSE).
