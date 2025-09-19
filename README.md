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
optimization in Ax is powered by
[BoTorch](https://github.com/facebookexternal/botorch), a modern library for
Bayesian optimization research built on PyTorch.

For full documentation and tutorials, see the [Ax website](https://ax.dev)

## Why Ax?

- **Expressive API**: Ax has an expressive API that can address many real-world
  optimization tasks. It handles complex search spaces, multiple objectives,
  constraints on both parameters and outcomes, and noisy observations. It
  supports suggesting multiple designs to evaluate in parallel (both
  synchronously and asynchronously) and the ability to early-stop evaluations.

- **Strong performance out of the box**: Ax abstracts away optimization details
  that are important but obscure, providing sensible defaults and enabling
  practitioners to leverage advanced techniques otherwise only accessible to
  optimization experts.

- **State-of-the-art methods**: Ax leverages state-of-the-art Bayesian
  optimization algorithms implemented in [BoTorch](https://botorch.org/), to
  deliver strong performance across a variety of problem classes.

- **Flexible:** Ax is highly configurable, allowing researchers to plug in novel
  optimization algorithms, models, and experimentation flows.

- **Production ready:** Ax offers automation and orchestration features as well
  as robust error handling for real-world deployment at scale.

## Getting Started

To run a simple optimization loop in Ax (using the
[Booth response surface](https://www.sfu.ca/~ssurjano/booth.html) as the
artificial evaluation function):

```python
>>> from ax import Client, RangeParameterConfig

>>> client = Client()
>>> client.configure_experiment(
      parameters=[
          RangeParameterConfig(
              name="x1",
              bounds=(-10.0, 10.0),
              parameter_type=ParameterType.FLOAT,
          ),
          RangeParameterConfig(
              name="x2",
              bounds=(-10.0, 10.0),
              parameter_type=ParameterType.FLOAT,
          ),
      ],
)
>>> client.configure_optimization(objective="-1 * booth")

>>> for _ in range(20):
>>>     for trial_index, parameters in client.get_next_trials(max_trials=1).items():
>>>         client.complete_trial(
>>>             trial_index=trial_index,
>>>             raw_data={
>>>                 "booth": (parameters["x1"] + 2 * parameters["x2"] - 7) ** 2
>>>                 + (2 * parameters["x1"] + parameters["x2"] - 5) ** 2
>>>             },
>>>         )

>>> client.get_best_parameterization()
```

## Installation

Ax requires Python 3.10 or newer. A full list of Ax's direct dependencies can be
found in [setup.py](https://github.com/facebook/Ax/blob/main/setup.py).

We recommend installing Ax via pip, even if using Conda environment:

```shell
pip install ax-platform
```

Installation will use Python wheels from PyPI, available for
[OSX, Linux, and Windows](https://pypi.org/project/ax-platform/#files).

_Note_: Make sure the `pip` being used to install `ax-platform` is actually the
one from the newly created Conda environment. If you're using a Unix-based OS,
you can use `which pip` to check.

### Installing with Extras

Ax can be installed with additional dependencies, which are not included in the
default installation. For example, in order to use Ax within a Jupyter notebook,
install Ax with the `notebook` extra:

```shell
pip install "ax-platform[notebook]"
```

Extras for using Ax with MySQL storage (`mysql`), for running Ax's tutorial's
locally (`tutorials`), and for installing all dependencies necessary for
developing Ax (`dev`) are also available.

## Install Ax from source

You can install the latest (bleeding edge) version from GitHub using `pip`.

The bleeding edge for Ax depends on bleeding edge versions of BoTorch and
GPyTorch. We therefore recommend installing those from Github, as well as
setting the following environment variables to allow the Ax to use the latest
version of both BoTorch and GPyTorch.

```shell
export ALLOW_LATEST_GPYTORCH_LINOP=true
export ALLOW_BOTORCH_LATEST=true

pip install git+https://github.com/cornellius-gp/gpytorch.git
pip install git+https://github.com/pytorch/botorch.git

pip install 'git+https://github.com/facebook/Ax.git#egg=ax-platform'
```

## Join the Ax Community

### Getting help

Please open an issue on our [issues page](https://github.com/facebook/Ax/issues)
with any questions, feature requests or bug reports! If posting a bug report,
please include a minimal reproducible example (as a code snippet) that we can
use to reproduce and debug the problem you encountered.

### Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

When contributing to Ax, we recommend cloning the
[repository](https://github.com/facebook/Ax) and installing all optional
dependencies:

```
pip install git+https://github.com/cornellius-gp/linear_operator.git
pip install git+https://github.com/cornellius-gp/gpytorch.git
export ALLOW_LATEST_GPYTORCH_LINOP=true
pip install git+https://github.com/pytorch/botorch.git
export ALLOW_BOTORCH_LATEST=true
git clone https://github.com/facebook/ax.git --depth 1
cd ax
pip install -e .[tutorial]
```

See recommendation for installing PyTorch for MacOS users above.

The above example limits the cloned directory size via the
[`--depth`](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt)
argument to `git clone`. If you require the entire commit history you may remove
this argument.

## Citing Ax

If you use Ax, please cite the following paper:
> [M. Olson, E. Santorella, L. C. Tiao, S. Cakmak, D. Eriksson, M. Garrard, S. Daulton, M. Balandat,  E. Bakshy, E. Kashtelyan, Z. J. Lin, S. Ament, B. Beckerman, E. Onofrey, P. Igusti, C. Lara, B. Letham, C. Cardoso, S. S. Shen, A. C. Lin, and M. Grange. Ax: A platform for Adaptive Experimentation. In AutoML 2025 ABCD Track, 2025.](https://openreview.net/forum?id=U1f6wHtG1g)

```
@inproceedings{olson2025ax,
  title = {{Ax: A Platform for Adaptive Experimentation}},
	author = {Olson, Miles and Santorella, Elizabeth and Tiao, Louis C. and Cakmak, Sait and Eriksson, David and Garrard, Mia and Daulton, Sam and Balandat, Maximilian and Bakshy, Eytan and Kashtelyan, Elena and Lin, Zhiyuan Jerry and Ament, Sebastian and Beckerman, Bernard and Onofrey, Eric and Igusti, Paschal and Lara, Cristian and Letham, Benjamin and Cardoso, Cesar and Shen, Shiyun Sunny and Lin, Andy Chenyuan and Grange, Matthew},
	booktitle = {AutoML 2025 ABCD Track},
	year = {2025}}
```

## License

Ax is licensed under the [MIT license](./LICENSE).
