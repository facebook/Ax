# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_problem import get_soo_opt_config
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace


DEFAULT_METRIC_NAME: str = "Train/val_accuracy"


def get_lcbench_search_space() -> SearchSpace:
    """Construct the LCBench search space."""
    search_space: SearchSpace = SearchSpace(
        parameters=[
            RangeParameter(
                name="batch_size",
                parameter_type=ParameterType.INT,
                lower=16,
                upper=512,
                log_scale=True,
            ),
            RangeParameter(
                name="max_dropout",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=1.0,  # Yes, really. Could make smaller if
                # we want to have it be more realistic.
                log_scale=False,
            ),
            RangeParameter(
                name="max_units",
                parameter_type=ParameterType.INT,
                lower=64,
                upper=1024,
                log_scale=True,
            ),
            RangeParameter(
                name="num_layers",
                parameter_type=ParameterType.INT,
                lower=1,
                upper=4,  # not a bug, even though it says 1-5 in the LCBench repo.
                # See https://github.com/automl/LCBench/issues/4
                log_scale=False,
            ),
            RangeParameter(
                name="learning_rate",
                parameter_type=ParameterType.FLOAT,
                lower=1e-4,
                upper=1e-1,
                log_scale=True,
            ),
            RangeParameter(
                name="momentum",
                parameter_type=ParameterType.FLOAT,
                lower=0.1,
                upper=0.99,
                log_scale=True,
            ),
            RangeParameter(
                name="weight_decay",
                parameter_type=ParameterType.FLOAT,
                lower=1e-5,
                upper=1e-1,
                log_scale=False,  # not a bug, see the LCBench repo.
            ),
        ]
    )
    return search_space


def get_lcbench_optimization_config(
    metric_name: str = DEFAULT_METRIC_NAME,
    observe_noise_sd: bool = False,
    use_map_metric: bool = False,
) -> OptimizationConfig:
    return get_soo_opt_config(
        outcome_names=[metric_name],
        lower_is_better=False,
        observe_noise_sd=observe_noise_sd,
        use_map_metric=use_map_metric,
    )


def get_lcbench_parameters() -> dict[str, RangeParameter]:
    return get_lcbench_search_space().parameters  # pyre-ignore [7]


def get_lcbench_parameter_names() -> list[str]:
    return list(get_lcbench_parameters().keys())


def get_lcbench_log_scale_parameter_names() -> list[str]:
    return [
        name
        for name, parameter in get_lcbench_parameters().items()
        if parameter.log_scale
    ]
