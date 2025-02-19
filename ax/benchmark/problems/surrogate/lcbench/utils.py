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

BASELINE_VALUES: dict[str, float] = {
    "APSFailure": 97.75948131763847,
    "Amazon_employee_access": 93.39364177908142,
    "Australian": 88.1445880383116,
    "Fashion-MNIST": 84.75904272864778,
    "KDDCup09_appetency": 96.13544312868322,
    "MiniBooNE": 85.8639428612948,
    "adult": 79.50334987749676,
    "airlines": 58.96099030718572,
    "albert": 63.885932360810884,
    "bank-marketing": 83.72755317459641,
    "blood-transfusion-service-center": 62.651717620524835,
    "car": 78.59464531457958,
    "christine": 72.22719165860138,
    "cnae-9": 92.24923138962973,
    "connect-4": 63.808749677494774,
    "covertype": 61.61393200315512,
    "credit-g": 70.45312807563056,
    "dionis": 53.71071232033245,
    "fabert": 64.44304132875557,
    "helena": 18.239085505279544,
    "higgs": 64.74999655474926,
    "jannis": 57.82155396833136,
    "jasmine": 80.48475426337272,
    "jungle_chess_2pcs_raw_endgame_complete": 65.58537332961572,
    "kc1": 77.28692486000287,
    "kr-vs-kp": 93.63368446446995,
    "mfeat-factors": 94.72758417873838,
    "nomao": 93.73968374826451,
    "numerai28.6": 51.60281273196557,
    "phoneme": 75.20979771001986,
    "segment": 78.81992685291081,
    "shuttle": 96.45744339531132,
    "sylvine": 91.15923021902736,
    "vehicle": 67.40729695042013,
    "volkert": 49.204981948803855,
}


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
