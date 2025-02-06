# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable, Mapping, Sequence

from dataclasses import dataclass, field, InitVar
from logging import Logger
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.problems.surrogate.lcbench.data import (
    DATASET_NAMES,
    load_lcbench_data,
)
from ax.benchmark.problems.surrogate.lcbench.transfer_learning import DEFAULT_NUM_TRIALS
from ax.benchmark.problems.surrogate.lcbench.utils import (
    DEFAULT_METRIC_NAME,
    get_lcbench_log_scale_parameter_names,
    get_lcbench_optimization_config,
    get_lcbench_parameter_names,
    get_lcbench_search_space,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.exceptions.core import UserInputError
from ax.utils.common.logger import get_logger

from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


logger: Logger = get_logger(__name__)

TRegressorProtocol = TypeVar("TRegressorProtocol", bound="RegressorProtocol")

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
OPTIMAL_VALUES: dict[str, float] = {
    "APSFailure": 98.97643280029295,
    "Amazon_employee_access": 94.1208953857422,
    "Australian": 92.1568603515625,
    "Fashion-MNIST": 89.417236328125,
    "KDDCup09_appetency": 98.33574676513672,
    "MiniBooNE": 90.79180908203124,
    "adult": 84.11111450195312,
    "airlines": 63.999061584472656,
    "albert": 66.25859832763672,
    "bank-marketing": 88.8466567993164,
    "blood-transfusion-service-center": 80.1204833984375,
    "car": 96.8668441772461,
    "christine": 75.04173278808594,
    "cnae-9": 97.48954010009766,
    "connect-4": 74.20671844482422,
    "covertype": 77.28044128417969,
    "credit-g": 77.47747802734375,
    "dionis": 87.68419647216797,
    "fabert": 67.32564544677734,
    "helena": 29.10163116455078,
    "higgs": 71.88652801513672,
    "jannis": 66.58744812011719,
    "jasmine": 82.60211944580078,
    "jungle_chess_2pcs_raw_endgame_complete": 84.13723754882812,
    "kc1": 87.36616516113281,
    "kr-vs-kp": 99.00990295410156,
    "mfeat-factors": 97.96839904785156,
    "nomao": 96.2865753173828,
    "numerai28.6": 52.32661819458008,
    "phoneme": 82.09204864501953,
    "segment": 90.60665130615234,
    "shuttle": 99.76605987548828,
    "sylvine": 94.97354125976562,
    "vehicle": 81.9148941040039,
    "volkert": 64.02699279785156,
}


class RegressorProtocol(Protocol):
    """
    A regressor that can fit and predict, such as `RandomForestRegressor`.
    """

    def fit(
        self: TRegressorProtocol, X: pd.DataFrame, y: pd.DataFrame | pd.Series
    ) -> TRegressorProtocol: ...
    def predict(self: TRegressorProtocol, X: pd.DataFrame) -> npt.NDArray: ...
    def set_params(self: TRegressorProtocol, **kwargs: Any) -> TRegressorProtocol: ...


def get_default_base_regressor() -> RegressorProtocol:
    return RandomForestRegressor(max_depth=30)


def _create_surrogate_regressor(
    base_regressor: RegressorProtocol,
    log_numeric_columns: Iterable,
    numeric_columns: Iterable,
    seed: int,
) -> RegressorProtocol:
    unit_scaler = MinMaxScaler()
    log_transformer = FunctionTransformer(
        func=np.log, inverse_func=np.exp, validate=True
    )

    log_numeric_transformer = make_pipeline(log_transformer, unit_scaler)

    preprocessor = make_column_transformer(
        (log_numeric_transformer, list(log_numeric_columns)),
        (unit_scaler, list(numeric_columns)),
        remainder="drop",
    )

    try:
        regressor = base_regressor.set_params(random_state=seed)
    except ValueError:
        # some models (e.g. K nearest neighbors) are deterministic by nature and do not
        # allow you to set a random seed
        logger.warning(
            f"Surrogate model `{base_regressor}` does not support specification of "
            "random seed, which *may* indicate that the model is already "
            "deterministic by nature. However, if you're unsure, this could lead to "
            "non-deterministic behavior in your experiments."
        )
        regressor = base_regressor

    return make_pipeline(
        preprocessor,
        TransformedTargetRegressor(regressor=regressor, transformer=log_transformer),
    )


@dataclass(kw_only=True)
class LearningCurveBenchmarkTestFunction(BenchmarkTestFunction):
    """A benchmark test function for LCBench early-stopping problems.

    This class represents a learning curve benchmark test function, which leverages a
    surrogate model trained to predict the performance of deep learning models at
    different stages of training. The test function takes in a set of hyperparameters
    and returns a tensor representing the predicted performance of the model at each
    stage (epoch) of training.

    To use this class, you would typically create an instance of it and pass it to a
    `BenchmarkProblem` along with a search space, optimization config, and other
    relevant parameters.

    Example:
        test_function = LearningCurveBenchmarkTestFunction(
            dataset_name="vehicle", seed=42
        )
        search_space = get_lcbench_search_space()
        optimization_config = get_lcbench_optimization_config(
            metric_name="Train/val_accuracy", observe_noise_sd=True, use_map_metric=True
        )
        problem = BenchmarkProblem(
            name="vehicle_Train/val_accuracy",
            search_space=search_space,
            optimization_config=optimization_config,
            test_function=test_function,
            step_runtime_function=None,
            ...
        )
    """

    n_steps: int = field(init=False)
    outcome_names: Sequence[str] = field(default_factory=lambda: [DEFAULT_METRIC_NAME])
    dataset_name: str
    metric_surrogate: RegressorProtocol = field(init=False)
    runtime_surrogate: RegressorProtocol = field(init=False)

    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    metric_base_surrogate: InitVar[RegressorProtocol] = get_default_base_regressor()
    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    runtime_base_surrogate: InitVar[RegressorProtocol] = get_default_base_regressor()
    # pyre-ignore [16]: Pyre doesn't understand InitVars.
    seed: InitVar[int]

    def __post_init__(
        self,
        metric_base_surrogate: RegressorProtocol,
        runtime_base_surrogate: RegressorProtocol,
        seed: int,
    ) -> None:
        if len(self.outcome_names) != 1:
            raise ValueError("Exactly one outcome is supported currently")

        metric_name = self.outcome_names[0]
        lcbench_data = load_lcbench_data(
            dataset_name=self.dataset_name,
            metric_name=metric_name,
            log_scale_parameter_names=[],
        )
        self.n_steps = lcbench_data.metric_df.shape[-1]

        parameter_names = get_lcbench_parameter_names()
        log_scale_parameter_names = get_lcbench_log_scale_parameter_names()
        numeric_columns = set(parameter_names) - set(log_scale_parameter_names)

        self.metric_surrogate = _create_surrogate_regressor(
            base_regressor=metric_base_surrogate,
            log_numeric_columns=log_scale_parameter_names,
            numeric_columns=numeric_columns,
            seed=seed,
        ).fit(X=lcbench_data.parameter_df, y=lcbench_data.metric_df)
        self.runtime_surrogate = _create_surrogate_regressor(
            base_regressor=runtime_base_surrogate,
            log_numeric_columns=log_scale_parameter_names,
            numeric_columns=numeric_columns,
            seed=seed,
        ).fit(X=lcbench_data.parameter_df, y=lcbench_data.average_runtime_series)

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> torch.Tensor:
        X = pd.DataFrame.from_records(data=[params])
        Y = self.metric_surrogate.predict(X)  # shape: (1, 50)
        return torch.from_numpy(Y)

    def step_runtime(self, params: Mapping[str, TParamValue]) -> float:
        X = pd.DataFrame.from_records(data=[params])
        Y = self.runtime_surrogate.predict(X)  # shape: (1,)
        return Y.item()


def get_lcbench_early_stopping_benchmark_problem(
    dataset_name: str,
    metric_name: str = DEFAULT_METRIC_NAME,
    num_trials: int = DEFAULT_NUM_TRIALS,
    constant_step_runtime: bool = False,
    noise_std: Mapping[str, float] | float = 0.0,
    observe_noise_sd: bool = False,
    seed: int = 0,
) -> BenchmarkProblem:
    """Construct an LCBench early-stopping benchmark problem.

    Args:
        dataset_name: Must be one of the keys of `DEFAULT_AND_OPTIMAL_VALUES`, which
            correspond to the names of the datasets available in LCBench.
        metric_name: The name of the metric to use for the objective.
        num_trials: The number of optimization trials to run.
        constant_step_runtime: Determines if the step runtime is fixed or varies
            based on the hyperparameters.
        noise_std: The standard deviation of the observation noise.
        observe_noise_sd: Whether to report the standard deviation of the
            obervation noise.
        seed: The random seed used in training the surrogate model to ensure
            reproducibility and consistency of results.

    Returns:
        An LCBench surrogate benchmark problem.
    """

    if dataset_name not in DATASET_NAMES:
        raise UserInputError(f"`dataset_name` must be one of {sorted(DATASET_NAMES)}")

    name = f"LCBench_Surrogate_{dataset_name}_{metric_name}:v1"

    optimal_value = OPTIMAL_VALUES[dataset_name]
    baseline_value = BASELINE_VALUES[dataset_name]

    search_space: SearchSpace = get_lcbench_search_space()
    optimization_config: OptimizationConfig = get_lcbench_optimization_config(
        metric_name=metric_name,
        observe_noise_sd=observe_noise_sd,
        use_map_metric=True,
    )

    test_function = LearningCurveBenchmarkTestFunction(
        dataset_name=dataset_name, seed=seed
    )

    step_runtime_function = (
        None if constant_step_runtime else test_function.step_runtime
    )

    return BenchmarkProblem(
        name=name,
        search_space=search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=optimal_value,
        baseline_value=baseline_value,
        test_function=test_function,
        step_runtime_function=step_runtime_function,
        noise_std=noise_std,
    )
