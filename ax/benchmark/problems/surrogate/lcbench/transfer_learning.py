# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from collections.abc import Mapping
from typing import Any

import torch
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.benchmark.problems.surrogate.lcbench.utils import (
    BASELINE_VALUES,
    DEFAULT_METRIC_NAME,
    get_lcbench_optimization_config,
)
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Cont_X_trans, Generators, Y_trans
from ax.modelbridge.torch import TorchAdapter
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.models.torch.botorch_modular.model import BoTorchGenerator
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.testing.mock import skip_fit_gpytorch_mll_context_manager
from botorch.models import SingleTaskGP
from gpytorch.priors import LogNormalPrior

from pyre_extensions import assert_is_instance


DEFAULT_NUM_TRIALS: int = 30


DEFAULT_AND_OPTIMAL_VALUES: dict[str, tuple[float, float]] = {
    "KDDCup09_appetency": (87.14437173839048, 100.41903197808242),
    "APSFailure": (97.3412499690734, 98.38099041845653),
    "albert": (64.42693765555859, 67.1082934765708),
    "Amazon_employee_access": (80.69975381128579, 98.85943103737361),
    "Australian": (78.15200826093329, 93.0325039665508),
    "Fashion-MNIST": (83.10219231927393, 89.07884250211491),
    "car": (64.26087451215653, 88.77391803474296),
    "christine": (70.9732126619125, 73.29816335805616),
    "cnae-9": (25.89740105397502, 119.55228152861949),
    "covertype": (62.13132918760403, 67.15439170116016),
    "dionis": (11.977294194995338, 101.64303302727558),
    "fabert": (37.72627877151164, 73.30035354875776),
    "helena": (7.455048985077637, 29.78291566900156),
    "higgs": (64.80984463924982, 71.89770865111743),
    "jannis": (58.17868556972097, 62.4080058894946),
    "jasmine": (76.76806487249725, 83.56868288456046),
    "kr-vs-kp": (79.76060013094786, 104.6216855876375),
    "mfeat-factors": (69.85128706899793, 111.67026074027292),
    "nomao": (92.85065022473196, 95.83471144381221),
    "shuttle": (98.86272845879327, 100.37428346365724),
    "sylvine": (83.1596613771663, 98.85179841137813),
    "volkert": (45.361097364985376, 58.133196667029864),
}


def get_lcbench_surrogate() -> Surrogate:
    """Construct a surrogate used to fit the LCBench data.

    Returns:
        A Surrogate with the specification used to fit the LCBench data.
    """
    return Surrogate(
        botorch_model_class=SingleTaskGP,
        covar_module_class=ScaleMaternKernel,
        covar_module_options={
            "nu": 1.5,
            "ard_num_dims": 7,
            "outputscale_prior": LogNormalPrior(-3, 0.0025),
        },
        input_transform_classes=None,
    )


def get_lcbench_benchmark_problem(
    dataset_name: str,
    metric_name: str = DEFAULT_METRIC_NAME,
    num_trials: int = DEFAULT_NUM_TRIALS,
    noise_stds: Mapping[str, float] | float = 0.0,
    observe_noise_stds: bool = False,
) -> BenchmarkProblem:
    """Construct a LCBench benchmark problem.

    Args:
        dataset_name: Must be one of the keys of `DEFAULT_AND_OPTIMAL_VALUES`, which
            correspond to the names of the dataset available in LCBench.
        metric_name: The name of the metric to use for the objective.
        num_trials: The number of optimization trials to run.
        noise_stds: The standard deviation of the observation noise.
        observe_noise_stds: Whether to report the standard deviation of the
            observation noise.

    Returns:
        An LCBench surrogate benchmark problem.
    """

    if dataset_name not in DEFAULT_AND_OPTIMAL_VALUES:
        raise UserInputError(
            f"`dataset_name` must be one of {sorted(DEFAULT_AND_OPTIMAL_VALUES)}"
        )
    _, optimal_value = DEFAULT_AND_OPTIMAL_VALUES[dataset_name]
    base_path = os.path.dirname(os.path.realpath(__file__))
    obj: dict[str, Any] = torch.load(
        f=os.path.join(
            base_path, "transfer_learning_data", f"lcbench_{dataset_name}.pt"
        ),
        weights_only=False,
    )
    optimization_config: OptimizationConfig = get_lcbench_optimization_config(
        metric_name=metric_name,
        observe_noise_sd=observe_noise_stds,
        use_map_metric=False,
    )

    def get_surrogate() -> TorchAdapter:
        """Construct a modelbridge with the LCBench surrogate and datasets.

        Returns:
            A fitted modelbridge with the LCBench Surrogate and data.
        """
        # We load the model hyperparameters from the saved state dict.
        with skip_fit_gpytorch_mll_context_manager():
            mb = Generators.BOTORCH_MODULAR(
                surrogate=get_lcbench_surrogate(),
                experiment=obj["experiment"],
                search_space=obj["experiment"].search_space,
                data=obj["data"],
                transforms=Cont_X_trans + Y_trans,
            )
        assert_is_instance(mb.model, BoTorchGenerator).surrogate.model.load_state_dict(
            obj["state_dict"]
        )
        return assert_is_instance(mb, TorchAdapter)

    name = f"LCBench_Surrogate_{dataset_name}:v1"

    test_function = SurrogateTestFunction(
        name=name, outcome_names=[metric_name], get_surrogate=get_surrogate
    )

    return BenchmarkProblem(
        name=name,
        search_space=obj["experiment"].search_space,
        optimization_config=optimization_config,
        num_trials=num_trials,
        optimal_value=optimal_value,
        baseline_value=BASELINE_VALUES[dataset_name],
        test_function=test_function,
        noise_std=0.0 if noise_stds is None else noise_stds,
    )
