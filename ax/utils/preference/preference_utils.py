#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.torch import TorchAdapter
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import DataRequiredError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.utils.common.constants import Keys
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize


def get_preference_adapter(
    experiment: Experiment,
    data: Data,
) -> TorchAdapter:
    """Obtain a TorchAdapter from a preference experiment and data.

    Args:
        experiment: The preference experiment. The preference experiment consists
            of a list of parameters we wish to model the preference over and a single
            binary metric (see
            `ax.utils.common.constants.Keys.PAIRWISE_PREFERENCE_QUERY`)
            indicating the preference label.
        data: The preference data. It contains a collection of batch trials' data with
            exactly two observations in each batch trial, with one's metric
            being 1 (preferred) and the other's metric being 0 (not preferred),
            indicating pairwise comparisons.

    Returns:
        A PairwiseAdapter that wraps around a fitted BoTorch preference GP model,
        typically a PairwiseGP.

    Raises:
        DataRequiredError: If the provided data is empty. Preference data with
            at least one pairwise comparison is required to fit the preference model.
    """
    # Check for empty data before creating the adapter
    if data.df.empty:
        raise DataRequiredError(
            "No preference data available. At least one pairwise comparison is "
            "required to fit the preference model."
        )

    # Configure TorchAdapter for preference modeling:
    # - fit_tracking_metrics=False: Only fit on preference labels, not all metrics
    #   in the data. Requires optimization_config to specify which metrics to use.
    pref_metric = Metric(name=Keys.PAIRWISE_PREFERENCE_QUERY.value)
    optimization_config = OptimizationConfig(
        objective=Objective(metric=pref_metric, minimize=False)
    )
    # Register the metric on the experiment if not already present.
    # This is required for _extract_observation_data filtering in TorchAdapter.
    if pref_metric.name not in experiment.metrics:
        experiment.add_tracking_metric(pref_metric)

    # Setting up the preference adapter
    return TorchAdapter(
        experiment=experiment,
        search_space=experiment.search_space,
        data=data,
        optimization_config=optimization_config,
        fit_tracking_metrics=False,
        generator=BoTorchGenerator(
            # acqf doesn't matter. We only use the adapter for
            # data parsing and preference model construction
            # botorch_acqf_class=qNoisyExpectedImprovement,
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=PairwiseGP,
                        mll_class=PairwiseLaplaceMarginalLogLikelihood,
                        input_transform_classes=[Normalize],
                        input_transform_options={
                            "Normalize": {
                                "d": len(experiment.parameters),
                                "bounds": None,
                                "indices": None,
                            }
                        },
                    )
                ]
            )
        ),
        transforms=[],
    )
