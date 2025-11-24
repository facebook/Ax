#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.torch import TorchAdapter
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
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
    """

    # Setting up the preference adapter
    return TorchAdapter(
        experiment=experiment,
        search_space=experiment.search_space,
        data=data,
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
