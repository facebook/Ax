#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Type

import torch

# Ax `Acquisition` imports
from ax.models.torch.botorch_modular.acquisition import Acquisition

# BoTorch `AcquisitionFunction` imports
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SaasFullyBayesianSingleTaskGP

# BoTorch `Model` imports
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP

# Miscellaneous BoTorch imports
from gpytorch.constraints import Interval
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood

# BoTorch `MarginalLogLikelihood` imports
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.leave_one_out_pseudo_likelihood import LeaveOneOutPseudoLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior

# NOTE: When adding a new registry for a class, make sure to make changes
# to `CLASS_TO_REGISTRY` and `CLASS_TO_REVERSE_REGISTRY` in this file.

"""
Mapping of modular Ax `Acquisition` classes to class name strings.
"""
ACQUISITION_REGISTRY: Dict[Type[Acquisition], str] = {
    Acquisition: "Acquisition",
}


"""
Mapping of BoTorch `Model` classes to class name strings.
"""
MODEL_REGISTRY: Dict[Type[Model], str] = {
    FixedNoiseGP: "FixedNoiseGP",
    FixedNoiseMultiFidelityGP: "FixedNoiseMultiFidelityGP",
    FixedNoiseMultiTaskGP: "FixedNoiseMultiTaskGP",
    MixedSingleTaskGP: "MixedSingleTaskGP",
    ModelListGP: "ModelListGP",
    MultiTaskGP: "MultiTaskGP",
    SingleTaskGP: "SingleTaskGP",
    SingleTaskMultiFidelityGP: "SingleTaskMultiFidelityGP",
    SaasFullyBayesianSingleTaskGP: "SaasFullyBayesianSingleTaskGP",
}


"""
Mapping of Botorch `AcquisitionFunction` classes to class name strings.
"""
ACQUISITION_FUNCTION_REGISTRY: Dict[Type[AcquisitionFunction], str] = {
    ExpectedImprovement: "ExpectedImprovement",
    qExpectedHypervolumeImprovement: "qExpectedHypervolumeImprovement",
    qNoisyExpectedHypervolumeImprovement: "qNoisyExpectedHypervolumeImprovement",
    qExpectedImprovement: "qExpectedImprovement",
    qKnowledgeGradient: "qKnowledgeGradient",
    qMaxValueEntropy: "qMaxValueEntropy",
    qMultiFidelityKnowledgeGradient: "qMultiFidelityKnowledgeGradient",
    qMultiFidelityMaxValueEntropy: "qMultiFidelityMaxValueEntropy",
    qNoisyExpectedImprovement: "qNoisyExpectedImprovement",
}


"""
Mapping of BoTorch `MarginalLogLikelihood` classes to class name strings.
"""
MLL_REGISTRY: Dict[Type[MarginalLogLikelihood], str] = {
    ExactMarginalLogLikelihood: "ExactMarginalLogLikelihood",
    LeaveOneOutPseudoLikelihood: "LeaveOneOutPseudoLikelihood",
    SumMarginalLogLikelihood: "SumMarginalLogLikelihood",
}

LIKELIHOOD_REGISTRY: Dict[Type[GaussianLikelihood], str] = {
    GaussianLikelihood: "GaussianLikelihood"
}

GPYTORCH_COMPONENT_REGISTRY: Dict[Type[torch.nn.Module], str] = {
    Interval: "Interval",
    GammaPrior: "GammaPrior",
}

"""
Overarching mapping from encoded classes to registry map.
"""
# pyre-fixme[5]: Global annotation cannot contain `Any`.
CLASS_TO_REGISTRY: Dict[Any, Dict[Type[Any], str]] = {
    Acquisition: ACQUISITION_REGISTRY,
    AcquisitionFunction: ACQUISITION_FUNCTION_REGISTRY,
    Likelihood: LIKELIHOOD_REGISTRY,
    MarginalLogLikelihood: MLL_REGISTRY,
    Model: MODEL_REGISTRY,
    Interval: GPYTORCH_COMPONENT_REGISTRY,
    GammaPrior: GPYTORCH_COMPONENT_REGISTRY,
}


"""
Reverse registries for decoding.
"""
REVERSE_ACQUISITION_REGISTRY: Dict[str, Type[Acquisition]] = {
    v: k for k, v in ACQUISITION_REGISTRY.items()
}


REVERSE_MODEL_REGISTRY: Dict[str, Type[Model]] = {
    v: k for k, v in MODEL_REGISTRY.items()
}


REVERSE_ACQUISITION_FUNCTION_REGISTRY: Dict[str, Type[AcquisitionFunction]] = {
    v: k for k, v in ACQUISITION_FUNCTION_REGISTRY.items()
}


REVERSE_MLL_REGISTRY: Dict[str, Type[MarginalLogLikelihood]] = {
    v: k for k, v in MLL_REGISTRY.items()
}

REVERSE_LIKELIHOOD_REGISTRY: Dict[str, Type[Likelihood]] = {
    v: k for k, v in LIKELIHOOD_REGISTRY.items()
}

REVERSE_GPYTORCH_COMPONENT_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    v: k for k, v in GPYTORCH_COMPONENT_REGISTRY.items()
}

"""
Overarching mapping from encoded classes to reverse registry map.
"""
# pyre-fixme[5]: Global annotation cannot contain `Any`.
CLASS_TO_REVERSE_REGISTRY: Dict[Any, Dict[str, Type[Any]]] = {
    Acquisition: REVERSE_ACQUISITION_REGISTRY,
    AcquisitionFunction: REVERSE_ACQUISITION_FUNCTION_REGISTRY,
    Likelihood: REVERSE_LIKELIHOOD_REGISTRY,
    MarginalLogLikelihood: REVERSE_MLL_REGISTRY,
    Model: REVERSE_MODEL_REGISTRY,
    Interval: REVERSE_GPYTORCH_COMPONENT_REGISTRY,
    GammaPrior: REVERSE_GPYTORCH_COMPONENT_REGISTRY,
}


def register_acquisition(acq_class: Type[Acquisition]) -> None:
    """Add a custom acquisition class to the SQA and JSON registries."""
    class_name = acq_class.__name__
    CLASS_TO_REGISTRY[Acquisition].update({acq_class: class_name})
    CLASS_TO_REVERSE_REGISTRY[Acquisition].update({class_name: acq_class})


def register_acquisition_function(acqf_class: Type[AcquisitionFunction]) -> None:
    """Add a custom acquisition class to the SQA and JSON registries."""
    class_name = acqf_class.__name__
    CLASS_TO_REGISTRY[AcquisitionFunction].update({acqf_class: class_name})
    CLASS_TO_REVERSE_REGISTRY[AcquisitionFunction].update({class_name: acqf_class})
