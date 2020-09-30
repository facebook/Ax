#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Type

# Ax `Acquisition` imports
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kg import (
    KnowledgeGradient,
    MultiFidelityKnowledgeGradient,
)
from ax.models.torch.botorch_modular.mes import (
    MaxValueEntropySearch,
    MultiFidelityMaxValueEntropySearch,
)

# BoTorch `AcquisitionFunction` imports
from botorch.acquisition.acquisition import AcquisitionFunction
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

# BoTorch `Model` imports
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.model import Model

# BoTorch `MarginalLogLikelihood` imports
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


# NOTE: When adding a new registry for a class, make sure to make changes
# to `CLASS_TO_REGISTRY` and `CLASS_TO_REVERSE_REGISTRY` in this file.

"""
Mapping of modular Ax `Acquisition` classes to ints.
"""
ACQUISITION_REGISTRY: Dict[Type[Acquisition], int] = {
    Acquisition: 0,
    KnowledgeGradient: 1,
    MultiFidelityKnowledgeGradient: 2,
    MaxValueEntropySearch: 3,
    MultiFidelityMaxValueEntropySearch: 4,
}


"""
Mapping of BoTorch `Model` classes to ints.
"""
MODEL_REGISTRY: Dict[Type[Model], int] = {
    FixedNoiseGP: 0,
    SingleTaskGP: 1,
    FixedNoiseMultiFidelityGP: 2,
    SingleTaskMultiFidelityGP: 3,
}


"""
Mapping of Botorch `AcquisitionFunction` classes to ints.
"""
ACQUISITION_FUNCTION_REGISTRY: Dict[Type[AcquisitionFunction], int] = {
    qExpectedImprovement: 0,
    qNoisyExpectedImprovement: 1,
    qKnowledgeGradient: 2,
    qMultiFidelityKnowledgeGradient: 3,
    qMaxValueEntropy: 4,
    qMultiFidelityMaxValueEntropy: 5,
}


"""
Mapping of BoTorch `MarginalLogLikelihood` classes to ints.
"""
MLL_REGISTRY: Dict[Type[MarginalLogLikelihood], int] = {ExactMarginalLogLikelihood: 0}


"""
Overarching mapping from encoded classes to registry map.
"""
CLASS_TO_REGISTRY: Dict[Any, Dict[Type[Any], int]] = {
    Acquisition: ACQUISITION_REGISTRY,
    AcquisitionFunction: ACQUISITION_FUNCTION_REGISTRY,
    MarginalLogLikelihood: MLL_REGISTRY,
    Model: MODEL_REGISTRY,
}


"""
Reverse registries for decoding.
"""
REVERSE_ACQUISITION_REGISTRY: Dict[int, Type[Acquisition]] = {
    v: k for k, v in ACQUISITION_REGISTRY.items()
}


REVERSE_MODEL_REGISTRY: Dict[int, Type[Model]] = {
    v: k for k, v in MODEL_REGISTRY.items()
}


REVERSE_ACQUISITION_FUNCTION_REGISTRY: Dict[int, Type[AcquisitionFunction]] = {
    v: k for k, v in ACQUISITION_FUNCTION_REGISTRY.items()
}


REVERSE_MLL_REGISTRY: Dict[int, Type[MarginalLogLikelihood]] = {
    v: k for k, v in MLL_REGISTRY.items()
}


"""
Overarching mapping from encoded classes to reverse registry map.
"""
CLASS_TO_REVERSE_REGISTRY: Dict[Any, Dict[int, Type[Any]]] = {
    Acquisition: REVERSE_ACQUISITION_REGISTRY,
    AcquisitionFunction: REVERSE_ACQUISITION_FUNCTION_REGISTRY,
    MarginalLogLikelihood: REVERSE_MLL_REGISTRY,
    Model: REVERSE_MODEL_REGISTRY,
}
