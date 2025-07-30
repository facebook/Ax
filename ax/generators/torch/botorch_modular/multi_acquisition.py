#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError

from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.surrogate import Surrogate
from ax.generators.torch_base import TorchOptConfig
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multioutput_acquisition import (
    MultiOutputAcquisitionFunctionWrapper,
)


class MultiAcquisition(Acquisition):
    """
    A MultiAcquisition class for generating points by optimizing multiple
    acquisition functions jointly.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction] | None,
        botorch_acqf_options: dict[str, Any] | None = None,
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ]
        | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        if botorch_acqf_classes_with_options is None:
            raise AxError(
                "botorch_acqf_classes_with_options must be specified for "
                "MultiAcquisition."
            )
        elif len(botorch_acqf_classes_with_options) < 2:
            raise AxError(
                "botorch_acqf_classes_with_options have at least two elements."
            )
        super().__init__(
            surrogate=surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=botorch_acqf_class,
            botorch_acqf_options=botorch_acqf_options,
            botorch_acqf_classes_with_options=botorch_acqf_classes_with_options,
            options=options,
        )

    def _instantiate_acquisition(
        self,
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ],
    ) -> None:
        """Constructs the acquisition function based on the provided AF clases.

        Args:
            botorch_acqf_classes: A list of BoTorch acquisition function classes.
        """
        acqfs = [
            self._construct_botorch_acquisition(
                botorch_acqf_class=botorch_acqf_class,
                botorch_acqf_options=botorch_acqf_options,
            )
            for botorch_acqf_class, botorch_acqf_options in (
                botorch_acqf_classes_with_options
            )
        ]
        self.acqf = MultiOutputAcquisitionFunctionWrapper(acqfs=acqfs)
