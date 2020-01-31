#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.modelbridge.factory import DEFAULT_TORCH_DEVICE
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.centered_unit_x import CenteredUnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.models.random.alebo_initializer import ALEBOInitializer
from ax.models.torch.alebo import ALEBO


def get_ALEBOInitializer(
    search_space: SearchSpace, B: np.ndarray, **model_kwargs: Any
) -> RandomModelBridge:
    return RandomModelBridge(
        search_space=search_space,
        model=ALEBOInitializer(B=B, **model_kwargs),
        transforms=[CenteredUnitX],
    )


def get_ALEBO(
    experiment: Experiment,
    search_space: SearchSpace,
    data: Data,
    B: torch.Tensor,
    **model_kwargs: Any,
) -> TorchModelBridge:
    if search_space is None:
        search_space = experiment.search_space
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space,
        data=data,
        model=ALEBO(B=B, **model_kwargs),
        transforms=[CenteredUnitX, StandardizeY],
        torch_dtype=B.dtype,
        torch_device=B.device,
    )


class ALEBOStrategy(GenerationStrategy):
    """Generation strategy for Adaptive Linear Embedding BO.

    Both quasirandom initialization and BO are done with the same random
    projection. All function evaluations are done within that projection.

    Args:
        D: Dimensionality of high-dimensional space
        d: Dimensionality of low-dimensional space
        init_size: Size of random initialization
        name: Name of strategy
        dtype: torch dtype
        device: torch device
        random_kwargs: kwargs passed along to random model
        gp_kwargs: kwargs passed along to GP model
        gp_gen_kwargs: kwargs passed along to gen call on GP
    """

    def __init__(
        self,
        D: int,
        d: int,
        init_size: int,
        name: str = "ALEBO",
        dtype: torch.dtype = torch.double,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        random_kwargs: Optional[Dict[str, Any]] = None,
        gp_kwargs: Optional[Dict[str, Any]] = None,
        gp_gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.D = D
        self.d = d
        self.init_size = init_size
        self.dtype = dtype
        self.device = device
        self.random_kwargs = random_kwargs if random_kwargs is not None else {}
        self.gp_kwargs = gp_kwargs if gp_kwargs is not None else {}
        self.gp_gen_kwargs = gp_gen_kwargs

        B = self.gen_projection(d=d, D=D, device=device, dtype=dtype)

        self.gp_kwargs.update({"B": B})
        self.random_kwargs.update({"B": B.cpu().numpy()})

        steps = [
            GenerationStep(
                model=get_ALEBOInitializer,
                num_trials=init_size,
                model_kwargs=self.random_kwargs,
            ),
            GenerationStep(
                model=get_ALEBO,
                num_trials=-1,
                model_kwargs=self.gp_kwargs,
                model_gen_kwargs=gp_gen_kwargs,
            ),
        ]
        super().__init__(steps=steps, name=name)

    def clone_reset(self) -> "ALEBOStrategy":
        """Copy without state."""
        return self.__class__(
            D=self.D,
            d=self.d,
            init_size=self.init_size,
            name=self.name,
            dtype=self.dtype,
            device=self.device,
            random_kwargs=self.random_kwargs,
            gp_kwargs=self.gp_kwargs,
            gp_gen_kwargs=self.gp_gen_kwargs,
        )

    def gen_projection(
        self, d: int, D: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Generate the projection matrix B as a (d x D) tensor
        """
        B0 = torch.randn(d, D, dtype=dtype, device=device)
        B = B0 / torch.sqrt((B0 ** 2).sum(dim=0))
        return B
