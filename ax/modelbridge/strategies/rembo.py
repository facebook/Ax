#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.search_space import SearchSpace
from ax.modelbridge.factory import DEFAULT_TORCH_DEVICE
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.centered_unit_x import CenteredUnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.models.random.rembo_initializer import REMBOInitializer
from ax.models.torch.rembo import REMBO
from ax.utils.common.typeutils import not_none


def get_rembo_initializer(
    search_space: SearchSpace,
    A: np.ndarray,
    bounds_d: List[Tuple[float, float]],
    **kwargs: Any,
) -> RandomModelBridge:
    """Instantiates a uniform random generator.

    Args:
        search_space: Search space.
        A: Projection matrix.
        bounds_d: Bounds in low-d space.
        kwargs: kwargs

    Returns:
        RandomModelBridge, with REMBOInitializer as model.
    """
    return RandomModelBridge(
        search_space=search_space,
        model=REMBOInitializer(A=A, bounds_d=bounds_d, **kwargs),
        transforms=[CenteredUnitX],
    )


def get_REMBO(
    experiment: Experiment,
    data: Data,
    A: torch.Tensor,
    initial_X_d: torch.Tensor,
    bounds_d: List[Tuple[float, float]],
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    **model_kwargs: Any,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if search_space is None:
        search_space = experiment.search_space
    if data.df.empty:  # pragma: no cover
        raise ValueError("REMBO model requires non-empty data.")
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space,
        data=data,
        model=REMBO(A=A, initial_X_d=initial_X_d, bounds_d=bounds_d, **model_kwargs),
        transforms=[CenteredUnitX, StandardizeY],
        torch_dtype=dtype,
        torch_device=device,
    )


class REMBOStrategy(GenerationStrategy):
    """Generation strategy for REMBO.

    Both quasirandom initialization and BO are done with the same random
    projection. As is done in the REMBO paper, k independent optimizations
    are done, each with an independently generated projection.

    Args:
        D: Dimensionality of high-dimensional space
        d: Dimensionality of low-dimensional space
        k: Number of random projections
        init_per_proj: Number of arms to use for random initialization of each
            of the k projections.
        name: Name of strategy
        dtype: torch dtype
        device: torch device
        gp_kwargs: kwargs sent along to the GP model
    """

    def __init__(
        self,
        D: int,
        d: int,
        init_per_proj: int,
        k: int = 4,
        name: str = "REMBO",
        dtype: torch.dtype = torch.double,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        gp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.D = D
        self.d = d
        self.k = k
        self.init_per_proj = init_per_proj
        self.dtype = dtype
        self.device = device
        self.gp_kwargs = gp_kwargs if gp_kwargs is not None else {}

        self.projections = {
            i: self.get_projection(
                D=self.D, d=self.d, dtype=self.dtype, device=self.device
            )
            for i in range(self.k)
        }

        self.X_d_by_proj = defaultdict(list)
        self.current_iteration = 0
        self.arms_by_proj: Dict[int, Set[str]] = {i: set({}) for i in range(self.k)}

        # The first GenerationStep, and super
        A, bounds_d = self.projections[0]
        steps = [
            GenerationStep(
                model=get_rembo_initializer,
                num_trials=1,
                model_kwargs={"A": A, "bounds_d": bounds_d},
            )
        ]
        super().__init__(steps=steps, name=name)

    @property
    def model_transitions(self) -> List[int]:
        """Generator changes every iteration with rotating strategy"""
        return list(range(self.current_iteration))

    def gen(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Generate new points, rotating through projections each time."""
        # Use all data in experiment if none is supplied
        data = data or experiment.fetch_data()

        # Get the next model in the rotation
        i = self.current_iteration % self.k
        data_by_proj = self._filter_data_to_projection(
            experiment=experiment, data=data, arm_sigs=self.arms_by_proj[i]
        )
        lgr = self.last_generator_run
        # NOTE: May need to `model_class.deserialize_model_state` in the
        # future if using non-readily serializable state.
        model_state = (
            not_none(lgr._model_state_after_gen)
            if lgr is not None and lgr._model_state_after_gen is not None
            else {}
        )

        A, bounds_d = self.projections[i]
        if (
            data_by_proj is None
            or len(data_by_proj.df["arm_name"].unique()) < self.init_per_proj
        ):
            # Not enough data to switch to GP, use Sobol for initialization
            m = get_rembo_initializer(
                search_space=experiment.search_space,
                A=A.double().numpy(),
                bounds_d=bounds_d,
                **model_state,
            )
        else:
            # We have enough data to switch to GP.
            m = get_REMBO(
                experiment=experiment,
                data=data_by_proj,
                A=A,
                initial_X_d=torch.tensor(
                    self.X_d_by_proj[i], dtype=self.dtype, device=self.device
                ),
                bounds_d=bounds_d,
                **self.gp_kwargs,
            )

        self.current_iteration += 1
        # Call gen
        gr = m.gen(n=n)
        self.X_d_by_proj[i].extend(not_none(m.model).X_d_gen)  # pyre-ignore[16]
        self.arms_by_proj[i].update(a.signature for a in gr.arms)
        return gr

    def clone_reset(self) -> "REMBOStrategy":
        """Copy without state."""
        return self.__class__(
            D=self.D,
            d=self.d,
            k=self.k,
            init_per_proj=self.init_per_proj,
            name=self.name,
            dtype=self.dtype,
            device=self.device,
            gp_kwargs=self.gp_kwargs,
        )

    def _filter_data_to_projection(
        self, experiment: Experiment, data: Data, arm_sigs: Set[str]
    ) -> Optional[Data]:
        """Extract the arms in data that are in arm_sigs.

        Return None if none.
        """
        arm_names: Set[str] = set({})
        for arm_name in data.df["arm_name"].unique():
            sig = experiment.arms_by_name[arm_name].signature
            if sig in arm_sigs:
                arm_names.add(arm_name)

        if len(arm_names) == 0:
            return None
        # Else,
        df_i = data.df[data.df["arm_name"].isin(arm_names)].copy()
        return Data(df_i)

    def get_projection(
        self, D: int, d: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Generate the projection matrix A as a (D x d) tensor

        Also return the box bounds for the low-d space.
        """
        A = torch.randn((D, d), dtype=dtype, device=device)
        bounds_d = [(-(math.sqrt(d)), math.sqrt(d))] * d
        return A, bounds_d


class HeSBOStrategy(REMBOStrategy):
    """Generation strategy for HeSBO.

    Args:
        D: Dimensionality of high-dimensional space
        d: Dimensionality of low-dimensional space
        k: Number of random projections
        init_per_proj: Number of arms to use for random initialization of each
            of the k projections.
        name: Name of strategy
        dtype: torch dtype
        device: torch device
        gp_kwargs: kwargs sent along to the GP model
    """

    def __init__(
        self,
        D: int,
        d: int,
        init_per_proj: int,
        k: int = 1,
        name: str = "HeSBO",
        dtype: torch.dtype = torch.double,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        gp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            D=D,
            d=d,
            init_per_proj=init_per_proj,
            k=k,
            name=name,
            dtype=dtype,
            device=device,
            gp_kwargs=gp_kwargs,
        )

    def get_projection(
        self, D: int, d: int, dtype: torch.dtype, device: torch.device
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Generate the projection matrix A as a (D x d) tensor

        Also return the box bounds for the low-d space.
        """
        A = torch.zeros((D, d), dtype=dtype, device=device)
        h = torch.randint(d, size=(D,))
        s = 2 * torch.randint(2, size=(D,)) - 1
        for i in range(D):
            A[i, h[i]] = s[i]

        bounds_d = [(-1.0, 1.0)] * d
        return A, bounds_d
