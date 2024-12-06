#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

import torch


def get_torch_test_data(
    dtype: torch.dtype = torch.float,
    cuda: bool = False,
    constant_noise: bool = True,
    task_features: list[int] | None = None,
    offset: float = 0.0,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[tuple[float, float]],
    list[int],
    list[str],
    list[str],
]:
    tkwargs: dict[str, Any] = {
        "device": torch.device("cuda" if cuda else "cpu"),
        "dtype": dtype,
    }
    Xs = [
        torch.tensor(
            [
                [1.0 + offset, 2.0 + offset, 3.0 + offset],
                [2.0 + offset, 3.0 + offset, 4.0 + offset],
            ],
            **tkwargs,
        )
    ]
    Ys = [torch.tensor([[3.0 + offset], [4.0 + offset]], **tkwargs)]
    if constant_noise:
        Yvar = torch.ones(2, 1, **tkwargs)
    else:
        Yvar = torch.tensor([[0.0 + offset], [2.0 + offset]], **tkwargs)
    Yvars = [Yvar]

    bounds = [
        (0.0 + offset, 1.0 + offset),
        (1.0 + offset, 4.0 + offset),
        (2.0 + offset, 5.0 + offset),
    ]
    feature_names = ["x1", "x2", "x3"]
    task_features = [] if task_features is None else task_features
    metric_names = ["y"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names, metric_names
