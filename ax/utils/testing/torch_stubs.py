#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

import torch


def get_optimizer_kwargs() -> Dict[str, int]:
    return {"num_restarts": 2, "raw_samples": 2, "maxiter": 2, "batch_limit": 1}


def get_torch_test_data(
    dtype=torch.float, cuda=False, constant_noise=True, task_features=None, offset=0.0
):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    Xs = [
        torch.tensor(
            [
                [1.0 + offset, 2.0 + offset, 3.0 + offset],
                [2.0 + offset, 3.0 + offset, 4.0 + offset],
            ],
            dtype=dtype,
            device=device,
        )
    ]
    Ys = [torch.tensor([[3.0 + offset], [4.0 + offset]], dtype=dtype, device=device)]
    Yvars = [torch.tensor([[0.0 + offset], [2.0 + offset]], dtype=dtype, device=device)]
    if constant_noise:
        Yvars[0].fill_(1.0)
    bounds = [
        (0.0 + offset, 1.0 + offset),
        (1.0 + offset, 4.0 + offset),
        (2.0 + offset, 5.0 + offset),
    ]
    feature_names = ["x1", "x2", "x3"]
    task_features = [] if task_features is None else task_features
    metric_names = ["y", "r"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names, metric_names
