#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

import torch
from torch._C import dtype


def get_optimizer_kwargs() -> Dict[str, int]:
    return {"num_restarts": 2, "raw_samples": 2, "maxiter": 2, "batch_limit": 1}


# pyre-fixme[3]: Return type must be annotated.
def get_torch_test_data(
    dtype: dtype=torch.float,
    cuda: bool = False,
    constant_noise: bool = True,
    # pyre-fixme[2]: Parameter must be annotated.
    task_features=None,
    offset: float = 0.0,
):
    tkwargs = {"device": torch.device("cuda" if cuda else "cpu"), "dtype": dtype}
    Xs = [
        torch.tensor(
            [
                [1.0 + offset, 2.0 + offset, 3.0 + offset],
                [2.0 + offset, 3.0 + offset, 4.0 + offset],
            ],
            # pyre-fixme[6]: For 2nd param expected `Optional[dtype]` but got
            #  `Union[dtype, device]`.
            # pyre-fixme[6]: For 2nd param expected `Union[None, str, device]` but
            #  got `Union[dtype, device]`.
            # pyre-fixme[6]: For 2nd param expected `bool` but got `Union[dtype,
            #  device]`.
            **tkwargs,
        )
    ]
    # pyre-fixme[6]: For 2nd param expected `Optional[dtype]` but got `Union[dtype,
    #  device]`.
    # pyre-fixme[6]: For 2nd param expected `Union[None, str, device]` but got
    #  `Union[dtype, device]`.
    # pyre-fixme[6]: For 2nd param expected `bool` but got `Union[dtype, device]`.
    Ys = [torch.tensor([[3.0 + offset], [4.0 + offset]], **tkwargs)]
    # pyre-fixme[6]: For 2nd param expected `Optional[dtype]` but got `Union[dtype,
    #  device]`.
    # pyre-fixme[6]: For 2nd param expected `Union[None, str, device]` but got
    #  `Union[dtype, device]`.
    # pyre-fixme[6]: For 2nd param expected `bool` but got `Union[dtype, device]`.
    Yvars = [torch.tensor([[0.0 + offset], [2.0 + offset]], **tkwargs)]
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
