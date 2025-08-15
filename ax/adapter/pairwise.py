#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.adapter.torch import TorchAdapter


# PairwiseAdapter was deprecated in Ax 1.1.0, so it should be reaped in Ax
# 1.2.0+
class PairwiseAdapter(TorchAdapter):
    # pyre-ignore[2]: Missing parameter annotations.
    def __init__(self, **kwargs) -> None:
        raise DeprecationWarning(
            "PairwiseAdapter is deprecated. Use TorchAdapter instead."
        )
