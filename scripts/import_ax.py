#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ax
from ax.service.ax_client import AxClient


if __name__ == "__main__":
    assert ax is not None
    assert AxClient is not None
