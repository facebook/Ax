#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from contextlib import contextmanager
from typing import Generator, Optional

import numpy as np
import torch


def set_rng_seed(seed: int) -> None:
    """Sets seeds for random number generators from numpy, pytorch,
    and the native random module.

    Args:
        seed: The random number generator seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@contextmanager
def with_rng_seed(seed: Optional[int]) -> Generator[None, None, None]:
    """Context manager that sets the random number generator seeds
    to a given value and restores the previous state on exit.

    Args:
        seed: The random number generator seed. If None, this
            does not set the random seed, but still restores
            the random seed upon exit.
    """
    old_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    try:
        if seed is not None:
            set_rng_seed(seed)
        yield
    finally:
        random.setstate(old_state["random"])
        np.random.set_state(old_state["numpy"])
        torch.set_rng_state(old_state["torch"])
