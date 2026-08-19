#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from collections.abc import Generator
from contextlib import contextmanager

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
def with_rng_seed(seed: int | None) -> Generator[None, None, None]:
    """Context manager that sets the random number generator seeds
    to a given value and restores the previous state on exit.

    If the seed is None, the context manager does nothing. This makes
    it possible to use the context manager without having to change
    the code based on whether the seed is specified.

    Args:
        seed: The random number generator seed.
    """
    if seed is None:
        yield
    else:
        old_state_native = random.getstate()
        old_state_numpy = np.random.get_state()
        try:
            with torch.random.fork_rng():
                set_rng_seed(seed)
                yield
        finally:
            random.setstate(old_state_native)
            np.random.set_state(old_state_numpy)
