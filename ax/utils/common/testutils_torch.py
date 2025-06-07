#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Defines a mixin that can be used along with `TestCase` to test PyTorch-related
functionality.
"""

from typing import Any

import torch


class AxTorchTestCaseMixin:
    """Mixin that can be used along with `TestCase` to test PyTorch-related
    functionality.

    Example:
        >>> from ax.utils.common.testutils_torch import AxTorchTestCaseMixin
        >>> from ax.utils.common.testutils import TestCase
        >>> class MyTest(TestCase, AxTorchTestCaseMixin):
        ...     def test_my_test(self) -> None:
        ...         self.assertAllClose(torch.tensor(1.0), torch.tensor(1.0))
    """

    # Copied from BoTorch assertAllClose
    def assertAllClose(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        input: Any,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        other: Any,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> None:
        r"""Assert that two tensors are close.

        Calls torch.testing.assert_close, using the signature and default behavior
        of torch.allclose.

        The formula asserted is abs(input - other) <= atol + rtol * abs(other).

        Args:
            input: First tensor or tensor-or-scalar-like to compare
            other: Second tensor or tensor-or-scalar-like to compare
            rtol: Relative tolerance
            atol: Absolute tolerance
            equal_nan: If True, consider NaN values as equal

        Example output:
            AssertionError: Scalars are not close!

            Absolute difference: 1.0000034868717194 (up to 0.0001 allowed)
            Relative difference: 0.8348668001940709 (up to 1e-05 allowed)
        """
        torch.testing.assert_close(
            input,
            other,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
