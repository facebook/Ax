# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, contextmanager
from typing import Tuple, Generator
from unittest import mock

import torch
from ax.modelbridge.factory import DEFAULT_TORCH_DEVICE
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from torch import Tensor


class ModelsMockingError(Exception):
    pass


@contextmanager
def mock_mbo() -> Generator[None, None, None]:
    """A context manager for mocking out the most computationally expensive BoTorch
    functions when using Modular BoTorch models in Ax.
    """

    def dummy_optimize_acqf(q: int, **kwargs) -> Tuple[Tensor, Tensor]:
        dtype = torch.double
        device = DEFAULT_TORCH_DEVICE

        return (
            torch.rand(q, 2, dtype=dtype, device=device),
            torch.tensor(0.0, dtype=dtype, device=device),
        )

    with ExitStack() as es:
        mock_optimize_acqf = es.enter_context(
            mock.patch(
                f"{Acquisition.__module__}.optimize_acqf",
                wraps=dummy_optimize_acqf,
            )
        )

        mock_fit_gpytorch_model = es.enter_context(
            mock.patch(f"{Surrogate.__module__}.fit_gpytorch_model", autospec=True)
        )

        yield

    if mock_optimize_acqf.call_count < 1 and mock_fit_gpytorch_model.call_count < 1:
        raise ModelsMockingError(
            "No Modular BoTorch mocks called in context. Please remove unused "
            "mocking context manager."
        )
