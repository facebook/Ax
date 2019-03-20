#!/usr/bin/env python3

import torch
from ax.exceptions.model import ModelError
from botorch.models import Model, MultiOutputGP
from botorch.models.constant_noise import ConstantNoiseGP
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP, SingleTaskGP
from torch import Tensor


NOISELESS_MODELS = {SingleTaskGP}
MIN_INFERRED_NOISE_LEVEL = 1e-5
MIN_OBSERVED_NOISE_LEVEL = 1e-7


def is_noiseless(model: Model) -> bool:
    """Check if a given (single-task) botorch model is noiseless"""
    if isinstance(model, MultiOutputGP):  # pyre-ignore: [16]
        raise ModelError(
            "Checking for noisless models only applies to sub-models of MultiOutputGP"
        )
    return model.__class__ in NOISELESS_MODELS


def _get_model(X: Tensor, Y: Tensor, Yvar: Tensor) -> Model:
    """Instantiate a model of type depending on the input data"""
    Yvar = Yvar.view(-1)  # last dimension is not needed for botorch
    # Determine if we want to treat the noise as constant
    mean_var = Yvar.mean().clamp_min_(MIN_OBSERVED_NOISE_LEVEL)  # pyre-ignore [16]
    # Look at relative variance in noise level
    if Yvar.nelement() > 1:  # pyre-ignore [16]
        Yvar_std = Yvar.std()  # pyre-ignore [16]
    else:
        Yvar_std = torch.tensor(0).to(
            device=Yvar.device, dtype=Yvar.dtype  # pyre-ignore [16]
        )
    if Yvar_std / mean_var < 0.1:
        model = ConstantNoiseGP(
            train_X=X, train_Y=Y.view(-1), train_Y_se=mean_var.sqrt()
        )
    else:
        Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)  # pyre-ignore [16]
        model = HeteroskedasticSingleTaskGP(
            train_X=X, train_Y=Y.view(-1), train_Y_se=Yvar.view(-1).sqrt()
        )
    model.to(dtype=X.dtype, device=X.device)  # pyre-ignore: [28]
    return model
