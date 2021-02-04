#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
from ax.models.torch.botorch import BotorchModel
from botorch.fit import fit_gpytorch_model
from botorch.models.contextual_multioutput import LCEMGP, FixedNoiseLCEMGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7


class LCEMBO(BotorchModel):
    r"""Does Bayesian optimization with LCE-M GP."""

    def __init__(
        self,
        context_cat_feature: Optional[Tensor] = None,
        context_emb_feature: Optional[Tensor] = None,
        embs_dim_list: Optional[List[int]] = None,
    ) -> None:
        self.context_cat_feature = context_cat_feature
        self.context_emb_feature = context_emb_feature
        self.embs_dim_list = embs_dim_list
        super().__init__(model_constructor=self.get_and_fit_model)

    def get_and_fit_model(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        task_features: List[int],
        fidelity_features: List[int],
        metric_names: List[str],
        state_dict: Optional[Dict[str, Tensor]] = None,
        fidelity_model_id: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelListGP:
        """Get a fitted multi-task contextual GP model for each outcome.
        Args:
            Xs: List of X data, one tensor per outcome.
            Ys: List of Y data, one tensor per outcome.
            Yvars:List of Noise variance of Yvar data, one tensor per outcome.
            task_features: List of columns of X that are tasks.
        Returns: ModeListGP that each model is a fitted LCEM GP model.
        """

        if len(task_features) == 1:
            task_feature = task_features[0]
        elif len(task_features) > 1:
            raise NotImplementedError(
                f"LCEMBO only supports 1 task feature (got {task_features})"
            )
        else:
            raise ValueError("LCEMBO requires context input as task features")

        models = []
        for i, X in enumerate(Xs):
            # validate input Yvars
            Yvar = Yvars[i].clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
            is_nan = torch.isnan(Yvar)
            all_nan_Yvar = torch.all(is_nan)
            if all_nan_Yvar:
                gp_m = LCEMGP(
                    train_X=X,
                    train_Y=Ys[i],
                    task_feature=task_feature,
                    context_cat_feature=self.context_cat_feature,
                    context_emb_feature=self.context_emb_feature,
                    embs_dim_list=self.embs_dim_list,
                )
            else:
                gp_m = FixedNoiseLCEMGP(
                    train_X=X,
                    train_Y=Ys[i],
                    train_Yvar=Yvar,
                    task_feature=task_feature,
                    context_cat_feature=self.context_cat_feature,
                    context_emb_feature=self.context_emb_feature,
                    embs_dim_list=self.embs_dim_list,
                )
            models.append(gp_m)
        # Use a ModelListGP
        model = ModelListGP(*models)
        model.to(Xs[0])
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        return model
