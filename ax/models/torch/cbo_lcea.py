#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata, TConfig
from ax.models.torch.alebo import ei_or_nei
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.cbo_sac import generate_model_space_decomposition
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.fit import fit_gpytorch_model
from botorch.models.contextual import LCEAGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7
logger = get_logger(__name__)


def get_map_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    decomposition: Dict[str, List[int]],
    train_embedding: bool = True,
    cat_feature_dict: Optional[Dict] = None,
    embs_feature_dict: Optional[Dict] = None,
    embs_dim_list: Optional[List[int]] = None,
    context_weight_dict: Optional[Dict] = None,
) -> Tuple[LCEAGP, ExactMarginalLogLikelihood]:
    """Obtain MAP fitting of Latent Context Embedding Additive (LCE-A) GP."""
    # assert train_X is non-batched
    assert train_X.dim() < 3, "Don't support batch training"
    model = LCEAGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
        decomposition=decomposition,
        train_embedding=train_embedding,
        embs_dim_list=embs_dim_list,
        cat_feature_dict=cat_feature_dict,
        embs_feature_dict=embs_feature_dict,
        context_weight_dict=context_weight_dict,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model, mll


class LCEABO(BotorchModel):
    r"""Does Bayesian optimization with Latent Context Embedding Additive (LCE-A) GP.
    The parameter space decomposition must be provided.

    Args:
        decomposition: Keys are context names. Values are the lists of parameter
            names belong to the context, e.g.
            {'context1': ['p1_c1', 'p2_c1'],'context2': ['p1_c2', 'p2_c2']}.
        gp_model_args: Dictionary of kwargs to pass to GP model training.
            - train_embedding: Boolen. If true, we will train context embedding;
            otherwise, we use pre-trained embeddings from embds_feature_dict only.
            Default is True.
    """

    def __init__(
        self,
        decomposition: Dict[str, List[str]],
        cat_feature_dict: Optional[Dict] = None,
        embs_feature_dict: Optional[Dict] = None,
        context_weight_dict: Optional[Dict] = None,
        embs_dim_list: Optional[List[int]] = None,
        gp_model_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        # add validation for input decomposition
        for param_list in list(decomposition.values()):
            assert len(param_list) == len(
                list(decomposition.values())[0]
            ), "Each Context should contain same number of parameters"
        self.decomposition = decomposition
        self.cat_feature_dict = cat_feature_dict
        self.embs_feature_dict = embs_feature_dict
        self.context_weight_dict = context_weight_dict
        self.embs_dim_list = embs_dim_list
        self.gp_model_args = gp_model_args if gp_model_args is not None else {}
        self.feature_names: List[str] = []
        self.train_embedding = self.gp_model_args.get("train_embedding", True)
        super().__init__(
            model_constructor=self.get_and_fit_model,
            acqf_constructor=ei_or_nei,  # pyre-ignore
        )

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        if len(search_space_digest.feature_names) == 0:
            raise ValueError("feature names are required for LCEABO")
        self.feature_names = search_space_digest.feature_names
        super().fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            search_space_digest=search_space_digest,
            metric_names=metric_names,
        )

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Optional[Tensor]:
        raise NotImplementedError

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
    ) -> GPyTorchModel:
        """Get a fitted LCEAGP model for each outcome.
        Args:
            Xs: X for each outcome.
            Ys: Y for each outcome.
            Yvars: Noise variance of Y for each outcome.
        Returns: Fitted LCEAGP model.
        """
        # generate model space decomposition dict
        decomp_index = generate_model_space_decomposition(
            decomposition=self.decomposition, feature_names=self.feature_names
        )

        models = []
        for i, X in enumerate(Xs):
            Yvar = Yvars[i].clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
            gp_m, _ = get_map_model(
                train_X=X,
                train_Y=Ys[i],
                train_Yvar=Yvar,
                decomposition=decomp_index,
                train_embedding=self.train_embedding,
                cat_feature_dict=self.cat_feature_dict,
                embs_feature_dict=self.embs_feature_dict,
                embs_dim_list=self.embs_dim_list,
                context_weight_dict=self.context_weight_dict,
            )
            models.append(gp_m)

        if len(models) == 1:
            model = models[0]
        else:
            model = ModelListGP(*models)
        model.to(Xs[0])
        return model
