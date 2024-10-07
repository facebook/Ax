#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any, cast, Union

from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_defaults import get_qLogNEI
from ax.models.torch.cbo_sac import generate_model_space_decomposition
from ax.models.torch_base import TorchModel, TorchOptConfig
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from botorch.fit import fit_gpytorch_mll
from botorch.models.contextual import LCEAGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.datasets import SupervisedDataset
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7
logger: Logger = get_logger(__name__)


def get_map_model(
    train_X: Tensor,
    train_Y: Tensor,
    train_Yvar: Tensor,
    decomposition: dict[str, list[int]],
    train_embedding: bool = True,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict` to avoid runtime subscripting errors.
    cat_feature_dict: dict | None = None,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict` to avoid runtime subscripting errors.
    embs_feature_dict: dict | None = None,
    embs_dim_list: list[int] | None = None,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict` to avoid runtime subscripting errors.
    context_weight_dict: dict | None = None,
) -> tuple[LCEAGP, ExactMarginalLogLikelihood]:
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
    fit_gpytorch_mll(mll)
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
        decomposition: dict[str, list[str]],
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        cat_feature_dict: dict | None = None,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        embs_feature_dict: dict | None = None,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict` to avoid runtime subscripting errors.
        context_weight_dict: dict | None = None,
        embs_dim_list: list[int] | None = None,
        gp_model_args: dict[str, Any] | None = None,
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
        # pyre-fixme[4]: Attribute must be annotated.
        self.gp_model_args = gp_model_args if gp_model_args is not None else {}
        self.feature_names: list[str] = []
        # pyre-fixme[4]: Attribute must be annotated.
        self.train_embedding = self.gp_model_args.get("train_embedding", True)
        super().__init__(
            model_constructor=self.get_and_fit_model, acqf_constructor=get_qLogNEI
        )

    @copy_doc(TorchModel.fit)
    def fit(
        self,
        datasets: list[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
    ) -> None:
        if len(search_space_digest.feature_names) == 0:
            raise ValueError("feature names are required for LCEABO")
        self.feature_names = search_space_digest.feature_names
        super().fit(
            datasets=datasets,
            search_space_digest=search_space_digest,
        )

    @copy_doc(TorchModel.best_point)
    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Tensor | None:
        raise NotImplementedError

    def get_and_fit_model(
        self,
        Xs: list[Tensor],
        Ys: list[Tensor],
        Yvars: list[Tensor],
        task_features: list[int],
        fidelity_features: list[int],
        metric_names: list[str],
        state_dict: dict[str, Tensor] | None = None,
        fidelity_model_id: int | None = None,
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

    @property
    def model(self) -> LCEAGP | ModelListGP:
        return cast(Union[LCEAGP, ModelListGP], super().model)
