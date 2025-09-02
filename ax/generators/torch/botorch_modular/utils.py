#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from logging import Logger
from typing import Any, cast, Mapping

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, UnsupportedError, UserInputError
from ax.generators.torch_base import TorchOptConfig
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import (
    qLogNoisyExpectedImprovement,
    qLogProbabilityOfFeasibility,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.fully_bayesian import (
    AbstractFullyBayesianSingleTaskGP,
    FullyBayesianSingleTaskGP,
)
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from botorch.utils.dispatcher import Dispatcher
from botorch.utils.types import _DefaultType, DEFAULT
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor


MIN_OBSERVED_NOISE_LEVEL = 1e-7
logger: Logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the BoTorch Model used in Surrogate.

    Args:
        botorch_model_class: ``Model`` class to be used as the underlying
            BoTorch model. If None is provided a model class will be selected (either
            one for all outcomes or a ModelList with separate models for each outcome)
            will be selected automatically based off the datasets at `construct` time.
        model_options: Dictionary of options / kwargs for the BoTorch
            ``Model`` constructed during ``Surrogate.fit``.
            Note that the corresponding attribute will later be updated to include any
            additional kwargs passed into ``BoTorchGenerator.fit``.
        mll_class: ``MarginalLogLikelihood`` class to use for model-fitting.
        mll_options: Dictionary of options / kwargs for the MLL.
        outcome_transform_classes: List of BoTorch outcome transforms classes. Passed
            down to the BoTorch ``Model``. Multiple outcome transforms can be chained
            together using ``ChainedOutcomeTransform``.
        outcome_transform_options: Outcome transform classes kwargs. The keys are
            class string names and the values are dictionaries of outcome transform
            kwargs. For example,
            `
            outcome_transform_classes = [Standardize]
            outcome_transform_options = {
                "Standardize": {"m": 1},
            `
            For more options see `botorch/models/transforms/outcome.py`.
        input_transform_classes: List of BoTorch input transforms classes.
            Passed down to the BoTorch ``Model``. Multiple input transforms
            will be chained together using ``ChainedInputTransform``.
            If `DEFAULT`, a default set of input transforms may be constructed
            based on the search space digest (in `_construct_default_input_transforms`).
            To disable this behavior, pass in `input_transform_classes=None`.
        input_transform_options: Input transform classes kwargs. The keys are
            class string names and the values are dictionaries of input transform
            kwargs. For example,
            `
            input_transform_classes = [Normalize, Round]
            input_transform_options = {
                "Normalize": {"d": 3},
                "Round": {"integer_indices": [0], "categorical_features": {1: 2}},
            }
            `
            For more input options see `botorch/models/transforms/input.py`.
        covar_module_class: Covariance module class. This gets initialized after
            parsing the ``covar_module_options`` in ``covar_module_argparse``,
            and gets passed to the model constructor as ``covar_module``.
        covar_module_options: Covariance module kwargs.
        likelihood_class: ``Likelihood`` class. This gets initialized with
            ``likelihood_options`` and gets passed to the model constructor.
        likelihood_options: Likelihood options.
        name: Name of the model config. This is used to identify the model config.
    """

    botorch_model_class: type[Model] | None = None
    model_options: dict[str, Any] = field(default_factory=dict)
    mll_class: type[MarginalLogLikelihood] | None = None
    mll_options: dict[str, Any] = field(default_factory=dict)
    input_transform_classes: list[type[InputTransform]] | _DefaultType | None = DEFAULT
    input_transform_options: dict[str, dict[str, Any]] | None = field(
        default_factory=dict
    )
    outcome_transform_classes: list[type[OutcomeTransform]] | None = None
    outcome_transform_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    covar_module_class: type[Kernel] | None = None
    covar_module_options: dict[str, Any] = field(default_factory=dict)
    likelihood_class: type[Likelihood] | None = None
    likelihood_options: dict[str, Any] = field(default_factory=dict)
    name: str | None = None

    @cached_property
    def identifier(self) -> str:
        """Returns a unique identifier for the model config."""
        return self.name if self.name is not None else str(self)


def use_model_list(
    datasets: Sequence[SupervisedDataset],
    search_space_digest: SearchSpaceDigest,
    model_configs: list[ModelConfig] | None = None,
    metric_to_model_configs: dict[str, list[ModelConfig]] | None = None,
    allow_batched_models: bool = True,
) -> bool:
    model_configs = model_configs or []
    metric_to_model_configs = metric_to_model_configs or {}
    if len(datasets) == 1 and datasets[0].Y.shape[-1] == 1:
        # There is only one outcome, so we can use a single model.
        return False
    elif (
        len(model_configs) > 1
        or len(metric_to_model_configs) > 0
        or any(len(model_config) for model_config in metric_to_model_configs.values())
    ):
        # There are multiple outcomes and outcomes might be modeled with different
        # models
        return True
    if len({type(d) for d in datasets}) > 1:
        # Use a `ModelList` if there are multiple dataset classes.
        return True

    botorch_model_class_set = {mc.botorch_model_class for mc in model_configs}
    # if any of the botorch_model_class is unspecified, we'd need to infer its class
    if (not botorch_model_class_set) or (None in botorch_model_class_set):
        inferred_botorch_model_class_set = {
            choose_model_class(dataset=dataset, search_space_digest=search_space_digest)
            for dataset in datasets
        }
        botorch_model_class_set = botorch_model_class_set.union(
            inferred_botorch_model_class_set
        )
        # Safe even if None is not present
        botorch_model_class_set.discard(None)

    if len(botorch_model_class_set) > 1:
        # Use a `ModelList` if there are multiple possible botorch_model_class classes.
        return True

    # Otherwise, the same model class is used for all outcomes.
    botorch_model_class = none_throws(next(iter(botorch_model_class_set)))
    if issubclass(botorch_model_class, FullyBayesianSingleTaskGP):
        # SAAS models do not support multiple outcomes.
        # Use model list if there are multiple outcomes.
        return len(datasets) > 1 or datasets[0].Y.shape[-1] > 1
    elif issubclass(botorch_model_class, MultiTaskGP):
        # We wrap multi-task models into `ModelListGP` when there are
        # multiple outcomes.
        return len(datasets) > 1 or datasets[0].Y.shape[-1] > 1
    elif len(datasets) == 1:
        # This method is called before multiple datasets are merged into
        # one if using a batched model. If there is one dataset here,
        # there should be a reason that a single model should be used:
        # e.g. a contextual model, where we want to jointly model the metric
        # each context (and context-level metrics are different outcomes).
        return False
    elif issubclass(botorch_model_class, BatchedMultiOutputGPyTorchModel) and all(
        torch.equal(datasets[0].X, ds.X) for ds in datasets[1:]
    ):
        # Use batch models if allowed
        return not allow_batched_models
    # If there are multiple Xs and they are not all equal, we use `ModelListGP`.
    return True


def copy_model_config_with_default_values(
    model_config: ModelConfig,
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
) -> ModelConfig:
    model_config_copy = deepcopy(model_config)

    if model_config_copy.botorch_model_class is None:
        model_config_copy.botorch_model_class = choose_model_class(
            dataset=dataset, search_space_digest=search_space_digest
        )

    if model_config_copy.mll_class is None:
        model_config_copy.mll_class = (
            PairwiseLaplaceMarginalLogLikelihood
            if model_config_copy.botorch_model_class is PairwiseGP
            else ExactMarginalLogLikelihood
        )

    # PairwiseGP does not use outcome transforms
    if model_config_copy.outcome_transform_classes is not None:
        if model_config_copy.botorch_model_class is PairwiseGP:
            model_config_copy.outcome_transform_classes = None

    return model_config_copy


def choose_model_class(
    dataset: SupervisedDataset,
    search_space_digest: SearchSpaceDigest,
) -> type[Model]:
    """Chooses a BoTorch `Model` class and `MarginalLogLikelihood` class
    using the given the dataset and search_space_digest.

    Args:
        dataset: The dataset on which the model will be fitted.
        search_space_digest: The digest of the search space the model will be
            fitted within.

    Returns:
        A BoTorch `Model` class.
    """
    if len(search_space_digest.fidelity_features) > 1:
        raise NotImplementedError(
            "Only a single fidelity feature supported "
            f"(got: {search_space_digest.fidelity_features})."
        )
    if len(search_space_digest.task_features) > 1:
        raise NotImplementedError(
            f"Only a single task feature supported "
            f"(got: {search_space_digest.task_features})."
        )
    if search_space_digest.task_features and search_space_digest.fidelity_features:
        raise NotImplementedError(
            "Multi-task multi-fidelity optimization not yet supported."
        )

    # Preference learning case
    if isinstance(dataset, RankingDataset):
        model_class = PairwiseGP

    # Multi-task case (when `task_features` is specified).
    elif search_space_digest.task_features:
        model_class = MultiTaskGP

    # Single-task multi-fidelity cases.
    elif search_space_digest.fidelity_features:
        model_class = SingleTaskMultiFidelityGP

    # Mixed optimization case. Note that presence of categorical
    # features in search space digest indicates that downstream in the
    # stack we chose not to perform continuous relaxation on those
    # features.
    elif search_space_digest.categorical_features:
        model_class = MixedSingleTaskGP

    # Single-task single-fidelity cases.
    else:
        model_class = SingleTaskGP

    logger.debug(f"Chose BoTorch model class: {model_class}.")
    return model_class


def choose_botorch_acqf_class(
    torch_opt_config: TorchOptConfig,
    datasets: Sequence[SupervisedDataset] | None,
    use_p_feasible: bool = True,
) -> type[AcquisitionFunction]:
    """Chooses the most suitable BoTorch `AcquisitionFunction` class.

    Args:
        torch_opt_config: The torch optimization config.
        datasets: The datasets that were used to fit the model.
        use_p_feasible: Whether we dispatch to `qLogProbabilityOfFeasibility` when
            there are no feasible points in the training data.

    Returns:
        A BoTorch `AcquisitionFunction` class. The current logic chooses between:
            - `qLogProbabilityOfFeasibility` if there are outcome constraints and
                no feasible point has been found.
            - `qLogNoisyExpectedImprovement` for single-objective optimization.
            - `qLogNoisyExpectedHypervolumeImprovement`` for multi-objective
                optimization.
    """

    # Check if the training data is feasible.
    if (
        use_p_feasible
        and torch_opt_config.outcome_constraints is not None
        and datasets is not None
    ):
        con_tfs = (
            get_outcome_constraint_transforms(torch_opt_config.outcome_constraints)
            or []
        )
        # NOTE: `convert_to_block_design` will drop points that are only observed by
        # some of the metrics which is natural as we are using observed values to
        # determine feasibility.
        dataset = convert_to_block_design(datasets=datasets, force=True)[0]
        con_observed = torch.stack([con(dataset.Y) for con in con_tfs], dim=-1)
        feas_point_found = (con_observed <= 0).all(dim=-1).any().item()

        if not feas_point_found:
            acqf_class = qLogProbabilityOfFeasibility
            logger.debug(f"Chose BoTorch acquisition function class: {acqf_class}.")
            return acqf_class

    if torch_opt_config.is_moo and not torch_opt_config.use_learned_objective:
        acqf_class = qLogNoisyExpectedHypervolumeImprovement
    else:
        acqf_class = qLogNoisyExpectedImprovement

    logger.debug(f"Chose BoTorch acquisition function class: {acqf_class}.")
    return acqf_class


def construct_acquisition_and_optimizer_options(
    acqf_options: TConfig,
    botorch_acqf_options: TConfig,
    model_gen_options: TConfig | None = None,
    botorch_acqf_classes_with_options: list[tuple[type[AcquisitionFunction], TConfig]]
    | None = None,
) -> tuple[
    TConfig, TConfig, TConfig, list[tuple[type[AcquisitionFunction], TConfig]] | None
]:
    """Extract acquisition and optimizer options from `model_gen_options`."""
    acq_options = acqf_options.copy()
    botorch_acqf_options = botorch_acqf_options.copy()
    opt_options = {}

    if model_gen_options:
        # Define the allowed paths

        if (
            len(
                extra_keys_in_model_gen_options := set(model_gen_options.keys())
                - {
                    Keys.OPTIMIZER_KWARGS.value,
                    Keys.ACQF_KWARGS.value,
                    Keys.AX_ACQUISITION_KWARGS.value,
                }
            )
            > 0
        ):
            raise ValueError(
                "Found forbidden keys in `model_gen_options`: "
                f"{extra_keys_in_model_gen_options}."
            )
        new_botorch_acqf_options = assert_is_instance(
            model_gen_options.get(Keys.ACQF_KWARGS, {}),
            dict,
        )
        if (
            # pyre-fixme [6]: Incompatible parameter type [6]: In call `len`,
            # for 1st positional argument, expected
            # `pyre_extensions.PyreReadOnly[Sized]` but got `dict`.
            len(new_botorch_acqf_options) > 0
            and botorch_acqf_classes_with_options is not None
        ):
            if len(botorch_acqf_classes_with_options) > 1:
                warnings.warn(
                    message="botorch_acqf_options are being ignored, due to using "
                    "MultiAcquisition. Specify options for each acquistion function"
                    "via botorch_acqf_classes_with_options.",
                    category=AxWarning,
                    stacklevel=4,
                )
            else:
                botorch_acqf_classes_with_options = deepcopy(
                    botorch_acqf_classes_with_options
                )
                botorch_acqf_classes_with_options[0][1].update(new_botorch_acqf_options)
        else:
            botorch_acqf_options.update(new_botorch_acqf_options)

        acq_options.update(
            assert_is_instance(
                model_gen_options.get(Keys.AX_ACQUISITION_KWARGS, {}),
                dict,
            )
        )

        # TODO: Add this if all acq. functions accept the `subset_model`
        # kwarg or opt for kwarg filtering.
        # acq_options[SUBSET_MODEL] = model_gen_options.get(SUBSET_MODEL)
        opt_options = assert_is_instance(
            model_gen_options.get(Keys.OPTIMIZER_KWARGS, {}),
            dict,
        ).copy()
    return (
        acq_options,
        botorch_acqf_options,
        opt_options,
        botorch_acqf_classes_with_options,
    )


def convert_to_block_design(
    datasets: Sequence[SupervisedDataset],
    force: bool = False,
) -> list[SupervisedDataset]:
    """Converts a list of datasets to a single block-design dataset that contains
    all outcomes.

    Args:
        datasets: A list of datasets to merge.
        force: If True, will force conversion of data not complying to a block
            design to block design by dropping observations that are not shared
            between outcomes.
            If only a subset of the outcomes have noise observations, all noise
            observations will be dropped.

    Returns:
        A single element list containing the merged dataset.
    """
    is_fixed = [ds.Yvar is not None for ds in datasets]
    if any(is_fixed) and not all(is_fixed):
        if force:
            logger.debug(
                "Only a subset of datasets have noise observations. "
                "Dropping all noise observations since `force=True`. "
            )
        else:
            raise UnsupportedError(
                "Cannot convert mixed data with and without variance "
                "observations to `block design`."
            )
    is_fixed = all(is_fixed)
    Xs = [dataset.X for dataset in datasets]
    for dset in datasets[1:]:
        if dset.feature_names != datasets[0].feature_names:
            raise ValueError(
                "Feature names must be the same across all datasets, "
                f"got {dset.feature_names} and {datasets[0].feature_names}"
            )

    # Join the outcome names of datasets.
    outcome_names = sum([ds.outcome_names for ds in datasets], [])

    if len({X.shape for X in Xs}) != 1 or not all(
        torch.equal(X, Xs[0]) for X in Xs[1:]
    ):
        if not force:
            raise UnsupportedError(
                "Cannot convert data to non-block design data. "
                "To force this and drop data not shared between "
                "outcomes use `force=True`."
            )
        logger.debug(
            "Forcing conversion of data not complying to a block design "
            "to block design by dropping observations that are not shared "
            "between outcomes."
        )
        X_shared, idcs_shared = _get_shared_rows(Xs=Xs)
        Y = torch.cat([ds.Y[i] for ds, i in zip(datasets, idcs_shared)], dim=-1)
        if is_fixed:
            Yvar = torch.cat(
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                [ds.Yvar[i] for ds, i in zip(datasets, idcs_shared)],
                dim=-1,
            )
        else:
            Yvar = None
        datasets = [
            SupervisedDataset(
                X=X_shared,
                Y=Y,
                Yvar=Yvar,
                feature_names=datasets[0].feature_names,
                outcome_names=outcome_names,
            )
        ]
        return datasets

    # data complies to block design, can concat with impunity
    Y = torch.cat([ds.Y for ds in datasets], dim=-1)
    if is_fixed:
        Yvar = torch.cat([none_throws(ds.Yvar) for ds in datasets], dim=-1)
    else:
        Yvar = None
    datasets = [
        SupervisedDataset(
            X=Xs[0],
            Y=Y,
            Yvar=Yvar,
            feature_names=datasets[0].feature_names,
            outcome_names=outcome_names,
        )
    ]
    return datasets


def _get_shared_rows(Xs: list[Tensor]) -> tuple[Tensor, list[Tensor]]:
    """Extract shared rows from a list of tensors

    Args:
        Xs: A list of m two-dimensional tensors with shapes
            `(n_1 x d), ..., (n_m x d)`. It is not required that
            the `n_i` are the same.

    Returns:
        A two-tuple containing (i) a Tensor with the rows that are
        shared between all the Tensors in `Xs`, and (ii) a list of
        index tensors that indicate the location of these rows
        in the respective elements of `Xs`.
    """
    if any(X.ndim != 2 for X in Xs):
        raise UserInputError("All inputs must be two-dimensional.")
    idcs_shared = []
    Xs_sorted = sorted(Xs, key=len)
    X_shared = Xs_sorted[0].clone().unique(dim=0)
    for X in Xs_sorted[1:]:
        X_shared = X_shared[(X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-2)]
    # get indices
    for X in Xs:
        _, inverse_indices = X.unique(dim=0, return_inverse=True)
        # set of unique original indices
        unique_indices = set(
            {v: i for i, v in enumerate(inverse_indices.tolist())}.values()
        )
        same = (X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-1)
        shared_idcs_list = []
        for i, is_shared in enumerate(same):
            if is_shared and i in unique_indices:
                shared_idcs_list.append(i)
        shared_idcs = torch.tensor(
            shared_idcs_list, dtype=torch.long, device=X_shared.device
        )
        idcs_shared.append(shared_idcs)
    return X_shared, idcs_shared


def subset_state_dict(
    state_dict: Mapping[str, Tensor],
    submodel_index: int,
) -> OrderedDict[str, Tensor]:
    """Get the state dict for a submodel from the state dict of a model list.

    Args:
        state_dict: A state dict.
        submodel_index: The index of the submodel to extract.

    Returns:
        The state dict for the submodel.
    """
    expected_substring = f"models.{submodel_index}."
    len_substring = len(expected_substring)
    new_items = [
        (k[len_substring:], v)
        for k, v in state_dict.items()
        if k.startswith(expected_substring)
    ]
    return OrderedDict(new_items)


# ----------------------- Model fitting helpers ----------------------- #


fit_botorch_model = Dispatcher(name="fit_botorch_model", encoder=_argparse_type_encoder)


@fit_botorch_model.register(ModelList)
def _fit_botorch_model_list(
    model: Model,
    mll_class: type[MarginalLogLikelihood],
    mll_options: dict[str, Any] | None = None,
) -> None:
    for m in cast(list[Model], model.models):
        fit_botorch_model(m, mll_class=mll_class, mll_options=mll_options)


@fit_botorch_model.register(GPyTorchModel)
@fit_botorch_model.register(PairwiseGP)
def _fit_botorch_model_gpytorch(
    model: GPyTorchModel | PairwiseGP,
    mll_class: type[MarginalLogLikelihood],
    mll_options: dict[str, Any] | None = None,
) -> None:
    """Fit a GPyTorch based BoTorch model."""
    mll_options = mll_options or {}
    mll = mll_class(likelihood=model.likelihood, model=model, **mll_options)
    fit_gpytorch_mll(mll)


@fit_botorch_model.register(
    (AbstractFullyBayesianSingleTaskGP, SaasFullyBayesianMultiTaskGP)
)
def _fit_botorch_model_fully_bayesian_nuts(
    model: AbstractFullyBayesianSingleTaskGP | SaasFullyBayesianMultiTaskGP,
    mll_class: type[MarginalLogLikelihood],
    mll_options: dict[str, Any] | None = None,
) -> None:
    mll_options = mll_options or {}
    mll_options.setdefault("disable_progbar", True)
    fit_fully_bayesian_model_nuts(model, **mll_options)


@fit_botorch_model.register(object)
def _fit_botorch_model_not_implemented(
    model: Model,
    mll_class: type[MarginalLogLikelihood],
    mll_options: dict[str, Any] | None = None,
) -> None:
    raise NotImplementedError(
        f"fit_botorch_model is not implemented for {model.__class__.__name__}. "
        "You can register a model fitting routine for it by adding new case "
        "to the `fit_botorch_model` dispatcher. To do so, decorate a function "
        "that accepts `model`, `mll_class` and `mll_options` inputs with "
        f"`@fit_botorch_model.register({model.__class__.__name__})`."
    )


@dataclass(frozen=True)
class CVFold:
    """
    Args:
        train_dataset: The training dataset for the fold.
        test_X: The test inputs for the fold.
        test_Y: The test outputs for the fold.
    """

    train_dataset: SupervisedDataset
    test_X: Tensor
    test_Y: Tensor


def get_cv_fold(
    dataset: SupervisedDataset, X: Tensor, Y: Tensor, idcs: Tensor
) -> CVFold:
    train_mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    train_mask[idcs] = 0
    return CVFold(
        train_dataset=dataset.clone(mask=train_mask),
        test_X=X[idcs],
        test_Y=Y[idcs],
    )


def get_all_task_values_from_ssd(search_space_digest: SearchSpaceDigest) -> list[int]:
    """Get all task values from a search space digest.

    Args:
        search_space_digest: The search space digest.

    Returns:
        A list of all task values.
    """
    task_feature = search_space_digest.task_features[0]
    task_bounds = search_space_digest.bounds[task_feature]
    return list(range(int(task_bounds[0]), int(task_bounds[1] + 1)))
