# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from logging import Logger
from typing import Any
from unittest import mock

import torch
from ax.adapter.adapter_utils import array_to_observation_data
from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import ExperimentData, extract_experiment_data
from ax.adapter.registry import (
    GENERATOR_KEY_TO_GENERATOR_SETUP,
    Generators,
    GeneratorSetup,
    MBM_X_trans,
    Y_trans,
)
from ax.adapter.torch import FIT_MODEL_ERROR, TorchAdapter
from ax.adapter.transfer_learning.utils import get_joint_search_space
from ax.adapter.transfer_learning.utils_torch import get_mapped_parameter_names
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.fixed_to_tunable import FixedToTunable
from ax.adapter.transforms.metadata_to_task import MetadataToTask
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.generation_strategy.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.generation_strategy.dispatch_utils import get_derelativize_config
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from gpytorch.kernels.kernel import Kernel
from pyre_extensions import assert_is_instance

TL_EXP: AuxiliaryExperimentPurpose = AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT
TARGET_TASK_VALUE: int = 0
logger: Logger = get_logger(__name__)


class TransferLearningAdapter(TorchAdapter):
    def __init__(
        self,
        *,
        experiment: Experiment,
        generator: BoTorchGenerator,
        search_space: SearchSpace | None = None,
        data: Data | None = None,
        transforms: Sequence[type[Transform]] | None = None,
        transform_configs: dict[str, TConfig] | None = None,
        optimization_config: OptimizationConfig | None = None,
        data_loader_config: DataLoaderConfig | None = None,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        default_model_gen_options: TConfig | None = None,
        torch_device: torch.device | None = None,
    ) -> None:
        r"""A adapter for transfer learning methods. In addition to the usual
        adapter functionality, this supports utilizing the observations
        available in auxiliary sources to improve the prediction accuracy
        in the current generator using multi-task models.

        This adapter requires using the `MetadataToTask` transform. Under the hood,
        this Adapter adds the task value to ObservationFeatures.metadata, which
        is then converted into a task parameter via `MetadataToTask`.

        Auxiliary sources must be attached to the experiment via
        ``experiment.auxiliary_experiments_by_purpose`` with type set to
        ``AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT``.
        """
        transforms = [] if transforms is None else list(transforms)
        if MetadataToTask not in transforms:
            raise UserInputError(
                "MetadataToTask must be in the list of transforms for "
                "TransferLearningAdapter."
            )
        if FixedToTunable not in transforms:
            transforms = [FixedToTunable] + transforms
        search_space = search_space or experiment.search_space
        # Extract auxiliary sources from the experiment.
        self.auxiliary_sources: list[AuxiliarySource] = []
        for aux_src in experiment.auxiliary_experiments_by_purpose.get(TL_EXP, []):
            try:
                # Remove data from `AuxiliarySource` constructor
                aux_src_dict = aux_src.__dict__.copy()
                aux_src_dict.pop("data", None)

                aux_src = AuxiliarySource(**aux_src_dict)
                self.auxiliary_sources.append(aux_src)
            except Exception as e:
                raise TypeError(
                    f"Failed to construct `AuxiliarySource` from {aux_src}, "
                    f"which is type {type(aux_src)}. Original error: {e}"
                )
        if not self.auxiliary_sources:
            raise UserInputError(
                "Provide auxiliary sources to use for transfer learning by attaching "
                "auxiliary sources to auxiliary_experiments_by_purpose argument of the "
                "experiment with type set to "
                "AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT."
            )

        # Check if we need to use heterogenous search space TL.
        filled_params: list[str] = []
        if (
            transform_configs is not None
            and "FillMissingParameters" in transform_configs
        ):
            filled_params.extend(
                transform_configs["FillMissingParameters"]  # pyre-ignore[16]
                .get("fill_values", {})
                .keys()
            )
        transform_configs = transform_configs or {}
        transform_configs["MetadataToTask"] = {
            "task_values": list(range(len(self.auxiliary_sources) + 1))
        }

        self.joint_search_space: SearchSpace = get_joint_search_space(
            search_space=search_space,
            auxiliary_sources=self.auxiliary_sources,
        )
        self._set_status_quo_backfill_values(
            target_experiment=experiment,
            target_search_space=search_space,
        )

        # Add source-only backfilled params as FixedParameter to the target
        # search space so that the compatibility check passes and the model
        # space includes these params (FixedToTunable will later convert them
        # to RangeParameter using the joint space bounds).
        search_space = search_space.clone()  # avoid mutating caller's object
        target_param_names = set(search_space.parameters.keys())
        for name, param in self.joint_search_space.parameters.items():
            if name not in target_param_names and param.backfill_value is not None:
                search_space.add_parameter(
                    FixedParameter(
                        name=name,
                        parameter_type=param.parameter_type,
                        value=param.backfill_value,
                    )
                )

        # Include backfill param names in filled_params so Phase 1 of
        # check_search_space_compatibility passes for target-only params.
        filled_params.extend(self.joint_search_space.backfill_values().keys())

        try:
            for s in self.auxiliary_sources:
                s.check_search_space_compatibility(
                    target_search_space=search_space, filled_params=filled_params
                )
            self._heterogeneous_search_space: bool = False
        except (UserInputError, ValueError):
            self._heterogeneous_search_space: bool = True

        self._task_value: int = TARGET_TASK_VALUE

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            generator=generator,
            transforms=transforms,
            transform_configs=transform_configs,
            torch_device=torch_device,
            optimization_config=optimization_config,
            data_loader_config=data_loader_config,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
            default_model_gen_options=default_model_gen_options,
        )

    def _transform_data(
        self,
        experiment_data: ExperimentData,
        search_space: SearchSpace,
        transforms: Sequence[type[Transform]] | None,
        transform_configs: Mapping[str, TConfig] | None,
        assign_transforms: bool = True,
    ) -> tuple[ExperimentData, SearchSpace]:
        """Initialize transforms and apply them to provided data.

        Differs from the base class implementation in that it uses the
        joint search space of all experiments to initialize the transforms.
        """
        # Initialize transforms
        search_space = search_space.clone()
        joint_search_space = self.joint_search_space.clone()
        if transforms is not None:
            if transform_configs is None:
                transform_configs = {}

            for t in transforms:
                try:
                    t_instance = t(
                        search_space=joint_search_space,
                        experiment_data=experiment_data,
                        adapter=self,
                        config=transform_configs.get(t.__name__, None),
                    )
                    joint_search_space = t_instance.transform_search_space(
                        search_space=joint_search_space
                    )
                    search_space = t_instance.transform_search_space(
                        search_space=search_space
                    )
                    experiment_data = t_instance.transform_experiment_data(
                        experiment_data=experiment_data,
                    )
                except DataRequiredError:
                    continue
                if assign_transforms:
                    self.transforms[t.__name__] = t_instance

        return experiment_data, search_space

    def get_training_data(self, filter_in_design: bool = False) -> ExperimentData:
        """Returns the training data for the current experiment, with its metadata
        updated to include the task value.
        """
        return self._update_metadata_with_task(
            experiment_data=super().get_training_data(filter_in_design)
        )

    def _update_metadata_with_task(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Updates the metadata of `experiment_data.arm_data` with the task value."""
        arm_data = experiment_data.arm_data
        # This is faster than using df.apply with a lambda function.
        arm_data["metadata"] = [
            {Keys.TASK_FEATURE_NAME.value: self._task_value, **md}
            for md in arm_data["metadata"]
        ]
        return experiment_data

    def _set_status_quo_backfill_values(
        self,
        target_experiment: Experiment,
        target_search_space: SearchSpace,
    ) -> None:
        """Set ``backfill_value`` on joint search space parameters from status_quo.

        For parameters that exist in some experiments but not others, uses the
        status_quo value from the experiment that contains the parameter as the
        ``backfill_value``. This allows ``FillMissingParameters`` to automatically
        fill in missing parameter values during data transformation, matching
        the behavior used in ``source_correlation._partition_params``.

        The target experiment's status_quo takes precedence over source
        experiments'. Existing ``backfill_value`` settings are never overridden.

        Args:
            target_experiment: The target experiment.
            target_search_space: The target search space.
        """
        target_param_names = set(target_search_space.parameters.keys())
        # Collect source param names (mapped to joint space names) per source.
        source_param_names_list: list[tuple[set[str], AuxiliarySource]] = []
        for aux_src in self.auxiliary_sources:
            source_ss = aux_src.experiment.search_space
            # Build the set of joint-space names for this source's params.
            # transfer_param_config maps target_name -> source_name, so the
            # reverse maps source_name -> target_name (= joint space name).
            reverse_config = {v: k for k, v in aux_src.transfer_param_config.items()}
            joint_names = {
                reverse_config.get(p_name, p_name)
                for p_name in source_ss.parameters.keys()
            }
            source_param_names_list.append((joint_names, aux_src))

        for name, param in self.joint_search_space.parameters.items():
            if param.backfill_value is not None:
                continue

            in_target = name in target_param_names
            in_all_sources = all(
                name in src_names for src_names, _ in source_param_names_list
            )
            if in_target and in_all_sources:
                # Param exists everywhere — no filling needed.
                continue

            # Try target status_quo first.
            if in_target:
                sq = target_experiment.status_quo
                if (
                    sq is not None
                    and name in sq.parameters
                    and sq.parameters[name] is not None
                ):
                    param._backfill_value = sq.parameters[name]
                    logger.info(
                        f"Set backfill_value for '{name}' from target "
                        f"status_quo: {sq.parameters[name]}"
                    )
                    continue

            # Try source status_quos.
            for src_names, aux_src in source_param_names_list:
                if name not in src_names:
                    continue
                source_name = aux_src.transfer_param_config.get(name, name)
                sq = aux_src.experiment.status_quo
                if (
                    sq is not None
                    and source_name in sq.parameters
                    and sq.parameters[source_name] is not None
                ):
                    param._backfill_value = sq.parameters[source_name]
                    logger.info(
                        f"Set backfill_value for '{name}' from source "
                        f"'{aux_src.experiment.name}' status_quo: "
                        f"{sq.parameters[source_name]}"
                    )
                    break

    def get_transferable_datasets(
        self,
        parameters: list[str],
        target_outcome: str,
    ) -> dict[str, list[SupervisedDataset]]:
        """Returns a list of datasets for each auxiliary source.

        Args:
            parameters: List of parameters, used for ordering columns of X.
            target_outcome: Outcome the transfer dataset is for.

        Returns: A dict keyed by auxiliary source experiment name with values the
            datasets for that transferable experiment.
        """
        transferable_datasets: dict[str, list[SupervisedDataset]] = {}
        for i, aux_src in enumerate(self.auxiliary_sources):
            try:
                self._task_value = i + 1
                experiment_data_raw = extract_experiment_data(
                    experiment=aux_src.experiment,
                    data_loader_config=self._data_loader_config,
                    data=aux_src.get_data_to_transfer_from(
                        target_metric=target_outcome
                    ),
                )
                experiment_data_raw = self._update_metadata_with_task(
                    experiment_data=experiment_data_raw
                )

            except ValueError:
                warnings.warn(
                    "Got an error while fetching source observations for "
                    f"{target_outcome} from {aux_src.experiment.name}. "
                    f"Skipping this source for {target_outcome}.",
                    stacklevel=2,
                )
                continue

            # Find status quo as Observation for the auxiliary source.
            if aux_src.experiment.status_quo is None:
                sq_obs = None
            else:
                # Get the latest SQ observation.
                sq_name = aux_src.experiment.status_quo.name
                try:
                    sq_obs = (
                        experiment_data_raw.filter_by_arm_names([sq_name])
                        .filter_latest_observations()
                        .convert_to_list_of_observations()[0]
                    )
                except (KeyError, IndexError):
                    # No SQ observation found.
                    sq_obs = None

            experiment_data = aux_src.map_experiment_data(
                experiment_data=experiment_data_raw,
                target_search_space=self.model_space,
            )
            # Transforms are applied to the auxiliary source observations that takes
            # the adapter as input. This is a hack to pass in the status quo and
            # training data of the auxiliary source to the transforms and not use
            # the ones from the target experiment.
            with (
                mock.patch.object(self, "_status_quo", sq_obs),
                mock.patch.object(self, "_training_data", experiment_data_raw),
                mock.patch.object(self, "_experiment", aux_src.experiment),
            ):
                experiment_data, _ = self._transform_data(
                    experiment_data=experiment_data,
                    search_space=self._model_space,
                    transforms=self._raw_transforms,
                    transform_configs=self._transform_configs,
                    assign_transforms=False,
                )
            # If using heterogeneous search spaces, we need to get all parameters
            # from the auxiliary source. Otherwise, we can filter down to the parameters
            # in the target space.
            params_to_fetch = (
                get_mapped_parameter_names(
                    aux_src=aux_src,
                    target_search_space=self._model_space,
                    transforms=self.transforms,
                )
                if self._heterogeneous_search_space
                else parameters
            )
            datasets, _, _ = self._convert_experiment_data(
                experiment_data=experiment_data,
                parameters=params_to_fetch,
                outcomes=aux_src.get_metrics_to_transfer_from(target_outcome),
                search_space_digest=None,
            )
            # Update the outcome names with <experiment_name>_<outcome_name>
            # to prevent naming collisions between source & target datasets.
            for ds in datasets:
                ds.outcome_names = [
                    f"{aux_src.experiment.name}_{outcome}"
                    for outcome in ds.outcome_names
                ]
            transferable_datasets[aux_src.experiment.name] = datasets
        # reset task value to target task
        self._task_value = TARGET_TASK_VALUE
        return transferable_datasets

    def _get_task_datasets(
        self,
        datasets: Sequence[SupervisedDataset | None],
        parameters: list[str],
    ) -> list[MultiTaskDataset]:
        task_datasets = []
        for dataset, outcome in zip(datasets, self.outcomes, strict=True):
            transferable_datasets = self.get_transferable_datasets(
                parameters=parameters, target_outcome=outcome
            )
            # This just concatenates the lists of transferable datasets.
            all_source_datasets = sum(transferable_datasets.values(), [])
            if dataset is None:
                # Target experiment has no data. Make empty dataset.
                tkwargs: dict[str, Any] = {"dtype": torch.double, "device": self.device}
                dataset = SupervisedDataset(
                    X=torch.empty((0, len(parameters)), **tkwargs),
                    Y=torch.empty((0, 1), **tkwargs),
                    Yvar=(
                        None
                        if all_source_datasets[0].Yvar is None
                        else torch.empty((0, 1), **tkwargs)
                    ),
                    feature_names=parameters,
                    outcome_names=[outcome],
                )
            task_datasets.append(
                MultiTaskDataset(
                    datasets=[dataset] + all_source_datasets,
                    target_outcome_name=outcome,
                    task_feature_index=-1,
                )
            )
        return task_datasets

    def _fit(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
        **kwargs: Any,
    ) -> None:
        if self.generator is not None and experiment_data == self._last_experiment_data:
            logger.debug(
                "The experiment data is identical to the last experiment data "
                "used to fit the generator. Skipping generator fitting."
            )
            return
        outcomes = self.outcomes
        if experiment_data.arm_data.empty:
            # Temporarily unset self.outcomes to avoid an error in _get_fit_args.
            self.outcomes = []
        datasets, candidate_metadata, search_space_digest = self._get_fit_args(
            search_space=search_space,
            experiment_data=experiment_data,
            update_outcomes_and_parameters=True,
        )
        if experiment_data.arm_data.empty:
            self.outcomes = outcomes
            # Temporarily set datasets to None. We will construct empty datasets
            # constructing the MultiTaskDataset.
            datasets = [None for _ in self.outcomes]
        # Add the task feature to SSD, to ensure that a multi-task model is selected.
        if len(search_space_digest.task_features) != 1:
            raise UnsupportedError(
                "Task features are not supported in transfer learning."
            )
        task_datasets = self._get_task_datasets(
            datasets=datasets,
            parameters=self.parameters,
        )
        # Fit
        self.generator.fit(
            # Datasets are guaranteed to have all outcomes here by construction.
            datasets=task_datasets,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            **kwargs,
        )
        # This is a bit of a hack to ensure that only the data for the target task
        # is used in the X_baseline. It also avoids task feature in X_baseline.
        self.generator.surrogate._training_data = [
            ds.datasets[ds.target_outcome_name] for ds in task_datasets
        ]

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: ExperimentData,
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        if self.parameters is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_cross_validate"))
        datasets, _, search_space_digest = self._get_fit_args(
            search_space=search_space,
            experiment_data=cv_training_data,
            update_outcomes_and_parameters=False,
        )
        # Add the task feature to SSD, to ensure that a multi-task model is selected.
        if len(search_space_digest.task_features) > 1:
            raise UnsupportedError(
                "Task features are not supported in transfer learning."
            )

        X_test = torch.tensor(
            [[obsf.parameters[p] for p in self.parameters] for obsf in cv_test_points],
            dtype=torch.double,
            device=self.device,
        )
        task_datasets = self._get_task_datasets(
            datasets=datasets,
            parameters=self.parameters,
        )
        # Use the generator to do the cross validation
        f_test, cov_test = assert_is_instance(
            self.generator,
            BoTorchGenerator,
        ).cross_validate(
            datasets=task_datasets,
            X_test=torch.as_tensor(X_test, dtype=torch.double, device=self.device),
            search_space_digest=search_space_digest,
            use_posterior_predictive=use_posterior_predictive,
        )
        # Convert array back to ObservationData
        return array_to_observation_data(
            f=f_test.detach().cpu().clone().numpy(),
            cov=cov_test.detach().cpu().clone().numpy(),
            outcomes=self.outcomes,
        )

    def gen(
        self,
        n: int,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        model_gen_options: TConfig | None = None,
    ) -> GeneratorRun:
        """Generates new points using the base ``Adapter.gen``.

        Once the ``GeneratorRun`` is produced, it checks for any fixed parameters
        that are not in the target search space and removes them. This is a hack
        around limitations of the ``RemoveFixed`` transform. Since we construct the
        transforms with the joint space, we end up adding back all fixed parameters
        from the joint space rather than adding only the parameters from the
        target search space. A proper fix would require passing in the search space
        to ``RemoveFixed.untransform_observation_features``, which requires updating
        the signature of all transforms.
        """
        # If a fixed parameter in the target search space, is
        # a range parameter in the joint search space, then we
        # should set it as a fixed feature here.
        search_space = search_space or self._search_space
        for name, target_p in search_space.parameters.items():
            if (
                isinstance(target_p, FixedParameter)
                and (p := self.joint_search_space.parameters.get(name)) is not None
                and isinstance(p, RangeParameter)
            ):
                # add to fixed features
                if fixed_features is None:
                    fixed_features = ObservationFeatures(parameters={})
                fixed_features.parameters.setdefault(name, target_p.value)
        generator_run = super().gen(
            n=n,
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
        )
        # Remove the parameters that are not in the target experiment's search
        # space, and update candidate_metadata_by_arm_signature to reflect the
        # new arm. We use the experiment's search space rather than
        # self._model_space because the model space may include source-only
        # parameters that were added as FixedParameter for backfilling purposes.
        for arm in generator_run.arms:
            # Capture the old arm signature before it changes (`arm.signature` is
            # dynamic and computed on the fly using the current arm parameter values, so
            # it changes when we remove parameters from the arm).
            old_signature = arm.signature
            for param in arm.parameters:
                if param not in self._experiment.search_space.parameters:
                    arm._parameters.pop(param)
                    candidate_metadata = (
                        generator_run.candidate_metadata_by_arm_signature
                    )
                    # if the old signature of the arm is in the metadata, associate
                    # its value with the new signature of the arm instead and remove
                    # the old signature from the metadata
                    if (
                        candidate_metadata is not None
                        and old_signature in candidate_metadata
                    ):
                        arm_metadata = candidate_metadata.pop(old_signature)
                        candidate_metadata[arm.signature] = arm_metadata
        return generator_run


def transfer_learning_generator_specs_constructor(
    model_class: type[MultiTaskGP] = MultiTaskGP,
    transforms: list[type[Transform]] | None = None,
    jit_compile: bool | None = None,
    torch_device: torch.device | None = None,
    derelativize_with_raw_status_quo: bool = False,
    use_model_selection: bool = True,
    mll_kwargs: dict[str, Any] | None = None,
    covar_module_class: type[Kernel] | None = None,
    covar_module_kwargs: dict[str, Any] | None = None,
    botorch_model_kwargs: dict[str, Any] | None = None,
    fit_tracking_metrics: bool = True,
    additional_generator_kwargs: dict[str, Any] | None = None,
) -> tuple[list[GeneratorSpec], SingleDiagnosticBestModelSelector | None]:
    r"""Constructs the generator specs for transfer learning.
    Args:
        model_class: The MultiTask BoTorch Model to use in the BOTL.
        transform: Optional list of transforms to use in the Adapter.
            Defaults to MBM_X_trans + [MetadataToTask] + Y_trans.
        jit_compile: Whether to use jit compilation in Pyro when the fully Bayesian
            model is used.
        torch_device: What torch device to use (defaults to None, i.e. falls back to
            the PyTorch default device).
        derelativize_with_raw_status_quo: Whether to derelativize using the raw status
            quo values in any transforms. This argument is primarily to allow automatic
            Winsorization when the optimization config contains relative outcome
            constraints or objective thresholds. Note: automatic Winsorization will fail
            if this is set to `False` and there are relative constraints present.
        use_model_selection: Enables model selection where both a multi task (BOTL)
            and a single task model are fit before generating each trial. If the
            single task model produces "better" cross validation results, it is used
            for candidate generation, with the aim of protecting against negative
            transfer. Defaults to True.
        mll_kwargs: Keyword arguments to pass to the MLL to be passed to SurrogateSpec.
        covar_module_class: Covariance module class to be passed to SurrogateSpec.
        covar_module_kwargs: Covariance module kwargs to be passed to SurrogateSpec.
        botorch_model_kwargs: Additional kwargs to be passed to the BoTorch model
            while initializing the model. These kwargs must be supported by the
            model's input constructor.
        fit_tracking_metrics: Whether to fit the generator on tracking metrics. Passed
            to the `TransferLearningAdapter`.
        additional_generator_kwargs: Additional kwargs to be passed
            to the BoTorchGenerator

    Returns:
        A tuple containing BOTL generator specs in case model selection is not enabled,
        and both BOTL and BOTORCH_MODULAR generator specs as well as a best model
        selector in case there is model selection enabled.
    """
    input_transform_classes: list[type[InputTransform]] = [Normalize]
    input_transform_options = {
        "Normalize": {
            # None for bounds here ensures we do not use bounds from
            # the search space digest.
            "bounds": None,
        }
    }
    transforms = transforms or MBM_X_trans + [MetadataToTask] + Y_trans
    transform_configs = get_derelativize_config(
        derelativize_with_raw_status_quo=derelativize_with_raw_status_quo
    )

    mll_kwargs = {} if mll_kwargs is None else mll_kwargs

    if model_class.__name__ == "SaasFullyBayesianMultiTaskGP":
        mll_kwargs.update({"jit_compile": jit_compile or False})

    additional_generator_kwargs = additional_generator_kwargs or {}
    generator_kwargs: dict[str, Any] = {
        "surrogate_spec": SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=model_class,
                    model_options=botorch_model_kwargs or {},
                    input_transform_classes=input_transform_classes,
                    input_transform_options=input_transform_options,
                    mll_options=mll_kwargs,
                    covar_module_class=covar_module_class,
                    covar_module_options=covar_module_kwargs or {},
                )
            ]
        ),
        "transforms": transforms,
        "transform_configs": transform_configs,
        "fit_tracking_metrics": fit_tracking_metrics,
        "torch_device": torch_device,
        **additional_generator_kwargs,
    }

    botl_specs = [
        GeneratorSpec(
            generator_enum=Generators.BOTL, generator_kwargs=generator_kwargs
        ),
    ]
    if use_model_selection:
        botl_specs.append(
            GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR),
        )
        best_model_selector = SingleDiagnosticBestModelSelector(
            diagnostic="Rank correlation",
            metric_aggregation=ReductionCriterion.MEAN,
            criterion=ReductionCriterion.MAX,
            cv_kwargs={"untransform": False},
        )
    else:
        best_model_selector = None

    return botl_specs, best_model_selector


# Register BOTL in the global generator registry.
# This is done here (rather than in registry.py) to avoid circular imports,
# following the pattern used by ax/fb/adapter/registry.py.
GENERATOR_KEY_TO_GENERATOR_SETUP["BOTL"] = GeneratorSetup(
    adapter_class=TransferLearningAdapter,
    generator_class=BoTorchGenerator,
    transforms=MBM_X_trans + [MetadataToTask] + Y_trans,
)
