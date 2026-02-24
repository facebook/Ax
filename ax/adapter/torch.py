#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from ax.adapter.adapter_utils import (
    arm_to_np_array,
    array_to_observation_data,
    extract_objective_thresholds,
    extract_objective_weight_matrix,
    extract_objective_weights,
    extract_outcome_constraints,
    extract_parameter_constraints,
    extract_search_space_digest,
    get_fixed_features,
    observation_data_to_array,
    observation_features_to_array,
    parse_observation_features,
    pending_observations_as_array_list,
    prep_pairwise_data,
    process_contextual_datasets,
    SearchSpaceDigest,
    transform_callback,
    validate_and_apply_final_transform,
)
from ax.adapter.base import Adapter, DataLoaderConfig, gen_arms, GenResults
from ax.adapter.data_utils import ExperimentData, extract_experiment_data
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.cast import Cast
from ax.adapter.transforms.choice_encode import (
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
)
from ax.adapter.transforms.derelativize import Derelativize
from ax.adapter.transforms.fill_missing_parameters import FillMissingParameters
from ax.adapter.transforms.int_range_to_choice import IntRangeToChoice
from ax.adapter.transforms.log import Log
from ax.adapter.transforms.relativize import (
    RelativizeWithConstantControl,
    SelectiveRelativizeWithConstantControl,
)
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.adapter.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.adapter.transforms.trial_as_task import TrialAsTask
from ax.adapter.transforms.unit_x import UnitX
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import extract_arm_predictions
from ax.core.metric import Metric
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.observation_utils import separate_observations
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    ScalarizedOutcomeConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TModelPredictArm
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.exceptions.generation_strategy import OptimizationConfigRequired
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_moo_utils import infer_objective_thresholds
from ax.generators.torch.utils import (
    _get_X_pending_and_observed,
    extract_objectives,
)
from ax.generators.torch_base import TorchGenerator, TorchOptConfig
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.models.model import Model
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from pyre_extensions import none_throws
from torch import Tensor

logger: Logger = get_logger(__name__)

FIT_MODEL_ERROR = "Generator must be fit before {action}."

# Transforms allowed for Bayesian Optimization with Preference Exploration.
# These transforms do not modify Y's tensor shape, order, or scale (except
# Relativize which is explicitly expected when using relativized outcomes).
# Any transform that modifies outcome values in ways that break preference
# model assumptions should NOT be in this set.
# Note: Y-transforms like StandardizeY, Winsorize, BilogY are NOT allowed
# as they modify outcome scale/distribution in ways incompatible with
# preference learning.
BOPE_ALLOWED_TRANSFORMS: set[type[Transform]] = {
    # Parameter only transforms
    Cast,
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
    FillMissingParameters,
    IntRangeToChoice,
    Log,
    RemoveFixed,
    SearchSpaceToChoice,
    TaskChoiceToIntTaskChoice,
    TrialAsTask,
    UnitX,
    # Outcome transforms
    Derelativize,  # Doesn't modify outcome values
    TransformToNewSQ,  # Doesn't distort outcome scales
    # not allowing Relativize here as it doesn't guarantee
    # the untransformed SEM is always valid
    RelativizeWithConstantControl,  # Outcome transfom for BOPE
    SelectiveRelativizeWithConstantControl,  # Conditional transform for BOPE
}


class TorchAdapter(Adapter):
    """An adapter for using torch-based generators.

    Specifies an interface that is implemented by TorchGenerator. In particular,
    generator should have methods fit, predict, and gen. See TorchGenerator for the
    API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the generator.
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        generator: TorchGenerator,
        search_space: SearchSpace | None = None,
        data: Data | None = None,
        transforms: Sequence[type[Transform]] | None = None,
        transform_configs: Mapping[str, TConfig] | None = None,
        optimization_config: OptimizationConfig | None = None,
        expand_model_space: bool = True,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        default_model_gen_options: TConfig | None = None,
        torch_device: torch.device | None = None,
        data_loader_config: DataLoaderConfig | None = None,
    ) -> None:
        """In addition to common arguments documented in the base ``Adapter`` class,
        ``TorchAdapter`` accepts the following arguments.

        Args:
            default_model_gen_options: A dictionary of default options to use
                during candidate generation. These will be overridden by any
                `model_gen_options` passed to the `Adapter.gen` method.
            torch_device: The device to use for any torch tensors and operations
                on these tensors.
            data_loader_config: A DataLoaderConfig of options for loading data. See the
                docstring of DataLoaderConfig for more details.
        """
        self.device: torch.device | None = torch_device
        self._default_model_gen_options: TConfig = default_model_gen_options or {}

        self.is_moo_problem: bool = False
        self.preference_profile_name: str | None = None
        if optimization_config or (experiment and experiment.optimization_config):
            optimization_config = none_throws(
                optimization_config or experiment.optimization_config
            )
            # Handle init for multi-objective optimization.
            self.is_moo_problem = optimization_config.is_moo_problem

            if isinstance(optimization_config, PreferenceOptimizationConfig):
                self.preference_profile_name = (
                    optimization_config.preference_profile_name
                )

        # Tracks last experiment data used to fit the generator, to skip
        # generator fitting when it's not necessary.
        self._last_experiment_data: ExperimentData | None = None

        # These are set in _fit.
        self.parameters: list[str] = []
        self.outcomes: list[str] = []

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            generator=generator,
            transforms=transforms,
            transform_configs=transform_configs,
            optimization_config=optimization_config,
            expand_model_space=expand_model_space,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
            data_loader_config=data_loader_config,
        )

        # Re-assign self.generator for more precise typing.
        self.generator: TorchGenerator = generator

    def feature_importances(self, metric_signature: str) -> dict[str, float]:
        importances_tensor = self.generator.feature_importances()
        importances_dict = dict(zip(self.outcomes, importances_tensor, strict=True))
        importances_arr = importances_dict[metric_signature].flatten()
        return dict(zip(self.parameters, importances_arr, strict=True))

    @property
    def botorch_model(self) -> Model:
        """Returns the underlying BoTorch model for BoTorchGenerator."""
        if not isinstance(self.generator, BoTorchGenerator):
            raise UnsupportedError(
                "Generator must be a BoTorchGenerator to "
                f"access botorch_model. Found {type(self.generator)}."
            )
        return self.generator.surrogate.model

    def infer_objective_thresholds(
        self,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> list[ObjectiveThreshold]:
        """Infer objective thresholds.

        This method is only applicable for Multi-Objective optimization problems.

        This method uses the generator-estimated Pareto frontier over the in-sample
        points to infer absolute (not relativized) objective thresholds.

        This uses a heuristic that sets the objective threshold to be a scaled nadir
        point, where the nadir point is scaled back based on the range of each
        objective across the current in-sample Pareto frontier.
        """
        if not self.is_moo_problem:
            raise UnsupportedError(
                "Objective thresholds are only supported for multi-objective "
                "optimization."
            )

        search_space = (search_space or self._search_space).clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            fixed_features=fixed_features,
        )
        # Get transformed args from TorchAdapter.
        search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            fixed_features=base_gen_args.fixed_features,
            pending_observations={},
            optimization_config=base_gen_args.optimization_config,
        )
        # Infer objective thresholds.
        if isinstance(self.generator, BoTorchGenerator):
            model = self.botorch_model
            Xs = self.generator.surrogate.Xs
        else:
            raise UnsupportedError(
                "Generator must be a Modular Botorch Generator to "
                f"infer_objective_thresholds. Found {type(self.generator)}."
            )

        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            pending_observations=torch_opt_config.pending_observations,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
        )
        if X_observed is None:
            raise DataRequiredError(
                "No complete observations found for the given optimization config. "
                "Cannot infer objective thresholds."
            )
        obj_thresholds = infer_objective_thresholds(
            model=none_throws(model),
            objective_weights=torch_opt_config.objective_weights,
            X_observed=X_observed,
            outcome_constraints=torch_opt_config.outcome_constraints,
        )

        return self._untransform_objective_thresholds(
            objective_thresholds=obj_thresholds,
            objective_weights=torch_opt_config.objective_weights,
            opt_config_metrics=torch_opt_config.opt_config_metrics,
            fixed_features=torch_opt_config.fixed_features,
        )

    def model_best_point(
        self,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        model_gen_options: TConfig | None = None,
    ) -> tuple[Arm, TModelPredictArm | None] | None:
        # Get modifiable versions
        if search_space is None:
            search_space = self._search_space
        search_space = search_space.clone()

        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )
        search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            model_gen_options=model_gen_options,
            optimization_config=base_gen_args.optimization_config,
        )
        try:
            xbest = self.generator.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        except NotImplementedError:
            xbest = None

        if xbest is None:
            return None

        best_obsf = ObservationFeatures(
            parameters={p: float(xbest[i]) for i, p in enumerate(self.parameters)}
        )

        for t in reversed(list(self.transforms.values())):
            best_obsf = t.untransform_observation_features([best_obsf])[0]

        best_point_predictions = extract_arm_predictions(
            model_predictions=self.predict([best_obsf]), arm_idx=0
        )

        best_arms, _ = gen_arms(
            observation_features=[best_obsf],
            arms_by_signature=self._arms_by_signature,
        )
        best_arm = best_arms[0]

        return best_arm, best_point_predictions

    def _array_callable_to_tensor_callable(
        self,
        array_func: Callable[[npt.NDArray], npt.NDArray],
    ) -> Callable[[Tensor], Tensor]:
        tensor_func: Callable[[Tensor], Tensor] = lambda x: (
            self._array_to_tensor(array_func(x.detach().cpu().clone().numpy()))
        )
        return tensor_func

    def _array_to_tensor(self, array: npt.NDArray | list[float]) -> Tensor:
        return torch.as_tensor(array, dtype=torch.double, device=self.device)

    def _convert_experiment_data(
        self,
        experiment_data: ExperimentData,
        outcomes: list[str],
        parameters: list[str],
        search_space_digest: SearchSpaceDigest | None,
    ) -> tuple[
        list[SupervisedDataset], list[str], list[list[TCandidateMetadata]] | None
    ]:
        """Converts ``ExperimentData`` to a dictionary of ``Dataset`` containers, a list
        of outcomes -- in the same order as the datasets -- and candidate metadata.
        The rows that have missing / NaN mean observations are dropped before
        constructing the dataset for the corresponding outcome.

        Args:
            experiment_data: A container of two dataframes ``arm_data`` and
                ``observation_data``, containing parameterizations, observations,
                and metadata extracted from ``Trial``s and ``Data`` of the experiment.
            outcomes: The names of the outcomes to extract observations for.
            parameters: The names of the parameters to extract. Any additional columns
                of ``arm_data`` that is not included in `parameters` will be ignored.
            search_space_digest: Optional ``SearchSpaceDigest`` containing information
                about the search space. This is used to convert datasets into a
                ``MultiTaskDataset`` where applicable.

        Returns:
            - A list of ``Dataset`` objects. The datasets will be for the set of
                outcomes specified in ``outcomes``, not necessarily in that order.
                Some outcomes will be grouped into a single dataset if there are
                contextual datasets.
            - A list of outcomes in the order that they appear in the datasets,
                accounting for reordering made necessary by contextual datasets.
            - An optional list of lists of candidate metadata. Each inner list
                corresponds to one outcome. Each element in the inner list corresponds
                to one observation.
                NOTE: Candidate metadata is currently only utilized in TRBO generator.
        """
        if len(outcomes) == 0:
            return [], [], None
        arm_data = experiment_data.arm_data
        obs_data = experiment_data.observation_data
        obs_data_mean = obs_data["mean"]
        sems_df = obs_data["sem"]
        # Check for duplication between parameter names and metric names
        obs_mean_cols = set(obs_data_mean.columns)
        arm_data_cols = set(arm_data.columns)
        duplicated_names = obs_mean_cols.intersection(arm_data_cols)

        # Join mean & arms to align and repeat the arm rows if necessary.
        # Add suffix if there are duplicate column names.
        mean_and_params = obs_data_mean.join(
            arm_data, how="left", lsuffix="_metric", rsuffix="_parameter"
        )
        # Reindex to only trial_index and arm_name, to move
        # any progression columns out of the index.
        levels_to_move = list(
            set(mean_and_params.index.names).difference({"trial_index", "arm_name"})
        )
        if len(levels_to_move) > 0:
            # This is a copy of the original df. We can modify in-place for cheaper.
            mean_and_params.reset_index(level=levels_to_move, inplace=True)
        # This will include the progression if it is in parameters.
        # This is also tolerant to missing columns, which is relevant for TL.
        params_np = (
            mean_and_params.filter(
                [i + "_parameter" if i in duplicated_names else i for i in parameters]
            ).to_numpy()
            # In some cases, this ends up as object, which is not supported by torch.
            # This can happen if an Int64 column had NaNs in it at some stage.
            .astype(float)
        )
        trial_indices_np = mean_and_params.index.get_level_values(
            "trial_index"
        ).to_numpy()
        metadata = mean_and_params["metadata"]
        datasets: list[SupervisedDataset] = []
        candidate_metadata = []
        for outcome in outcomes:
            outcome_col_name = (
                outcome + "_metric" if outcome in duplicated_names else outcome
            )
            if outcome_col_name not in mean_and_params:
                raise DataRequiredError(
                    f"Attempting to extract a dataset for {outcome=} but no "
                    "corresponding data was found in the experiment data."
                )
            # Drop NaN columns from means & corresponding params.
            outcome_means = mean_and_params[outcome_col_name].to_numpy()
            to_keep = ~np.isnan(outcome_means)
            Y = torch.from_numpy(outcome_means[to_keep]).double().view(-1, 1)
            X = torch.from_numpy(params_np[to_keep]).double()
            sem = sems_df[outcome].to_numpy()[to_keep]
            if np.all(np.isnan(sem)):
                Yvar = None
            else:
                Yvar = torch.from_numpy(sem).double().square().view(-1, 1)
            group_indices = torch.from_numpy(trial_indices_np[to_keep])
            if outcome == Keys.PAIRWISE_PREFERENCE_QUERY.value:
                dataset = prep_pairwise_data(
                    X=X.to(device=self.device),
                    Y=Y.to(dtype=torch.long, device=self.device),
                    group_indices=group_indices,
                    outcome=outcome,
                    parameters=parameters,
                )
            else:
                dataset = SupervisedDataset(
                    X=X.to(device=self.device),
                    Y=Y.to(device=self.device),
                    Yvar=Yvar.to(device=self.device) if Yvar is not None else None,
                    feature_names=parameters,
                    outcome_names=[outcome],
                    group_indices=group_indices,
                )
            datasets.append(dataset)
            candidate_metadata.append(metadata.loc[to_keep].to_list())

        # If the search space digest specifies a task feature,
        # convert the datasets into MultiTaskDataset.
        if search_space_digest is not None and (
            task_features := search_space_digest.task_features
        ):
            if len(task_features) > 1:
                raise UnsupportedError("Multiple task features are not supported.")
            target_task_value = search_space_digest.target_values[task_features[0]]
            # Both the MTGP and the MTDataset expect integer valued task features.
            # Check that they're indeed integers.
            task_choices = search_space_digest.discrete_choices[task_features[0]]
            if any(int(t) != t for t in task_choices):
                raise ValueError(
                    "The values of the task feature must be integers. "
                    "This is often accomplished using a "
                    "TaskChoiceToIntTaskChoice transform. "
                    "Check that the generator is using the correct set of transforms. "
                    f"Got {task_choices=}."
                )
            datasets = [
                MultiTaskDataset.from_joint_dataset(
                    dataset=dataset,
                    task_feature_index=task_features[0],
                    target_task_value=int(target_task_value),
                )
                for dataset in datasets
            ]
        # Check if there is a `parameter_decomposition` experiment property to
        # decide whether it is a contextual experiment.
        if self._experiment_properties.get("parameter_decomposition", None) is not None:
            # Convert to a list of ContextualDateset for contextual experiments.
            # pyre-ignore [9]: ContextualDataset is a subclass of SupervisedDataset.
            datasets = process_contextual_datasets(
                datasets=datasets,
                outcomes=outcomes,
                parameter_decomposition=self._experiment_properties[
                    "parameter_decomposition"
                ],
                metric_decomposition=self._experiment_properties.get(
                    "metric_decomposition", None
                ),
            )

        # Get the order of outcomes
        ordered_outcomes = []
        for d in datasets:
            ordered_outcomes.extend(d.outcome_names)
        # Re-order candidate metadata
        if not metadata.isnull().all():
            ordered_metadata = []
            for outcome in ordered_outcomes:
                ordered_metadata.append(candidate_metadata[outcomes.index(outcome)])
        else:
            ordered_metadata = None

        return datasets, ordered_outcomes, ordered_metadata

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: ExperimentData,
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Make predictions at ``cv_test_points`` using only the data
        in ``cv_training_data``.
        """
        if self.parameters is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_cross_validate"))
        datasets, _, search_space_digest = self._get_fit_args(
            search_space=search_space,
            experiment_data=cv_training_data,
            update_outcomes_and_parameters=False,
        )
        X_test = torch.tensor(
            [[obsf.parameters[p] for p in self.parameters] for obsf in cv_test_points],
            dtype=torch.double,
            device=self.device,
        )
        # Use the generator to do the cross validation
        f_test, cov_test = self.generator.cross_validate(
            datasets=datasets,
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

    def evaluate_acquisition_function(
        self,
        observation_features: (
            list[ObservationFeatures] | list[list[ObservationFeatures]]
        ),
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        acq_options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Evaluate the acquisition function for given set of observation
        features.

        Args:
            observation_features: Either a list or a list of lists of observation
                features, representing parameterizations, for which to evaluate the
                acquisition function. If a single list is passed, the acquisition
                function is evaluated for each observation feature. If a list of lists
                is passed each element (itself a list of observation features)
                represents a batch of points for which to evaluate the joint acquisition
                value.
            search_space: Search space for fitting the generator.
            optimization_config: Optimization config defining how to optimize
                the generator.
            pending_observations: A map from metric name to pending observations for
                that metric.
            fixed_features: An ObservationFeatures object containing any features that
                should be fixed at specified values during generation.
            acq_options: Keyword arguments used to construct the acquisition function.

        Returns:
            A list of acquisition function values, in the same order as the
            input observation features.
        """
        search_space = search_space or self._search_space
        optimization_config = optimization_config or self._optimization_config
        if optimization_config is None:
            raise ValueError(
                "The `optimization_config` must be specified either while initializing "
                "the Adapter or to the `evaluate_acquisition_function` call."
            )
        # pyre-ignore Incompatible parameter type [9]
        obs_feats: list[list[ObservationFeatures]] = deepcopy(observation_features)
        if not isinstance(obs_feats[0], list):
            obs_feats = [[obs] for obs in obs_feats]

        for t in self.transforms.values():
            for i, batch in enumerate(obs_feats):
                obs_feats[i] = t.transform_observation_features(batch)

        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )

        return self._evaluate_acquisition_function(
            observation_features=obs_feats,
            search_space=base_gen_args.search_space,
            optimization_config=none_throws(base_gen_args.optimization_config),
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            acq_options=acq_options,
        )

    def _evaluate_acquisition_function(
        self,
        observation_features: list[list[ObservationFeatures]],
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        acq_options: dict[str, Any] | None = None,
    ) -> list[float]:
        if not self.parameters:
            raise RuntimeError(
                FIT_MODEL_ERROR.format(action="_evaluate_acquisition_function")
            )
        search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(
            search_space=search_space,
            pending_observations=pending_observations or {},
            fixed_features=fixed_features or ObservationFeatures({}),
            optimization_config=optimization_config,
        )
        X = np.array(
            [
                observation_features_to_array(self.parameters, obsf)
                for obsf in observation_features
            ]
        )
        evals = self.generator.evaluate_acquisition_function(
            X=self._array_to_tensor(X),
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        return evals.tolist()

    def _update_w_aux_exp_datasets(
        self, datasets: list[SupervisedDataset]
    ) -> list[SupervisedDataset]:
        aux_datasets = []
        # Extract datasets needed from auxiliary experiments
        # For preference exploration
        optimization_config = self._optimization_config
        if isinstance(optimization_config, PreferenceOptimizationConfig):
            target_pe_aux_exp_name = optimization_config.preference_profile_name
            target_pe_aux_exp = none_throws(
                self._experiment.find_auxiliary_experiment_by_name(
                    purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
                    auxiliary_experiment_name=target_pe_aux_exp_name,
                    raise_if_not_found=True,
                )
            )
            # This is the name of the PE experiment on which we fit
            # the learned objective preference model
            self.preference_profile_name = target_pe_aux_exp_name
            pe_exp, pe_data = target_pe_aux_exp.experiment, target_pe_aux_exp.data

            if pe_data.df.empty:
                raise DataRequiredError(
                    "No data found in the auxiliary preference exploration "
                    "experiment. Play the preference game first or use another "
                    "preference profile with recorded preference data."
                )

            pe_experiment_data = extract_experiment_data(
                experiment=pe_exp,
                data=pe_data,
                data_loader_config=self._data_loader_config,
            )
            pe_exp_param_names = list(pe_exp.search_space.parameters.keys())

            # Get all relevant information on the parameters
            pe_ssd = extract_search_space_digest(
                search_space=pe_exp.search_space, param_names=pe_exp_param_names
            )
            pe_outcomes = [Keys.PAIRWISE_PREFERENCE_QUERY.value]

            pe_datasets, _, _ = self._convert_experiment_data(
                experiment_data=pe_experiment_data,
                outcomes=pe_outcomes,
                parameters=pe_exp_param_names,
                search_space_digest=pe_ssd,
            )
            aux_datasets.extend(pe_datasets)
        return datasets + aux_datasets

    def _get_fit_args(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
        update_outcomes_and_parameters: bool,
    ) -> tuple[
        list[SupervisedDataset],
        list[list[TCandidateMetadata]] | None,
        SearchSpaceDigest,
    ]:
        """Helper for consolidating some common argument processing between
        ``fit`` and ``cross_validate`` methods. Extracts datasets and candidate metadata
        from ``experiment_data``, and ``search_space_digest`` from the ``search_space``.

        Args:
            search_space: A transformed search space for fitting the generator.
            experiment_data: A container of two dataframes ``arm_data`` and
                ``observation_data``, containing parameterizations, observations,
                and metadata extracted from ``Trial``s and ``Data`` of the experiment.
            update_outcomes_and_parameters: Whether to update `self.outcomes` with
                all outcomes found in the observations and `self.parameters` with
                all parameters in the search space. Typically only used in `_fit`.

        Returns:
            The datasets & metadata, extracted from the ``experiment_data``, and the
            ``search_space_digest``.
        """
        self._last_experiment_data = experiment_data
        if update_outcomes_and_parameters:
            self.parameters = list(search_space.parameters.keys())
            # Make sure that task feature is the last parameter. This is important
            # for heterogeneous search spaces
            if Keys.TASK_FEATURE_NAME.value in self.parameters:
                idx = self.parameters.index(Keys.TASK_FEATURE_NAME.value)
                if idx != len(self.parameters) - 1:
                    self.parameters = (
                        self.parameters[:idx]
                        + self.parameters[idx + 1 :]
                        + [Keys.TASK_FEATURE_NAME.value]
                    )
        # Only update outcomes if fitting a model on tracking metrics. Otherwise,
        # we will only fit models to the outcomes that are extracted from optimization
        # config in Adapter.__init__.
        if update_outcomes_and_parameters and self._fit_tracking_metrics:
            self.outcomes = sorted(experiment_data.metric_signatures)
        # Get all relevant information on the parameters
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        # Convert observations to datasets
        datasets, ordered_outcomes, candidate_metadata = self._convert_experiment_data(
            experiment_data=experiment_data,
            outcomes=self.outcomes,
            parameters=self.parameters,
            search_space_digest=search_space_digest,
        )
        datasets = self._update_w_aux_exp_datasets(datasets=datasets)

        if update_outcomes_and_parameters:
            self.outcomes = ordered_outcomes
        else:
            assert ordered_outcomes == self.outcomes, (
                f"Unexpected ordering of outcomes: {ordered_outcomes} != "
                f"{self.outcomes}"
            )
        return datasets, candidate_metadata, search_space_digest

    def _fit(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
        **kwargs: Any,
    ) -> None:
        if experiment_data == self._last_experiment_data:
            logger.debug(
                "The experiment data is identical to the last experiment data "
                "used to fit the generator. Skipping generator fitting."
            )
            return
        datasets, candidate_metadata, search_space_digest = self._get_fit_args(
            search_space=search_space,
            experiment_data=experiment_data,
            update_outcomes_and_parameters=True,
        )
        self.generator.fit(
            datasets=datasets,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            **kwargs,
        )

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: dict[str, list[ObservationFeatures]],
        fixed_features: ObservationFeatures | None,
        model_gen_options: TConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
    ) -> GenResults:
        """Generate new candidates according to search_space and
        optimization_config.

        The outcome constraints should be transformed to no longer be relative.
        """
        if not self.parameters:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))

        # Ensure the preference model we fit is the one used in optimization_config
        if isinstance(optimization_config, PreferenceOptimizationConfig) and (
            optimization_config.preference_profile_name != self.preference_profile_name
        ):
            raise UserInputError(
                "The preference profile name in the optimization config does not match "
                "the name of the preference profile used to fit the preference model. "
                f"Expected {self.preference_profile_name} but got "
                f"{optimization_config.preference_profile_name}. "
                "Consider updating `experiment.optimization_config` and refit "
                "the model before proceeding with gen."
            )

        # Validate preference learning configuration
        if isinstance(optimization_config, PreferenceOptimizationConfig):
            self._validate_preference_config(optimization_config)

        augmented_model_gen_options = {
            **self._default_model_gen_options,
            **(model_gen_options or {}),
        }
        search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(
            search_space=search_space,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=augmented_model_gen_options,
            optimization_config=optimization_config,
        )

        # Generate the candidates
        gen_results = self.generator.gen(
            n=n,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
        )

        gen_metadata = dict(gen_results.gen_metadata)
        if (
            isinstance(optimization_config, MultiObjectiveOptimizationConfig)
            and gen_metadata.get("objective_thresholds", None) is not None
        ):
            # If objective_thresholds are supplied by the user, then the transformed
            # user-specified objective thresholds are in gen_metadata. Otherwise,
            # if using a hypervolume based acquisition function, then
            # the inferred objective thresholds are in gen_metadata.
            gen_metadata["objective_thresholds"] = (
                self._untransform_objective_thresholds(
                    objective_thresholds=gen_metadata["objective_thresholds"],
                    objective_weights=torch_opt_config.objective_weights,
                    opt_config_metrics=torch_opt_config.opt_config_metrics,
                    fixed_features=torch_opt_config.fixed_features,
                )
            )

        # Transform array to observations
        observation_features = self._array_to_observation_features(
            X=gen_results.points.detach().cpu().clone().numpy(),
            candidate_metadata=gen_results.candidate_metadata,
        )
        try:
            xbest = self.generator.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        except NotImplementedError:
            xbest = None

        best_obsf = None
        if xbest is not None:
            best_obsf = ObservationFeatures(
                parameters={
                    p: float(x) for p, x in zip(self.parameters, xbest, strict=True)
                }
            )

        return GenResults(
            observation_features=observation_features,
            weights=gen_results.weights.tolist(),
            best_observation_features=best_obsf,
            gen_metadata=gen_metadata,
        )

    def _predict(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        if not self.parameters:
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        # Convert observation features to array
        X = observation_features_to_array(self.parameters, observation_features)
        f, cov = self.generator.predict(
            X=self._array_to_tensor(X),
            use_posterior_predictive=use_posterior_predictive,
        )
        f = f.detach().cpu().clone().numpy()
        cov = cov.detach().cpu().clone().numpy()
        if f.shape[-2] != X.shape[-2]:
            raise NotImplementedError(
                "Expected same number of predictions as the number of inputs but got "
                f"predictions of shape {f.shape} for inputs of shape {X.shape}. "
            )
        # Convert resulting arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _array_to_observation_features(
        self,
        X: npt.NDArray,
        candidate_metadata: Sequence[TCandidateMetadata] | None,
    ) -> list[ObservationFeatures]:
        return parse_observation_features(
            X=X, param_names=self.parameters, candidate_metadata=candidate_metadata
        )

    def _transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> Tensor:
        """Apply terminal transform to given observation features and return result
        as an N x D array of points.
        """
        try:
            tobfs = np.array(
                [
                    [float(of.parameters[p]) for p in self.parameters]
                    for of in observation_features
                ]
            )
        except (KeyError, TypeError):
            raise ValueError("Invalid formatting of observation features.")
        return self._array_to_tensor(tobfs)

    def _get_transformed_model_gen_args(
        self,
        search_space: SearchSpace,
        pending_observations: dict[str, list[ObservationFeatures]],
        fixed_features: ObservationFeatures | None,
        model_gen_options: TConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
    ) -> tuple[SearchSpaceDigest, TorchOptConfig]:
        # Validation
        if not self.parameters:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))
        # Extract search space info
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        if optimization_config is None:
            raise OptimizationConfigRequired(
                f"{self.__class__.__name__} requires an OptimizationConfig "
                "to be specified"
            )

        validate_transformed_optimization_config(optimization_config, self.outcomes)
        objective_weights = extract_objective_weight_matrix(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=self.outcomes,
        )
        pruning_target_point = arm_to_np_array(
            arm=optimization_config.pruning_target_parameterization,
            parameters=self.parameters,
        )
        linear_constraints = extract_parameter_constraints(
            search_space.parameter_constraints, self.parameters
        )
        fixed_features_dict = get_fixed_features(fixed_features, self.parameters)

        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=optimization_config.objective_thresholds,
                objective=optimization_config.objective,
                outcomes=self.outcomes,
            )
        else:
            objective_thresholds = None
        opt_config_metrics = optimization_config.metrics

        pending_array = pending_observations_as_array_list(
            pending_observations, self.outcomes, self.parameters
        )
        obj_w, out_c, lin_c, pend_o, obj_t, pruning_target_p = (
            validate_and_apply_final_transform(
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                pending_observations=pending_array,
                objective_thresholds=objective_thresholds,
                pruning_target_point=pruning_target_point,
                final_transform=self._array_to_tensor,
            )
        )
        rounding_func = self._array_callable_to_tensor_callable(
            transform_callback(self.parameters, self.transforms)
        )

        use_learned_objective = False
        if isinstance(optimization_config, PreferenceOptimizationConfig):
            use_learned_objective = True

        torch_opt_config = TorchOptConfig(
            objective_weights=obj_w,
            outcome_constraints=out_c,
            objective_thresholds=obj_t,
            linear_constraints=lin_c,
            fixed_features=fixed_features_dict,
            pending_observations=pend_o,
            model_gen_options=model_gen_options or {},
            rounding_func=rounding_func,
            opt_config_metrics=opt_config_metrics,
            use_learned_objective=use_learned_objective,
            pruning_target_point=pruning_target_p,
        )
        return search_space_digest, torch_opt_config

    def _transform_observations(
        self, observations: list[Observation]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply terminal transform to given observation data and return result.

        Converts a set of observations to a tuple of
            - a (n x d) array of X
            - an (n x m) array of means
            - an (n x m x m) array of covariances
        """
        observation_features, observation_data = separate_observations(observations)
        try:
            mean, cov = observation_data_to_array(
                outcomes=self.outcomes, observation_data=observation_data
            )
        except (KeyError, TypeError):
            raise ValueError("Invalid formatting of observation data.")
        X = self._transform_observation_features(
            observation_features=observation_features
        )
        return X, self._array_to_tensor(mean), self._array_to_tensor(cov)

    def _untransform_objective_thresholds(
        self,
        objective_thresholds: Tensor,
        objective_weights: Tensor,
        opt_config_metrics: dict[str, Metric],
        fixed_features: dict[int, float] | None,
    ) -> list[ObjectiveThreshold]:
        """Converts tensor-valued (possibly inferred) objective thresholds to
        ``ObjectiveThreshold`` objects, and untransforms to ensure they are
        on the same raw scale as the original optimization config.

        Args:
            objective_thresholds: A tensor of (possibly inferred) objective thresholds
                of shape `(num_metrics)`.
            objective_weights: A ``(n_objectives, n_outcomes)`` tensor of
                objective weights.
            opt_config_metrics: A dictionary mapping the metric name to the ``Metric``
                object from the original optimization config.
            fixed_features: A map {feature_index: value} for features that should be
                fixed to a particular value during generation. This typically includes
                the target trial index for multi-task applications.

        Returns:
            A list of ``ObjectiveThreshold``s on the raw, untransformed scale.
        """
        obj_indices, obj_weights = extract_objectives(objective_weights)
        thresholds = []
        for idx, w in zip(obj_indices, obj_weights):
            sign = torch.sign(w)
            thresholds.append(
                ObjectiveThreshold(
                    metric=opt_config_metrics[self.outcomes[idx]],
                    bound=float(objective_thresholds[idx].item()),
                    relative=False,
                    op=ComparisonOp.LEQ if sign < 0 else ComparisonOp.GEQ,
                )
            )
        fixed_features = fixed_features or {}
        fixed_features_obs = ObservationFeatures(
            parameters={
                name: fixed_features[i]
                for i, name in enumerate(self.parameters)
                if i in fixed_features
            }
        )

        for t in reversed(list(self.transforms.values())):
            if not isinstance(t, Cast):
                # Cast transform requires a valid hierarchical parameterization.
                # `fixed_features_obs` is incomplete, so it leads to an error.
                fixed_features_obs = t.untransform_observation_features(
                    [fixed_features_obs]
                )[0]
            thresholds = t.untransform_outcome_constraints(
                outcome_constraints=thresholds,
                fixed_features=fixed_features_obs,
            )

        return thresholds

    def _validate_preference_config(
        self, optimization_config: PreferenceOptimizationConfig
    ) -> None:
        """Validate BOPE configuration."""
        pe_aux_exp = self._experiment.find_auxiliary_experiment_by_name(
            purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            auxiliary_experiment_name=optimization_config.preference_profile_name,
            raise_if_not_found=True,
        )
        if pe_aux_exp is None or not pe_aux_exp.experiment.trials:
            raise DataRequiredError(
                f"Preference profile '{optimization_config.preference_profile_name}' "
                "has no data. Play the preference game first or use another "
                "preference profile with recorded preference data."
            )

        transform_classes = {type(transform) for transform in self.transforms.values()}

        # not checking Relativize here as it doesn't guarantee
        # the untransformed SEM is always valid
        has_relativize = RelativizeWithConstantControl in transform_classes
        expects_relativized = optimization_config.expect_relativized_outcomes

        if expects_relativized and not has_relativize:
            raise UnsupportedError(
                "Preference model expects outcomes in relative scale "
                "(expect_relativized_outcomes=True), but no Relativize "
                "transform found in pipeline."
            )
        elif not expects_relativized and has_relativize:
            raise UnsupportedError(
                "Relativize transform found in pipeline, but preference model "
                "expects outcomes in absolute scale "
                "(expect_relativized_outcomes=False)."
            )

        disallowed_transforms = transform_classes - BOPE_ALLOWED_TRANSFORMS
        if disallowed_transforms:
            disallowed_names = {t.__name__ for t in disallowed_transforms}
            allowed_names = {t.__name__ for t in BOPE_ALLOWED_TRANSFORMS}
            raise UnsupportedError(
                f"Transforms {disallowed_names} are not allowed with BOPE. "
                f"Allowed transforms: {allowed_names}"
            )

        self._validate_preference_metric_ordering(
            pe_aux_exp=pe_aux_exp,
            optimization_config=optimization_config,
        )

    def _validate_preference_metric_ordering(
        self,
        pe_aux_exp: Any,
        optimization_config: PreferenceOptimizationConfig,
    ) -> None:
        """Validate metric ordering between outcome and preference models."""
        preference_model_input_order = list(
            pe_aux_exp.experiment.search_space.parameters.keys()
        )

        pref_opt_metrics = [m.name for m in optimization_config.objective.metrics]

        # Get outcome order from the fitted surrogate. We must use surrogate.outcomes
        # which excludes auxiliary datasets (e.g., pairwise preference queries).
        # BOPE only supports BoTorchGenerator (MBM models).
        if not isinstance(self.generator, BoTorchGenerator):
            raise UnsupportedError(
                "Preference optimization requires a BoTorchGenerator. "
                f"Got {type(self.generator).__name__}."
            )
        outcome_model_output_order = self.generator.surrogate.outcomes

        outcome_model_pref_metrics = [
            m for m in outcome_model_output_order if m in pref_opt_metrics
        ]

        if set(pref_opt_metrics) != set(outcome_model_pref_metrics):
            missing = set(pref_opt_metrics) - set(outcome_model_pref_metrics)
            extra = set(outcome_model_pref_metrics) - set(pref_opt_metrics)
            raise UserInputError(
                f"Preference optimization metrics mismatch:\n"
                f"  Missing from outcome model: {missing}\n"
                f"  Extra in outcome model: {extra}\n"
                "Ensure all metrics in PreferenceOptimizationConfig.objective "
                "are present in the main experiment and fit in the outcome model."
            )

        if preference_model_input_order != outcome_model_pref_metrics:
            raise UserInputError(
                "Metric ordering mismatch (will optimize wrong objectives!):\n"
                f"  Preference model expects: {preference_model_input_order}\n"
                f"  Outcome model produces:   {outcome_model_pref_metrics}"
            )


def validate_transformed_optimization_config(
    optimization_config: OptimizationConfig, outcomes: list[str]
) -> None:
    """Validate optimization config against generator fitted outcomes.

    Args:
        optimization_config: Config to validate.
        outcomes: List of metric names w/ valid generator fits.

    Raises if:
            1. In the modeling layer, absolute constraints are required, however,
               specifying relative constraints is supported. We handle this by
               either 1) transforming the observations to be relative (so that the
               constraints are absolute w.r.t relativized metrics), or 2)
               derelativizing the constraint. If relative constraints are found at
               this layer, we raise an error as likely either `Relativize` or
               `Derelativized` transforms were expected to be applied but were not.
            2. Optimization metrics are not present in generator fitted outcomes.
    """
    for c in optimization_config.outcome_constraints:
        if c.relative:
            raise ValueError(
                f"Passed {c} as a relative constraint. This likely indicates that "
                "either a `Relativize` or `Derelativize` transform was expected to be"
                " applied but was not."
            )
        if isinstance(c, ScalarizedOutcomeConstraint):
            for c_metric in c.metrics:
                if c_metric.signature not in outcomes:
                    raise DataRequiredError(
                        f"Scalarized constraint metric component "
                        f"{c.metric.signature} not found in fitted data."
                    )
        elif c.metric.signature not in outcomes:
            raise DataRequiredError(
                f"Outcome constraint metric {c.metric.signature} not found in fitted "
                "data."
            )
    obj_metric_signatures = [m.signature for m in optimization_config.objective.metrics]
    for obj_metric_signature in obj_metric_signatures:
        if obj_metric_signature not in outcomes:
            raise DataRequiredError(
                f"Objective metric {obj_metric_signature} not found in fitted data."
            )
