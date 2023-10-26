#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import extract_arm_predictions
from ax.core.metric import Metric
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    separate_observations,
)
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    ScalarizedOutcomeConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TModelPredictArm
from ax.exceptions.core import DataRequiredError, UnsupportedError
from ax.modelbridge.base import gen_arms, GenResults, ModelBridge
from ax.modelbridge.modelbridge_utils import (
    array_to_observation_data,
    extract_objective_thresholds,
    extract_objective_weights,
    extract_outcome_constraints,
    extract_parameter_constraints,
    extract_risk_measure,
    extract_search_space_digest,
    get_fixed_features,
    observation_data_to_array,
    observation_features_to_array,
    parse_observation_features,
    pending_observations_as_array_list,
    process_contextual_datesets,
    SearchSpaceDigest,
    transform_callback,
    validate_and_apply_final_transform,
)
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cast import Cast
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch_base import TorchModel, TorchOptConfig
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from torch import Tensor

logger: Logger = get_logger(__name__)

FIT_MODEL_ERROR = "Model must be fit before {action}."


# pyre-fixme [13]: Attributes are never initialized.
class TorchModelBridge(ModelBridge):
    """A model bridge for using torch-based models.

    Specifies an interface that is implemented by TorchModel. In particular,
    model should have methods fit, predict, and gen. See TorchModel for the
    API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the model.
    """

    model: Optional[TorchModel] = None
    outcomes: List[str]
    parameters: List[str]
    _default_model_gen_options: TConfig
    _last_observations: Optional[List[Observation]] = None

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: TorchModel,
        transforms: List[Type[Transform]],
        transform_configs: Optional[Dict[str, TConfig]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        torch_device: Optional[torch.device] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        fit_abandoned: bool = False,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        self.dtype: torch.dtype = torch.double if torch_dtype is None else torch_dtype
        self.device = torch_device
        # pyre-ignore [4]: Attribute `_default_model_gen_options` of class
        # `TorchModelBridge` must have a type that does not contain `Any`.
        self._default_model_gen_options = default_model_gen_options or {}

        # Handle init for multi-objective optimization.
        self.is_moo_problem: bool = False
        if optimization_config or (experiment and experiment.optimization_config):
            optimization_config = not_none(
                optimization_config or experiment.optimization_config
            )
            self.is_moo_problem = optimization_config.is_moo_problem

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=model,
            transforms=transforms,
            transform_configs=transform_configs,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
            fit_out_of_design=fit_out_of_design,
            fit_abandoned=fit_abandoned,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
        )

    def feature_importances(self, metric_name: str) -> Dict[str, float]:
        importances_tensor = not_none(self.model).feature_importances()
        importances_dict = dict(zip(self.outcomes, importances_tensor))
        importances_arr = importances_dict[metric_name].flatten()
        return dict(zip(self.parameters, importances_arr))

    def infer_objective_thresholds(
        self,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[ObjectiveThreshold]:
        """Infer objective thresholds.

        This method is only applicable for Multi-Objective optimization problems.

        This method uses the model-estimated Pareto frontier over the in-sample points
        to infer absolute (not relativized) objective thresholds.

        This uses a heuristic that sets the objective threshold to be a scaled nadir
        point, where the nadir point is scaled back based on the range of each
        objective across the current in-sample Pareto frontier.
        """
        if not self.is_moo_problem:
            raise UnsupportedError(
                "Objective thresholds are only supported for multi-objective "
                "optimization."
            )

        search_space = (search_space or self._model_space).clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            fixed_features=fixed_features,
        )
        # Get transformed args from TorchModelbridge.
        search_space_digest, torch_opt_config = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            fixed_features=base_gen_args.fixed_features,
            pending_observations={},
            optimization_config=base_gen_args.optimization_config,
        )
        if torch_opt_config.risk_measure is not None:
            raise UnsupportedError(
                "`infer_objective_thresholds` does not support risk measures."
            )
        # Infer objective thresholds.
        if isinstance(self.model, MultiObjectiveBotorchModel):
            model = self.model.model
            Xs = self.model.Xs
        elif isinstance(self.model, BoTorchModel):
            model = self.model.surrogate.model
            Xs = self.model.surrogate.Xs
        else:
            raise UnsupportedError(
                "Model must be a MultiObjectiveBotorchModel or an appropriate Modular "
                "Botorch Model to infer_objective_thresholds. Found "
                f"{type(self.model)}."
            )

        obj_thresholds = infer_objective_thresholds(
            model=not_none(model),
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            outcome_constraints=torch_opt_config.outcome_constraints,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            Xs=Xs,
        )

        return self._untransform_objective_thresholds(
            objective_thresholds=obj_thresholds,
            objective_weights=torch_opt_config.objective_weights,
            bounds=search_space_digest.bounds,
            # we should never be in a situation where we call this without there
            # being an optimization config involved.
            opt_config_metrics=not_none(torch_opt_config.opt_config_metrics),
            fixed_features=torch_opt_config.fixed_features,
        )

    def model_best_point(
        self,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[Tuple[Arm, Optional[TModelPredictArm]]]:
        # Get modifiable versions
        if search_space is None:
            search_space = self._model_space
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
            model_gen_options=None,
            optimization_config=base_gen_args.optimization_config,
        )
        try:
            xbest = not_none(self.model).best_point(
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
        self, array_func: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[Tensor], Tensor]:
        tensor_func: Callable[[Tensor], Tensor] = lambda x: (
            self._array_to_tensor(array_func(x.detach().cpu().clone().numpy()))
        )
        return tensor_func

    def _array_list_to_tensors(self, arrays: List[np.ndarray]) -> List[Tensor]:
        return [self._array_to_tensor(x) for x in arrays]

    def _array_to_tensor(self, array: Union[np.ndarray, List[float]]) -> Tensor:
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def _convert_observations(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
        outcomes: List[str],
        parameters: List[str],
        search_space_digest: Optional[SearchSpaceDigest],
    ) -> Tuple[
        List[Optional[SupervisedDataset]], Optional[List[List[TCandidateMetadata]]]
    ]:
        """Converts observations to a dictionary of `Dataset` containers and (optional)
        candidate metadata.

        Args:
            observation_data: A list of `ObservationData` from which to extract
                mean `Y` and variance `Yvar` observations. Must correspond 1:1 to
                the `observation_features`.
            observation_features: A list of `ObservationFeatures` from which to extract
                parameter values. Must correspond 1:1 to the `observation_data`.
            outcomes: The names of the outcomes to extract observations for.
            parameters: The names of the parameters to extract. Any observation features
                that are not included in `parameters` will be ignored.
            search_space_digest: An optional `SearchSpaceDigest` containing information
                about the search space. This is used to convert datasets into a
                `MultiTaskDataset` where applicable.

        Returns:
            - A list of `Dataset` objects.
                For non-contextual experiment, each element in the list corresponds to
                one outcome and the list is sorted based on the outcomes.
                For contextual experiment, each element in the list can correspond to
                either one outcome (overall) or multiple outcomes
                (context-level outcome).
                The list is sorted based on the outcomes only if there is no
                contextl-level outcome; when there is a mixed of context-level and
                overall outcomes, the ordering will be handled by the downstream
                GP model.
                If the outcome does not have any observations, then the corresponding
                element in the list will be `None`.
            - An optional list of lists of candidate metadata. Each inner list
                corresponds to one outcome. Each element in the inner list corresponds
                to one observation.
        """
        (
            Xs,
            Ys,
            Yvars,
            candidate_metadata_dict,
            any_candidate_metadata_is_not_none,
        ) = self._extract_observation_data(
            observation_data, observation_features, parameters
        )

        datasets: List[Optional[SupervisedDataset]] = []
        candidate_metadata = []
        for outcome in outcomes:
            if outcome not in Xs:
                # This may happen when we update the data of only some metrics
                datasets.append(None)
                candidate_metadata.append(None)
                continue
            X = torch.stack(Xs[outcome], dim=0)
            Y = torch.tensor(
                Ys[outcome], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            Yvar = torch.tensor(
                Yvars[outcome], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            if Yvar.isnan().all():
                Yvar = None
            else:
                Yvar = Yvar.clamp_min(1e-6)
            dataset = SupervisedDataset(
                X=X,
                Y=Y,
                Yvar=Yvar,
                feature_names=parameters,
                outcome_names=[outcome],
            )
            datasets.append(dataset)
            candidate_metadata.append(candidate_metadata_dict[outcome])
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
                    "This is often accomplished using a TaskEncode transform. "
                    "Check that the model is using the correct set of transforms. "
                    f"Got {task_choices=}."
                )
            datasets = [
                MultiTaskDataset.from_joint_dataset(
                    dataset=dataset,
                    task_feature_index=task_features[0],
                    target_task_value=int(target_task_value),
                )
                if dataset is not None
                else None
                for dataset in datasets
            ]
        # check whether is a `parameter_decomposition` experiment property to
        # decide whether it is a contextual experiment
        if self._experiment_properties.get("parameter_decomposition", None) is not None:
            full_datasets = datasets
            # handle the case that the dataset can be None
            datasets = [dataset for dataset in full_datasets if dataset is None]
            # convert to a list of ContextualDateset for contextual experiments
            datasets.extend(
                process_contextual_datesets(
                    datasets=[
                        dataset for dataset in full_datasets if dataset is not None
                    ],
                    outcomes=outcomes,
                    parameter_decomposition=self._experiment_properties[
                        "parameter_decomposition"
                    ],
                    metric_decomposition=self._experiment_properties.get(
                        "metric_decomposition", None
                    ),
                )
            )

        if not any_candidate_metadata_is_not_none:
            return datasets, None

        return datasets, candidate_metadata

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: List[Observation],
        cv_test_points: List[ObservationFeatures],
        parameters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data.
        """
        if self.model is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_cross_validate"))
        datasets, candidate_metadata, search_space_digest = self._get_fit_args(
            search_space=search_space,
            observations=cv_training_data,
            parameters=parameters,
            update_outcomes_and_parameters=False,
        )
        for outcome, dataset in zip(self.outcomes, datasets):
            if dataset is None:
                raise UnsupportedError(
                    f"{self.__class__._cross_validate} requires observations "
                    f"for all outcomes, but no observations for {outcome}"
                )
        if parameters is None:
            parameters = self.parameters
        X_test = torch.tensor(
            [[obsf.parameters[p] for p in parameters] for obsf in cv_test_points],
            dtype=self.dtype,
            device=self.device,
        )
        # Use the model to do the cross validation
        f_test, cov_test = not_none(self.model).cross_validate(
            datasets=[not_none(dataset) for dataset in datasets],
            metric_names=self.outcomes,
            X_test=torch.as_tensor(X_test, dtype=self.dtype, device=self.device),
            search_space_digest=search_space_digest,
            **kwargs,
        )
        # Convert array back to ObservationData
        return array_to_observation_data(
            f=f_test.detach().cpu().clone().numpy(),
            cov=cov_test.detach().cpu().clone().numpy(),
            outcomes=self.outcomes,
        )

    def evaluate_acquisition_function(
        self,
        observation_features: Union[
            List[ObservationFeatures], List[List[ObservationFeatures]]
        ],
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
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
            search_space: Search space for fitting the model.
            optimization_config: Optimization config defining how to optimize
                the model.
            pending_observations: A map from metric name to pending observations for
                that metric.
            fixed_features: An ObservationFeatures object containing any features that
                should be fixed at specified values during generation.
            acq_options: Keyword arguments used to contruct the acquisition function.

        Returns:
            A list of acquisition function values, in the same order as the
            input observation features.
        """
        search_space = search_space or self._model_space
        optimization_config = optimization_config or self._optimization_config
        if optimization_config is None:
            raise ValueError(
                "The `optimization_config` must be specified either while initializing "
                "the ModelBridge or to the `evaluate_acquisition_function` call."
            )
        # pyre-ignore Incompatible parameter type [9]
        obs_feats: List[List[ObservationFeatures]] = deepcopy(observation_features)
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
            optimization_config=not_none(base_gen_args.optimization_config),
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            acq_options=acq_options,
        )

    def _evaluate_acquisition_function(
        self,
        observation_features: List[List[ObservationFeatures]],
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        if self.model is None:
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
        evals = not_none(self.model).evaluate_acquisition_function(
            X=self._array_to_tensor(X),
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            acq_options=acq_options,
        )
        return evals.tolist()

    def _get_fit_args(
        self,
        search_space: SearchSpace,
        observations: List[Observation],
        parameters: Optional[List[str]],
        update_outcomes_and_parameters: bool,
    ) -> Tuple[
        List[Optional[SupervisedDataset]],
        Optional[List[List[TCandidateMetadata]]],
        SearchSpaceDigest,
    ]:
        """Helper for consolidating some common argument processing between
        fit and cross validate methods. Extracts datasets and candidate metadate
        from observations and the search space digest from the search space.

        Args:
            search_space: A transformed search space for fitting the model.
            observations: The observations to fit the model with. These should
                also be transformed.
            parameters: Names of parameters to be used in the model. Defaults to
                all parameters in the search space.
            update_outcomes_and_parameters: Whether to update `self.outcomes` with
                all outcomes found in the observations and `self.parameters` with
                all parameters in the search space. Typically only used in `_fit`.

        Returns:
            The datasets & metadata extracted from the observations and the
            search space digest.
        """
        self._last_observations = observations
        if update_outcomes_and_parameters:
            self.parameters = list(search_space.parameters.keys())
        if parameters is None:
            parameters = self.parameters
        all_metric_names: Set[str] = set()
        observation_features, observation_data = separate_observations(observations)
        # Only update outcomes if fitting a model on tracking metrics. Otherwise,
        # we will only fit models to the outcomes that are extracted from optimization
        # config in ModelBridge.__init__.
        if update_outcomes_and_parameters and self._fit_tracking_metrics:
            for od in observation_data:
                all_metric_names.update(od.metric_names)
            self.outcomes = sorted(all_metric_names)  # Deterministic order
        # Get all relevant information on the parameters
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        # Convert observations to datasets
        datasets, candidate_metadata = self._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=parameters,
            search_space_digest=search_space_digest,
        )
        return datasets, candidate_metadata, search_space_digest

    def _fit(
        self,
        model: TorchModel,
        search_space: SearchSpace,
        observations: List[Observation],
        parameters: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        if self.model is not None and observations == self._last_observations:
            logger.info(
                "The observations are identical to the last set of observations "
                "used to fit the model. Skipping model fitting."
            )
            return
        datasets, candidate_metadata, search_space_digest = self._get_fit_args(
            search_space=search_space,
            observations=observations,
            parameters=parameters,
            update_outcomes_and_parameters=True,
        )
        # Fit
        self.model = model
        self.model.fit(
            # datasets are guaranteed to have all outcomes here by construction
            datasets=[not_none(dataset) for dataset in datasets],
            metric_names=self.outcomes,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
            **kwargs,
        )

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: Optional[ObservationFeatures],
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> GenResults:
        """Generate new candidates according to search_space and
        optimization_config.

        The outcome constraints should be transformed to no longer be relative.
        """
        if self.model is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))

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
        # TODO(ehotaj): For some reason, we're getting models which do not support MOO
        # even when optimization_config has multiple objectives, so we can't use
        # self.is_moo_problem here.
        is_moo_problem = self.is_moo_problem and isinstance(
            self.model, (BoTorchModel, MultiObjectiveBotorchModel)
        )
        gen_results = not_none(self.model).gen(
            n=n,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
        )

        gen_metadata = gen_results.gen_metadata
        if is_moo_problem and "objective_thresholds" in gen_metadata:
            # If objective_thresholds are supplied by the user, then the transformed
            # user-specified objective thresholds are in gen_metadata. Otherwise,
            # if using a hypervolume based acquisition function, then
            # the inferred objective thresholds are in gen_metadata.
            opt_config_metrics = (
                torch_opt_config.opt_config_metrics
                or not_none(self._optimization_config).metrics
            )
            gen_metadata[
                "objective_thresholds"
            ] = self._untransform_objective_thresholds(
                objective_thresholds=gen_metadata["objective_thresholds"],
                objective_weights=torch_opt_config.objective_weights,
                bounds=search_space_digest.bounds,
                opt_config_metrics=opt_config_metrics,
                fixed_features=torch_opt_config.fixed_features,
            )

        # Transform array to observations
        observation_features = self._array_to_observation_features(
            X=gen_results.points.detach().cpu().clone().numpy(),
            candidate_metadata=gen_results.candidate_metadata,
        )
        try:
            xbest = not_none(self.model).best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        except NotImplementedError:
            xbest = None

        best_obsf = None
        if xbest is not None:
            best_obsf = ObservationFeatures(
                parameters={p: float(xbest[i]) for i, p in enumerate(self.parameters)}
            )

        return GenResults(
            observation_features=observation_features,
            weights=gen_results.weights.tolist(),
            best_observation_features=best_obsf,
            gen_metadata=gen_metadata,
        )

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        if not self.model:
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        # Convert observation features to array
        X = observation_features_to_array(self.parameters, observation_features)
        f, cov = not_none(self.model).predict(X=self._array_to_tensor(X))
        f = f.detach().cpu().clone().numpy()
        cov = cov.detach().cpu().clone().numpy()
        if f.shape[-2] != X.shape[-2]:
            raise NotImplementedError(
                "Expected same number of predictions as the number of inputs but got "
                f"predictions of shape {f.shape} for inputs of shape {X.shape}. "
                "This was likely due to the use of one-to-many input transforms -- "
                "typically used for robust optimization -- which are not supported in"
                "TorchModelBridge.predict."
            )
        # Convert resulting arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _array_to_observation_features(
        self, X: np.ndarray, candidate_metadata: Optional[List[TCandidateMetadata]]
    ) -> List[ObservationFeatures]:
        return parse_observation_features(
            X=X, param_names=self.parameters, candidate_metadata=candidate_metadata
        )

    def _transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> Tensor:
        """Apply terminal transform to given observation features and return result
        as an N x D array of points.
        """
        try:
            tobfs = np.array(
                [
                    # pyre-ignore[6]: Except statement below should catch wrongly
                    # typed parameters.
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
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: Optional[ObservationFeatures],
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> Tuple[SearchSpaceDigest, TorchOptConfig]:
        # Validation
        if not self.parameters:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))
        # Extract search space info
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        if optimization_config is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an OptimizationConfig "
                "to be specified"
            )

        validate_optimization_config(optimization_config, self.outcomes)
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=self.outcomes,
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
            opt_config_metrics = optimization_config.metrics
        else:
            objective_thresholds, opt_config_metrics = None, None

        pending_array = pending_observations_as_array_list(
            pending_observations, self.outcomes, self.parameters
        )
        obj_w, out_c, lin_c, pend_o, obj_t = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_array,
            objective_thresholds=objective_thresholds,
            final_transform=self._array_to_tensor,
        )
        rounding_func = self._array_callable_to_tensor_callable(
            transform_callback(self.parameters, self.transforms)
        )
        risk_measure = (
            optimization_config.risk_measure
            if optimization_config is not None
            else None
        )
        if risk_measure is not None:
            if not not_none(self.model)._supports_robust_optimization:
                raise UnsupportedError(
                    f"{self.model.__class__.__name__} does not support robust "
                    "optimization. Consider using modular BoTorch model instead."
                )
            else:
                risk_measure = extract_risk_measure(risk_measure=risk_measure)
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
            is_moo=optimization_config.is_moo_problem,
            risk_measure=risk_measure,
        )
        return search_space_digest, torch_opt_config

    def _transform_observations(
        self, observations: List[Observation]
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
        X = self._transform_observation_features(observation_features)
        return X, self._array_to_tensor(mean), self._array_to_tensor(cov)

    def _untransform_objective_thresholds(
        self,
        objective_thresholds: Tensor,
        objective_weights: Tensor,
        bounds: List[Tuple[Union[int, float], Union[int, float]]],
        opt_config_metrics: Dict[str, Metric],
        fixed_features: Optional[Dict[int, float]],
    ) -> List[ObjectiveThreshold]:
        thresholds_np = objective_thresholds.cpu().numpy()
        idxs = objective_weights.nonzero().view(-1).tolist()

        # Create transformed ObjectiveThresholds from numpy thresholds.
        thresholds = []
        for idx in idxs:
            sign = torch.sign(objective_weights[idx])
            thresholds.append(
                ObjectiveThreshold(
                    metric=opt_config_metrics[self.outcomes[idx]],
                    bound=thresholds_np[idx],
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

    def _validate_observation_data(
        self, observation_data: List[ObservationData]
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError(
                "Torch models cannot be fit without observation data. Possible "
                "reasons include empty data being passed to the model's constructor "
                "or data being excluded because it is out-of-design. Try setting "
                "`fit_out_of_design`=True during construction to fix the latter."
            )

    def _extract_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
        parameters: List[str],
    ) -> Tuple[
        Dict[str, List[Tensor]],
        Dict[str, List[Tensor]],
        Dict[str, List[Tensor]],
        Dict[str, List[TCandidateMetadata]],
        bool,
    ]:
        """Extract observation features & data into tensors and metadata.

        Args:
            observation_data: A list of `ObservationData` from which to extract
                mean `Y` and variance `Yvar` observations. Must correspond 1:1 to
                the `observation_features`.
            observation_features: A list of `ObservationFeatures` from which to extract
                parameter values. Must correspond 1:1 to the `observation_data`.
            parameters: The names of the parameters to extract. Any observation features
                that are not included in `parameters` will be ignored.

        Returns:
            - A dictionary mapping metric names to lists of corresponding feature
                tensors `X`.
            - A dictionary mapping metric names to lists of corresponding mean
                observation tensors `Y`.
            - A dictionary mapping metric names to lists of corresponding variance
                observation tensors `Yvar`.
            - A dictionary mapping metric names to lists of corresponding metadata.
            - A boolean denoting whether any candidate metadata is not none.
        """
        Xs: Dict[str, List[Tensor]] = defaultdict(list)
        Ys: Dict[str, List[Tensor]] = defaultdict(list)
        Yvars: Dict[str, List[Tensor]] = defaultdict(list)
        candidate_metadata_dict: Dict[str, List[TCandidateMetadata]] = defaultdict(list)
        any_candidate_metadata_is_not_none = False

        for obsd, obsf in zip(observation_data, observation_features):
            try:
                x = torch.tensor(
                    [obsf.parameters[p] for p in parameters],
                    dtype=self.dtype,
                    device=self.device,
                )
            except (KeyError, TypeError):
                raise ValueError("Out of design points cannot be converted.")
            for metric_name, mean, var in zip(
                obsd.metric_names, obsd.means, obsd.covariance.diagonal()
            ):
                Xs[metric_name].append(x)
                Ys[metric_name].append(mean)
                Yvars[metric_name].append(var)
                if obsf.metadata is not None:
                    any_candidate_metadata_is_not_none = True
                candidate_metadata_dict[metric_name].append(obsf.metadata)

        return (
            Xs,
            Ys,
            Yvars,
            candidate_metadata_dict,
            any_candidate_metadata_is_not_none,
        )


def validate_optimization_config(
    optimization_config: OptimizationConfig, outcomes: List[str]
) -> None:
    """Validate optimization config against model fitted outcomes.

    Args:
        optimization_config: Config to validate.
        outcomes: List of metric names w/ valid model fits.

    Raises:
        ValueError if:
            1. Relative constraints are found
            2. Optimization metrics are not present in model fitted outcomes.
    """
    for c in optimization_config.outcome_constraints:
        if c.relative:
            raise ValueError(f"{c} is a relative constraint.")
        if isinstance(c, ScalarizedOutcomeConstraint):
            for c_metric in c.metrics:
                if c_metric.name not in outcomes:
                    raise DataRequiredError(
                        f"Scalarized constraint metric component {c.metric.name} "
                        + "not found in fitted data."
                    )
        elif c.metric.name not in outcomes:
            raise DataRequiredError(
                f"Outcome constraint metric {c.metric.name} not found in fitted data."
            )
    obj_metric_names = [m.name for m in optimization_config.objective.metrics]
    for obj_metric_name in obj_metric_names:
        if obj_metric_name not in outcomes:
            raise DataRequiredError(
                f"Objective metric {obj_metric_name} not found in fitted data."
            )
