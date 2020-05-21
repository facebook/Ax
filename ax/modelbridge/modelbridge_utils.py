#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types import TBounds, TCandidateMetadata, TParamValue
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.typeutils import not_none


def extract_parameter_constraints(
    parameter_constraints: List[ParameterConstraint], param_names: List[str]
) -> Optional[TBounds]:
    """Extract parameter constraints."""
    if len(parameter_constraints) > 0:
        A = np.zeros((len(parameter_constraints), len(param_names)))
        b = np.zeros((len(parameter_constraints), 1))
        for i, c in enumerate(parameter_constraints):
            b[i, 0] = c.bound
            for name, val in c.constraint_dict.items():
                A[i, param_names.index(name)] = val
        linear_constraints: TBounds = (A, b)
    else:
        linear_constraints = None
    return linear_constraints


def get_bounds_and_task(
    search_space: SearchSpace, param_names: List[str]
) -> Tuple[List[Tuple[float, float]], List[int], Dict[int, TParamValue]]:
    """Extract box bounds from a search space in the usual Scipy format.
    Identify integer parameters as task features.
    """
    bounds: List[Tuple[float, float]] = []
    task_features: List[int] = []
    target_fidelities: Dict[int, TParamValue] = {}
    for i, p_name in enumerate(param_names):
        p = search_space.parameters[p_name]
        # Validation
        if not isinstance(p, RangeParameter):
            raise ValueError(f"{p} not RangeParameter")
        elif p.log_scale:
            raise ValueError(f"{p} is log scale")
        # Set value
        bounds.append((p.lower, p.upper))
        if p.parameter_type == ParameterType.INT:
            task_features.append(i)
        if p.is_fidelity:
            target_fidelities[i] = p.target_value

    return bounds, task_features, target_fidelities


def get_fixed_features(
    fixed_features: ObservationFeatures, param_names: List[str]
) -> Optional[Dict[int, float]]:
    """Reformat a set of fixed_features."""
    fixed_features_dict = {}
    for p_name, val in fixed_features.parameters.items():
        # These all need to be floats at this point.
        # pyre-ignore[6]: All float here.
        val_ = float(val)
        fixed_features_dict[param_names.index(p_name)] = val_
    fixed_features_dict = fixed_features_dict if len(fixed_features_dict) > 0 else None
    return fixed_features_dict


def pending_observations_as_array(
    pending_observations: Dict[str, List[ObservationFeatures]],
    outcome_names: List[str],
    param_names: List[str],
) -> Optional[List[np.ndarray]]:
    """Re-format pending observations.

    Args:
        pending_observations: List of raw numpy pending observations.
        outcome_names: List of outcome names.
        param_names: List fitted param names.

    Returns:
        Filtered pending observations data, by outcome and param names.
    """
    if len(pending_observations) == 0:
        pending_array: Optional[List[np.ndarray]] = None
    else:
        pending_array = [np.array([]) for _ in outcome_names]
        for metric_name, po_list in pending_observations.items():
            # It is possible that some metrics attached to the experiment should
            # not be included in pending features for a given model. For example,
            # if a model is fit to the initial data that is missing some of the
            # metrics on the experiment or if a model just should not be fit for
            # some of the metrics attached to the experiment, so metrics that
            # appear in pending_observations (drawn from an experiment) but not
            # in outcome_names (metrics, expected for the model) are filtered out.ß
            if metric_name not in outcome_names:
                continue
            pending_array[outcome_names.index(metric_name)] = np.array(
                [[po.parameters[p] for p in param_names] for po in po_list]
            )
    return pending_array


def parse_observation_features(
    X: np.ndarray,
    param_names: List[str],
    candidate_metadata: Optional[List[TCandidateMetadata]] = None,
) -> List[ObservationFeatures]:
    """Re-format raw model-generated candidates into ObservationFeatures.

    Args:
        param_names: List of param names.
        X: Raw np.ndarray of candidate values.
        candidate_metadata: Model's metadata for candidates it produced.

    Returns:
        List of candidates, represented as ObservationFeatures.
    """
    if candidate_metadata and len(candidate_metadata) != len(X):
        raise ValueError(
            "Observations metadata list provided is not of "
            "the same size as the number of candidates."
        )
    observation_features = []
    for i, x in enumerate(X):
        observation_features.append(
            ObservationFeatures(
                parameters=dict(zip(param_names, x)),
                metadata=candidate_metadata[i] if candidate_metadata else None,
            )
        )
    return observation_features


def transform_callback(
    param_names: List[str], transforms: MutableMapping[str, Transform]
) -> Callable[[np.ndarray], np.ndarray]:
    """A closure for performing the `round trip` transformations.

    The function round points by de-transforming points back into
    the original space (done by applying transforms in reverse), and then
    re-transforming them.
    This function is specifically for points which are formatted as numpy
    arrays. This function is passed to _model_gen.

    Args:
        param_names: Names of parameters to transform.
        transforms: Ordered set of transforms which were applied to the points.

    Returns:
        a function with for performing the roundtrip transform.
    """

    def _roundtrip_transform(x: np.ndarray) -> np.ndarray:
        """Inner function for performing aforementioned functionality.

        Args:
            x: points in the transformed space (e.g. all transforms have been applied
                to them)

        Returns:
            points in the transformed space, but rounded via the original space.
        """
        # apply reverse terminal transform to turn array to ObservationFeatures
        observation_features = [
            ObservationFeatures(
                parameters={p: float(x[i]) for i, p in enumerate(param_names)}
            )
        ]
        # reverse loop through the transforms and do untransform
        for t in reversed(transforms.values()):
            observation_features = t.untransform_observation_features(
                observation_features
            )
        # forward loop through the transforms and do transform
        for t in transforms.values():
            observation_features = t.transform_observation_features(
                observation_features
            )
        # parameters are guaranteed to be float compatible here, but pyre doesn't know
        new_x: List[float] = [
            # pyre-fixme[6]: Expected `Union[_SupportsIndex, bytearray, bytes, str,
            #  typing.SupportsFloat]` for 1st param but got `Union[None, bool, float,
            #  int, str]`.
            float(observation_features[0].parameters[p])
            for p in param_names
        ]
        # turn it back into an array
        return np.array(new_x)

    return _roundtrip_transform


def get_pending_observation_features(
    experiment: Experiment, include_failed_as_pending: bool = False
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """Computes a list of pending observation features (corresponding to arms that
    have been generated and deployed in the course of the experiment, but have not
    been completed with data).

    Args:
        experiment: Experiment, pending features on which we seek to compute.
        include_failed_as_pending: Whether to include failed trials as pending
            (for example, to avoid the model suggesting them again).

    Returns:
        An optional mapping from metric names to a list of observation features,
        pending for that metric (i.e. do not have evaluation data for that metric).
        If there are no pending features for any of the metrics, return is None.
    """
    pending_features = {}
    # Note that this assumes that if a metric appears in fetched data, the trial is
    # not pending for the metric. Where only the most recent data matters, this will
    # work, but may need to add logic to check previously added data objects, too.
    for trial_index, trial in experiment.trials.items():
        for metric_name in experiment.metrics:
            if metric_name not in pending_features:
                pending_features[metric_name] = []
            include_since_failed = include_failed_as_pending and trial.status.is_failed
            if isinstance(trial, BatchTrial):
                if (
                    (trial.status.is_deployed or include_since_failed)
                    and metric_name not in trial.fetch_data().df.metric_name.values
                    and trial.arms is not None
                ):
                    for arm in trial.arms:
                        not_none(pending_features.get(metric_name)).append(
                            ObservationFeatures.from_arm(
                                arm=arm, trial_index=np.int64(trial_index)
                            )
                        )
            if isinstance(trial, Trial):
                if (
                    (trial.status.is_deployed or include_since_failed)
                    and metric_name not in trial.fetch_data().df.metric_name.values
                    and trial.arm is not None
                ):
                    not_none(pending_features.get(metric_name)).append(
                        ObservationFeatures.from_arm(
                            arm=not_none(trial.arm), trial_index=np.int64(trial_index)
                        )
                    )
    return pending_features if any(x for x in pending_features.values()) else None
