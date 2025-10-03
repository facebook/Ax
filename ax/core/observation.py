#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
from copy import deepcopy
from logging import Logger

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.core.arm import Arm
from ax.core.types import TCandidateMetadata, TParameterization
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class ObservationFeatures(Base):
    """The features of an observation.

    These include both the arm parameters and the features of the
    observation found in the Data object: trial index, start_time,
    and end_time. This object is meant to contain everything needed to
    represent this observation in a model feature space. It is essentially a
    row of Data joined with the arm parameters.

    An ObservationFeatures object would typically have a corresponding
    ObservationData object that provides the observed outcomes.

    Attributes:
        parameters: arm parameters
        trial_index: trial index
        start_time: batch start time
        end_time: batch end time

    """

    def __init__(
        self,
        parameters: TParameterization,
        trial_index: int | None = None,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        metadata: TCandidateMetadata = None,
    ) -> None:
        self.parameters = parameters
        self.trial_index = trial_index
        self.start_time = start_time
        self.end_time = end_time
        self.metadata = metadata

    @staticmethod
    def from_arm(
        arm: Arm,
        trial_index: int | None = None,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        metadata: TCandidateMetadata = None,
    ) -> ObservationFeatures:
        """Convert a Arm to an ObservationFeatures, including additional
        data as specified.
        """
        return ObservationFeatures(
            # NOTE: Arm.parameters makes a copy of the original dict, so any
            # modifications to the parameters dict will not be reflected in
            # the original Arm parameters.
            parameters=arm.parameters,
            trial_index=trial_index,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata,
        )

    def update_features(self, new_features: ObservationFeatures) -> ObservationFeatures:
        """Updates the existing ObservationFeatures with the fields of the the input.

        Adds all of the new parameters to the existing parameters and overwrites
        any other fields that are not None on the new input features."""
        self.parameters.update(new_features.parameters)
        if new_features.trial_index is not None:
            self.trial_index = new_features.trial_index
        if new_features.start_time is not None:
            self.start_time = new_features.start_time
        if new_features.end_time is not None:
            self.end_time = new_features.end_time
        return self

    def clone(
        self, replace_parameters: TParameterization | None = None
    ) -> ObservationFeatures:
        """Make a copy of these ``ObservationFeatures``.

        Args:
            replace_parameters: An optimal parameterization, to which to set the
                parameters of the cloned ``ObservationFeatures``. Useful when
                transforming observation features in a way that requires a
                change to parameterization –– for example, while casting it to
                a hierarchical search space.
        """
        parameters = (
            self.parameters if replace_parameters is None else replace_parameters
        )
        return ObservationFeatures(
            parameters=parameters.copy(),
            trial_index=self.trial_index,
            start_time=self.start_time,
            end_time=self.end_time,
            metadata=deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        strs = []
        for attr in ["trial_index", "start_time", "end_time"]:
            if getattr(self, attr) is not None:
                strs.append(f", {attr}={getattr(self, attr)}")
        repr_str = "ObservationFeatures(parameters={parameters}".format(
            parameters=self.parameters
        )
        repr_str += "".join(strs) + ")"
        return repr_str

    def __hash__(self) -> int:
        parameters = self.parameters.copy()
        for k, v in parameters.items():
            if type(v) is np.int64:
                parameters[k] = int(v)
            elif type(v) is np.float32:
                parameters[k] = float(v)
        return hash(
            (
                json.dumps(parameters, sort_keys=True),
                self.trial_index,
                self.start_time,
                self.end_time,
            )
        )


class ObservationData(Base):
    """Outcomes observed at a point.

    The "point" corresponding to this ObservationData would be an
    ObservationFeatures object.

    Attributes:
        metric_signatures: A list of k metric signatures that were observed
        means: a k-array of observed means
        covariance: a (k x k) array of observed covariances
    """

    def __init__(
        self,
        metric_signatures: list[str],
        means: npt.NDArray,
        covariance: npt.NDArray,
    ) -> None:
        k = len(metric_signatures)
        if means.shape != (k,):
            raise ValueError(f"Shape of means should be {(k,)}, is {(means.shape)}.")
        if covariance.shape != (k, k):
            raise ValueError(
                "Shape of covariance should be {}, is {}.".format(
                    (k, k), (covariance.shape)
                )
            )
        self.metric_signatures = metric_signatures
        self.means = means
        self.covariance = covariance

    @property
    def means_dict(self) -> dict[str, float]:
        """Extract means from this observation data as mapping from metric name to
        mean.
        """
        return dict(zip(self.metric_signatures, self.means))

    @property
    def covariance_matrix(self) -> dict[str, dict[str, float]]:
        """Extract covariance matric from this observation data as mapping from
        metric name (m1) to mapping of another metric name (m2) to the covariance
        of the two metrics (m1 and m2).
        """
        return {
            m1: {
                m2: float(self.covariance[idx1][idx2])
                for idx2, m2 in enumerate(self.metric_signatures)
            }
            for idx1, m1 in enumerate(self.metric_signatures)
        }

    def __repr__(self) -> str:
        return (
            "ObservationData(metric_signatures={mn}, means={m}, covariance={c})".format(
                mn=self.metric_signatures, m=self.means, c=self.covariance
            )
        )


class Observation(Base):
    """Represents an observation.

    A set of features (ObservationFeatures) and corresponding measurements
    (ObservationData). Optionally, an arm name associated with the features.

    Attributes:
        features (ObservationFeatures)
        data (ObservationData)
        arm_name (Optional[str])
    """

    def __init__(
        self,
        features: ObservationFeatures,
        data: ObservationData,
        arm_name: str | None = None,
    ) -> None:
        self.features = features
        self.data = data
        self.arm_name = arm_name

    def __repr__(self) -> str:
        return (
            "Observation(\n"
            f"    features={self.features},\n"
            f"    data={self.data},\n"
            f"    arm_name='{self.arm_name}',\n"
            ")"
        )
