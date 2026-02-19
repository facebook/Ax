# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import TYPE_CHECKING

from ax.core.data import Data
from ax.utils.common.base import SortableBase

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class AuxiliaryExperiment(SortableBase):
    """Class for defining an auxiliary experiment."""

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        is_active: bool = True,
        data: Data | None = None,
    ) -> None:
        """
        Lightweight container of an experiment, and its data,
        that will be used as auxiliary information for another experiment.
        Args:
            experiment: The Ax experiment with the auxiliary data.
            is_active: Whether the auxiliary experiment relation to the target
                experiment is currently active.
            data: Auxiliary data.
        """
        self.experiment = experiment
        self.data: Data = data or experiment.lookup_data()
        self.is_active = is_active

    def _unique_id(self) -> str:
        # While there can be multiple `AuxiliarySource`-s made from the same
        # experiment (and thus sharing the experiment name), the uniqueness
        # here is only needed w.r.t. parent object ("main experiment", for which
        # this will be an auxiliary source for).
        return self.experiment.name


@unique
class AuxiliaryExperimentPurpose(StrEnum):
    # BOPE Aux Experiment Usage pattern:
    # 1. Run the exploratory batch for the main / BO experiment.
    # 2. Use the BO experiment as the auxiliary experiment for the PE experiment
    #    to construct the outcome model to estimate metric movements during
    #    preference exploration.
    # 3. Use the PE experiment as the auxiliary experiment to get the learned
    #    objective continue iterating with the BO experiment.
    # [BOPE] The BO experiment for which preference exploration is being done.
    BO_EXPERIMENT = "bo_experiment"
    # [BOPE] The preference exploration experiment itself.
    PE_EXPERIMENT = "pe_experiment"
    # [BOTL] The transferable/source experiment for transfer learning.
    TRANSFERABLE_EXPERIMENT = "transferable_experiment"


@dataclass
class AuxiliaryExperimentMetadata(ABC):
    """Abstract base class for metadata associated with auxiliary experiments.

    Subclasses define specific metadata for different auxiliary experiment purposes.
    Typically, this is used to store information about the experiment poperty relevant
    to the auxiliary experiment purpose (e.g. parameters for transfer learning).
    """

    pass


@dataclass
class PreferenceExplorationMetadata(AuxiliaryExperimentMetadata):
    """Metadata for preference exploration auxiliary experiments."""

    overlap_metrics: list[str]


@dataclass
class TransferLearningMetadata(AuxiliaryExperimentMetadata):
    """Metadata for transfer learning auxiliary experiments."""

    overlap_parameters: list[str]


@dataclass
class AuxiliaryExperimentValidation:
    """Result of validating a source experiment as an auxiliary experiment.

    Whether an experiment can be used as a source experiment for a given purpose.

    Attributes:
        is_valid: Whether the source experiment is valid for the given purpose.
        invalid_reason: Human-readable explanation if validation failed, None otherwise.
        metadata: Purpose-specific metadata computed during validation.
    """

    is_valid: bool
    invalid_reason: str | None
    metadata: AuxiliaryExperimentMetadata | None = None
