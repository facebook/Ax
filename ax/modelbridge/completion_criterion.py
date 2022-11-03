# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

from ax.core.base_trial import TrialStatus

from ax.core.experiment import Experiment
from ax.utils.common.base import Base
from ax.utils.common.serialization import SerializationMixin


class CompletionCriterion(Base, SerializationMixin):
    """
    Simple class to descibe a condition which must be met for a GenerationStraytegy
    to move to its next GenerationStep.
    """

    def __init__(self) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def is_met(self, experiment: Experiment) -> bool:
        pass  # pragma: no cover


class MinimumTrialsInStatus(CompletionCriterion):
    def __init__(self, status: TrialStatus, threshold: int) -> None:
        self.status = status
        self.threshold = threshold

    def is_met(self, experiment: Experiment) -> bool:
        return len(experiment.trial_indices_by_status[self.status]) >= self.threshold


class MinimumPreferenceOccurances(CompletionCriterion):
    def __init__(self, metric_name: str, threshold: int) -> None:
        self.metric_name = metric_name
        self.threshold = threshold

    def is_met(self, experiment: Experiment) -> bool:
        data = experiment.fetch_data(metrics=[experiment.metrics[self.metric_name]])

        count_no = (data.df["mean"] == 0).sum()
        count_yes = (data.df["mean"] != 0).sum()

        return count_no >= self.threshold and count_yes >= self.threshold
