# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Optional

from ax.core.base_trial import TrialStatus

from ax.core.experiment import Experiment
from ax.utils.common.base import Base
from ax.utils.common.serialization import SerializationMixin


class TransitionCriterion(Base, SerializationMixin):
    """
    Simple class to descibe a condition which must be met for a GenerationStrategy
    to move to its next GenerationNode.
    """

    # TODO: @mgarrard add `transition_to` attribute to define the next node
    def __init__(self) -> None:
        pass

    @abstractmethod
    def is_met(self, experiment: Experiment) -> bool:
        pass


class MinimumTrialsInStatus(TransitionCriterion):
    """
    Simple class to decide if the number of trials of a given status in the
    GenerationStrategy experiment has reached a certain threshold.
    """

    def __init__(self, status: TrialStatus, threshold: int) -> None:
        self.status = status
        self.threshold = threshold

    def is_met(self, experiment: Experiment) -> bool:
        return len(experiment.trial_indices_by_status[self.status]) >= self.threshold


class MaxTrials(TransitionCriterion):
    """
    Simple class to enforce a maximum threshold for the number of trials generated
    by a specific GenerationNode.

    Args:
        threshold: the designated maximum number of trials
        enforce: whether or not to enforce the max trial constraint
        only_in_status: optional argument for specifying only checking trials with
            this status. If not specified, all trial statuses are counted.
    """

    def __init__(
        self,
        threshold: int,
        enforce: bool,
        only_in_status: Optional[TrialStatus] = None,
    ) -> None:
        self.threshold = threshold
        self.enforce = enforce
        # Optional argument for specifying only checking trials with this status
        self.only_in_status = only_in_status

    def is_met(self, experiment: Experiment) -> bool:
        if self.enforce:
            if self.only_in_status is not None:
                return (
                    len(experiment.trial_indices_by_status[self.only_in_status])
                    >= self.threshold
                )
            return experiment.num_trials >= self.threshold
        return True


class MinimumPreferenceOccurances(TransitionCriterion):
    """
    In a preference Experiment (i.e. Metric values may either be zero for No and
    nonzero for Yes) do not transition until a minimum number of both Yes and No
    responses have been received.
    """

    def __init__(self, metric_name: str, threshold: int) -> None:
        self.metric_name = metric_name
        self.threshold = threshold

    def is_met(self, experiment: Experiment) -> bool:
        # TODO: @mgarrard replace fetch_data with lookup_data
        data = experiment.fetch_data(metrics=[experiment.metrics[self.metric_name]])

        count_no = (data.df["mean"] == 0).sum()
        count_yes = (data.df["mean"] != 0).sum()

        return count_no >= self.threshold and count_yes >= self.threshold
