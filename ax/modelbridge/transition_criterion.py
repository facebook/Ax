# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import abstractmethod
from logging import Logger
from typing import List, Optional, Set

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.exceptions.core import DataRequiredError
from ax.exceptions.generation_strategy import MaxParallelismReachedException

from ax.utils.common.base import SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import SerializationMixin, serialize_init_args

logger: Logger = get_logger(__name__)


class TransitionCriterion(SortableBase, SerializationMixin):
    # TODO: @mgarrard rename to ActionCriterion
    """
    Simple class to describe a condition which must be met for this GenerationNode to
    take an action such as generation, transition, etc.

    Args:
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
    """

    _transition_to: Optional[str] = None

    def __init__(
        self,
        transition_to: Optional[str] = None,
        block_transition_if_unmet: Optional[bool] = True,
        block_gen_if_met: Optional[bool] = False,
    ) -> None:
        self._transition_to = transition_to
        self.block_transition_if_unmet = block_transition_if_unmet
        self.block_gen_if_met = block_gen_if_met

    @property
    def transition_to(self) -> Optional[str]:
        """The name of the next GenerationNode after this TransitionCriterion is
        completed, if it exists.
        """
        return self._transition_to

    @abstractmethod
    def is_met(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]] = None
    ) -> bool:
        """If the criterion of this TransitionCriterion is met, returns True."""
        pass

    @abstractmethod
    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        """Error to be raised if the `block_gen_if_met` flag is set to True."""
        pass

    @property
    def criterion_class(self) -> str:
        """Name of the class of this TransitionCriterion."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.criterion_class}({serialize_init_args(obj=self)})"

    @property
    def _unique_id(self) -> str:
        """Unique id for this TransitionCriterion."""
        # TODO @mgarrard validate that this is unique enough
        return str(self)


class TrialBasedCriterion(TransitionCriterion):
    """Common class for transition criterion that are based on trial information.

    Args:
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to Flase since we can complete and move on from
            this node without ever reaching its threshold.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        only_in_statuses: A list of trial statuses to filter on when checking the
            criterion threshold.
        not_in_statuses: A list of trial statuses to exclude when checking the
            criterion threshold.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
    """

    def __init__(
        self,
        threshold: int,
        block_transition_if_unmet: Optional[bool] = True,
        block_gen_if_met: Optional[bool] = False,
        only_in_statuses: Optional[List[TrialStatus]] = None,
        not_in_statuses: Optional[List[TrialStatus]] = None,
        transition_to: Optional[str] = None,
        use_all_trials_in_exp: Optional[bool] = False,
    ) -> None:
        self.threshold = threshold
        self.only_in_statuses = only_in_statuses
        self.not_in_statuses = not_in_statuses
        self.use_all_trials_in_exp = use_all_trials_in_exp
        super().__init__(
            transition_to=transition_to,
            block_transition_if_unmet=block_transition_if_unmet,
            block_gen_if_met=block_gen_if_met,
        )

    def experiment_trials_by_status(
        self, experiment: Experiment, statuses: List[TrialStatus]
    ) -> Set[int]:
        """Get the trial indices from the entire experiment with the desired
        statuses.

        Args:
            experiment: The experiment associated with this GenerationStrategy.
            statuses: The trial statuses to filter on.
        Returns:
            The trial indices in the experiment with the desired statuses.
        """
        exp_trials_with_statuses = set()
        for status in statuses:
            exp_trials_with_statuses = exp_trials_with_statuses.union(
                experiment.trial_indices_by_status[status]
            )
        return exp_trials_with_statuses

    def all_trials_to_check(self, experiment: Experiment) -> Set[int]:
        """All the trials to check from the entire experiment that meet
        all the provided status filters.

        Args:
            experiment: The experiment associated with this GenerationStrategy.
        """
        trials_to_check = set(experiment.trials.keys())
        if self.only_in_statuses is not None:
            trials_to_check = self.experiment_trials_by_status(
                experiment=experiment, statuses=self.only_in_statuses
            )
        # exclude the trials to those not in the specified statuses
        if self.not_in_statuses is not None:
            trials_to_check -= self.experiment_trials_by_status(
                experiment=experiment, statuses=self.not_in_statuses
            )
        return trials_to_check

    def num_contributing_to_threshold(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]]
    ) -> int:
        """Returns the number of trials contributing to the threshold.

        Args:
            experiment: The experiment associated with this GenerationStrategy.
            trials_from_node: The set of trials generated by this GenerationNode.
        """
        all_trials_to_check = self.all_trials_to_check(experiment=experiment)
        # Some criteria may rely on experiment level data, instead of only trials
        # generated from the node associated with the criterion.
        if self.use_all_trials_in_exp:
            return len(all_trials_to_check)

        if trials_from_node is None:
            logger.warning(
                "`trials_from_node` is None, will check threshold on"
                + " experiment level.",
            )
            return len(all_trials_to_check)
        return len(trials_from_node.intersection(all_trials_to_check))

    def num_till_threshold(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]]
    ) -> int:
        """Returns the number of trials needed to meet the threshold.

        Args:
            experiment: The experiment associated with this GenerationStrategy.
            trials_from_node: The set of trials generated by this GenerationNode.
        """
        return self.threshold - self.num_contributing_to_threshold(
            experiment=experiment, trials_from_node=trials_from_node
        )

    def is_met(
        self,
        experiment: Experiment,
        trials_from_node: Optional[Set[int]] = None,
        block_continued_generation: Optional[bool] = False,
    ) -> bool:
        """Returns if this criterion has been met given its constraints.
        Args:
            experiment: The experiment associated with this GenerationStrategy.
            trials_from_node: The set of trials generated by this GenerationNode.
            block_continued_generation: A flag to prevent continued generation from the
                associated GenerationNode if this criterion is met but other criterion
                remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
                MaxTrials has been reached. If this flag is set to true on MaxTrials
                then we will raise an error, otherwise we will continue to generate
                trials until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        """
        return (
            self.num_contributing_to_threshold(
                experiment=experiment, trials_from_node=trials_from_node
            )
            >= self.threshold
        )


class MaxGenerationParallelism(TrialBasedCriterion):
    """Specific TransitionCriterion implementation which defines the maximum number
    of trials that can simultaneously be in the designated trial statuses. The
    default behavior is to block generation from the associated GenerationNode if the
    threshold is met. This is configured via the `block_gen_if_met` flag being set to
    True. This criterion defaults to not blocking transition to another node via the
    `block_transition_if_unmet` flag being set to False.

    Args:
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        only_in_statuses: A list of trial statuses to filter on when checking the
            criterion threshold.
        not_in_statuses: A list of trial statuses to exclude when checking the
            criterion threshold.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: Optional[List[TrialStatus]] = None,
        not_in_statuses: Optional[List[TrialStatus]] = None,
        transition_to: Optional[str] = None,
        block_transition_if_unmet: Optional[bool] = False,
        block_gen_if_met: Optional[bool] = True,
        use_all_trials_in_exp: Optional[bool] = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
        )

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        """If the block_continued_generation flag is set, raises the
        MaxParallelismReachedException error.
        """
        assert (
            node_name is not None and model_name is not None and experiment is not None
        )

        if self.block_gen_if_met:
            raise MaxParallelismReachedException(
                node_name=node_name,
                model_name=model_name,
                num_running=self.num_contributing_to_threshold(
                    experiment=experiment, trials_from_node=trials_from_node
                ),
            )


class MaxTrials(TrialBasedCriterion):
    """
    Simple class to enforce a maximum threshold for the number of trials with the
    designated statuses being generated by a specific GenerationNode. The default
    behavior is to block transition to the next node if the threshold is unmet, but
    not affect continued generation.

    Args:
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        only_in_statuses: A list of trial statuses to filter on when checking the
            criterion threshold.
        not_in_statuses: A list of trial statuses to exclude when checking the
            criterion threshold.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: Optional[List[TrialStatus]] = None,
        not_in_statuses: Optional[List[TrialStatus]] = None,
        transition_to: Optional[str] = None,
        block_transition_if_unmet: Optional[bool] = True,
        block_gen_if_met: Optional[bool] = False,
        use_all_trials_in_exp: Optional[bool] = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
        )

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        """If the block_continued_generation flag is set, raises an error because the
        remaining TransitionCriterion cannot be completed in the current state.
        """
        if self.block_gen_if_met:
            raise DataRequiredError(
                "All trials for current model have been generated, but not enough "
                "data has been observed to fit next model. Try again when more data"
                " are available."
            )


class MinTrials(TrialBasedCriterion):
    """
    Simple class to enforce a minimum threshold for the number of trials with the
    designated statuses being generated by a specific GenerationNode. The default
    behavior is to block transition to the next node if the threshold is unmet, but
    not affect continued generation.

    Args:
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        only_in_statuses: A list of trial statuses to filter on when checking the
            criterion threshold.
        not_in_statuses: A list of trial statuses to exclude when checking the
            criterion threshold.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: Optional[List[TrialStatus]] = None,
        not_in_statuses: Optional[List[TrialStatus]] = None,
        transition_to: Optional[str] = None,
        block_transition_if_unmet: Optional[bool] = True,
        block_gen_if_met: Optional[bool] = False,
        use_all_trials_in_exp: Optional[bool] = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
        )

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        """If the enforce flag is set, raises an error because the remaining
        TransitionCriterion cannot be completed in the current state.
        """
        if self.block_gen_if_met:
            raise DataRequiredError(
                f"This criterion, {self.criterion_class} has been met but cannot "
                "continue generation from its associated GenerationNode."
            )


class MinimumPreferenceOccurances(TransitionCriterion):
    """
    In a preference Experiment (i.e. Metric values may either be zero for No and
    nonzero for Yes) do not transition until a minimum number of both Yes and No
    responses have been received.

    Args:
        metric_name: name of the metric to check for preference occurrences.
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: MinimumTrialsInStatus has not been met yet, but
            MaxTrials has been reached. If this flag is set to true on MaxTrials then
            we will raise an error, otherwise we will continue to generate trials
            until MinimumTrialsInStatus is met (thus overriding MaxTrials).
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
    """

    def __init__(
        self,
        metric_name: str,
        threshold: int,
        transition_to: Optional[str] = None,
        block_gen_if_met: Optional[bool] = False,
        block_transition_if_unmet: Optional[bool] = True,
    ) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        super().__init__(
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
        )

    def is_met(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]] = None
    ) -> bool:
        # TODO: @mgarrard replace fetch_data with lookup_data
        data = experiment.fetch_data(metrics=[experiment.metrics[self.metric_name]])

        count_no = (data.df["mean"] == 0).sum()
        count_yes = (data.df["mean"] != 0).sum()

        return count_no >= self.threshold and count_yes >= self.threshold

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        pass


# TODO: Deprecate once legacy usecase is updated
class MinimumTrialsInStatus(TransitionCriterion):
    """
    Deprecated and replaced with more flexible MinTrials criterion.
    """

    def __init__(
        self,
        status: TrialStatus,
        threshold: int,
        transition_to: Optional[str] = None,
    ) -> None:
        self.status = status
        self.threshold = threshold
        super().__init__(transition_to=transition_to)

    def is_met(
        self, experiment: Experiment, trials_from_node: Optional[Set[int]] = None
    ) -> bool:
        return len(experiment.trial_indices_by_status[self.status]) >= self.threshold

    def block_continued_generation_error(
        self,
        node_name: Optional[str],
        model_name: Optional[str],
        experiment: Optional[Experiment],
        trials_from_node: Optional[Set[int]] = None,
    ) -> None:
        pass
