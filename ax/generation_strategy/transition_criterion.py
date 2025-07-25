# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ax.core import MultiObjectiveOptimizationConfig

from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.experiment import Experiment

from ax.core.trial_status import TrialStatus
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.exceptions.generation_strategy import MaxParallelismReachedException

if TYPE_CHECKING:
    from ax.generation_strategy.generation_node import GenerationNode

from ax.utils.common.base import SortableBase
from ax.utils.common.serialization import SerializationMixin, serialize_init_args
from pyre_extensions import none_throws


DATA_REQUIRED_MSG = (
    "All trials for current node {node_name} have been generated, "
    "but not enough data has been observed to proceed to the next "
    "Generation node. Try again when more is are available."
)


class TransitionCriterion(SortableBase, SerializationMixin):
    """
    Simple class to describe a condition which must be met for this GenerationNode to
    take an action such as generation, transition, etc.

    Args:
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
    """

    _transition_to: str | None = None

    def __init__(
        self,
        transition_to: str | None = None,
        block_transition_if_unmet: bool | None = True,
        block_gen_if_met: bool | None = False,
        continue_trial_generation: bool | None = False,
    ) -> None:
        self._transition_to = transition_to
        self.block_transition_if_unmet = block_transition_if_unmet
        self.block_gen_if_met = block_gen_if_met
        self.continue_trial_generation = continue_trial_generation

    @property
    def transition_to(self) -> str | None:
        """The name of the next GenerationNode after this TransitionCriterion is
        completed, if it exists.
        """
        return self._transition_to

    @abstractmethod
    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        """If the criterion of this TransitionCriterion is met, returns True."""
        pass

    @abstractmethod
    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
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


class AutoTransitionAfterGen(TransitionCriterion):
    """A class to designate automatic transition from one GenerationNode to another.

    Args:
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to next.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: This criterion defaults to
            setting this to True to ensure we validate a GeneratorRun is generated by
            the current GenerationNode.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
    """

    def __init__(
        self,
        transition_to: str,
        block_transition_if_unmet: bool | None = True,
        continue_trial_generation: bool | None = True,
    ) -> None:
        super().__init__(
            transition_to=transition_to,
            block_transition_if_unmet=block_transition_if_unmet,
            continue_trial_generation=continue_trial_generation,
        )

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        """Return True as soon as any GeneratorRun is generated by this
        GenerationNode.
        """
        # Handle edge case where the InputConstructor for a GenerationNode
        # with this criterion requests no arms to be generated, therefore, indicating
        # that this GenerationNode should be skipped and so we can transition to the
        # next node as defined by this criterion.
        if curr_node._should_skip:
            return True
        last_gr_from_gs = curr_node.generation_strategy.last_generator_run
        return (
            last_gr_from_gs._generation_node_name == curr_node.node_name
            if last_gr_from_gs is not None
            else False
        )

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        """Error to be raised if the `block_gen_if_met` flag is set to True."""
        pass


class IsSingleObjective(TransitionCriterion):
    """A class to initiate transition based on whether the experiment is optimizing
    for a single objective or multiple objectives.

    Args:
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to next.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: This criterion defaults to
            setting this to True to ensure we validate a GeneratorRun is generated by
            the current GenerationNode.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
    """

    def __init__(
        self,
        transition_to: str,
        block_transition_if_unmet: bool | None = True,
        continue_trial_generation: bool | None = False,
    ) -> None:
        super().__init__(
            transition_to=transition_to,
            block_transition_if_unmet=block_transition_if_unmet,
            continue_trial_generation=continue_trial_generation,
        )

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        """Return True if the optimization config is not of type
        ``MultiObjectiveOptimizationConfig``."""
        return (
            not isinstance(
                experiment.optimization_config, MultiObjectiveOptimizationConfig
            )
            if experiment.optimization_config is not None
            else True
        )

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        # TODO: @mgarrard add error message that makes sense
        pass


class TrialBasedCriterion(TransitionCriterion):
    """Common class for transition criterion that are based on trial information.

    Args:
        threshold: The threshold as an integer for this criterion. Ex: If we want to
            generate at most 3 trials, then the threshold is 3.
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        only_in_statuses: A list of trial statuses to filter on when checking the
            criterion threshold.
        not_in_statuses: A list of trial statuses to exclude when checking the
            criterion threshold.
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
        count_only_trials_with_data: If set to True, only trials with data will be
            counted towards the ``threshold``. Defaults to False.
    """

    def __init__(
        self,
        threshold: int,
        block_transition_if_unmet: bool | None = True,
        block_gen_if_met: bool | None = False,
        only_in_statuses: list[TrialStatus] | None = None,
        not_in_statuses: list[TrialStatus] | None = None,
        transition_to: str | None = None,
        use_all_trials_in_exp: bool | None = False,
        continue_trial_generation: bool | None = False,
        count_only_trials_with_data: bool = False,
    ) -> None:
        self.threshold = threshold
        self.only_in_statuses = only_in_statuses
        self.not_in_statuses = not_in_statuses
        self.use_all_trials_in_exp = use_all_trials_in_exp
        self.count_only_trials_with_data = count_only_trials_with_data
        super().__init__(
            transition_to=transition_to,
            block_transition_if_unmet=block_transition_if_unmet,
            block_gen_if_met=block_gen_if_met,
            continue_trial_generation=continue_trial_generation,
        )

    def experiment_trials_by_status(
        self, experiment: Experiment, statuses: list[TrialStatus]
    ) -> set[int]:
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

    def all_trials_to_check(self, experiment: Experiment) -> set[int]:
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
        self, experiment: Experiment, trials_from_node: set[int]
    ) -> int:
        """Returns the number of trials contributing to the threshold.

        Args:
            experiment: The experiment associated with this GenerationStrategy.
            trials_from_node: The set of trials generated by this GenerationNode.
        """
        all_trials_to_check = self.all_trials_to_check(experiment=experiment)
        if self.count_only_trials_with_data:
            all_trials_to_check = {
                trial_index
                for trial_index in all_trials_to_check
                # TODO[@mgarrard]: determine if we need to actually check data with
                # more granularity, e.g. number of days of data, etc.
                if trial_index in experiment.data_by_trial
            }
        # Some criteria may rely on experiment level data, instead of only trials
        # generated from the node associated with the criterion.
        if self.use_all_trials_in_exp:
            return len(all_trials_to_check)

        return len(trials_from_node.intersection(all_trials_to_check))

    def num_till_threshold(
        self, experiment: Experiment, trials_from_node: set[int]
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
        curr_node: GenerationNode,
    ) -> bool:
        """Returns if this criterion has been met given its constraints.
        Args:
            experiment: The experiment associated with this GenerationStrategy.
            trials_from_node: The set of trials generated by this GenerationNode.
            block_continued_generation: A flag to prevent continued generation from the
                associated GenerationNode if this criterion is met but other criterion
                remain unmet. Ex: ``MinTrials`` has not been met yet, but
                MinTrials has been reached. If this flag is set to true on MinTrials
                then we will raise an error, otherwise we will continue to generate
                trials until ``MinTrials`` is met (thus overriding MinTrials).
        """
        return (
            self.num_contributing_to_threshold(
                experiment=experiment, trials_from_node=curr_node.trials_from_node
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
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True. Defaults to
            False for MaxGenerationParallelism since this criterion isn't currently
            used for node -> node or trial -> trial transition.
        count_only_trials_with_data: If set to True, only trials with data will be
            counted towards the ``threshold``. Defaults to False.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: list[TrialStatus] | None = None,
        not_in_statuses: list[TrialStatus] | None = None,
        transition_to: str | None = None,
        block_transition_if_unmet: bool | None = False,
        block_gen_if_met: bool | None = True,
        use_all_trials_in_exp: bool | None = False,
        continue_trial_generation: bool | None = True,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
            continue_trial_generation=continue_trial_generation,
        )

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        """Raises the appropriate error (should only be called when the
        ``GenerationNode`` is blocked from continued generation). For this
        class, the exception is ``MaxParallelismReachedException``.
        """
        assert self.block_gen_if_met  # Sanity check.
        raise MaxParallelismReachedException(
            node_name=node_name,
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
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
        count_only_trials_with_data: If set to True, only trials with data will be
            counted towards the ``threshold``. Defaults to False.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: list[TrialStatus] | None = None,
        not_in_statuses: list[TrialStatus] | None = None,
        transition_to: str | None = None,
        block_transition_if_unmet: bool | None = True,
        block_gen_if_met: bool | None = False,
        use_all_trials_in_exp: bool | None = False,
        continue_trial_generation: bool | None = False,
        count_only_trials_with_data: bool = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
            continue_trial_generation=continue_trial_generation,
            count_only_trials_with_data=count_only_trials_with_data,
        )

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        """Raises the appropriate error (should only be called when the
        ``GenerationNode`` is blocked from continued generation). For this
        class, the exception is ``DataRequiredError``.
        """
        assert self.block_gen_if_met  # Sanity check.
        raise DataRequiredError(DATA_REQUIRED_MSG.format(node_name=node_name))


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
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        use_all_trials_in_exp: A flag to use all trials in the experiment, instead of
            only those generated by the current GenerationNode.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
        count_only_trials_with_data: If set to True, only trials with data will be
            counted towards the ``threshold``. Defaults to False.
    """

    def __init__(
        self,
        threshold: int,
        only_in_statuses: list[TrialStatus] | None = None,
        not_in_statuses: list[TrialStatus] | None = None,
        transition_to: str | None = None,
        block_transition_if_unmet: bool | None = True,
        block_gen_if_met: bool | None = False,
        use_all_trials_in_exp: bool | None = False,
        continue_trial_generation: bool | None = False,
        count_only_trials_with_data: bool = False,
    ) -> None:
        super().__init__(
            threshold=threshold,
            only_in_statuses=only_in_statuses,
            not_in_statuses=not_in_statuses,
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
            use_all_trials_in_exp=use_all_trials_in_exp,
            continue_trial_generation=continue_trial_generation,
            count_only_trials_with_data=count_only_trials_with_data,
        )

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        """Raises the appropriate error (should only be called when the
        ``GenerationNode`` is blocked from continued generation). For this
        class, the exception is ``DataRequiredError``.
        """
        assert self.block_gen_if_met  # Sanity check.
        raise DataRequiredError(DATA_REQUIRED_MSG.format(node_name=node_name))


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
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
    """

    def __init__(
        self,
        metric_name: str,
        threshold: int,
        transition_to: str | None = None,
        block_gen_if_met: bool | None = False,
        block_transition_if_unmet: bool | None = True,
    ) -> None:
        self.metric_name = metric_name
        self.threshold = threshold
        super().__init__(
            transition_to=transition_to,
            block_gen_if_met=block_gen_if_met,
            block_transition_if_unmet=block_transition_if_unmet,
        )

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        # TODO: @mgarrard replace fetch_data with lookup_data
        data = experiment.fetch_data(metrics=[experiment.metrics[self.metric_name]])

        count_no = (data.df["mean"] == 0).sum()
        count_yes = (data.df["mean"] != 0).sum()

        return count_no >= self.threshold and count_yes >= self.threshold

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        pass


class AuxiliaryExperimentCheck(TransitionCriterion):
    """A class to transition from one GenerationNode to another by checking if certain
    types of Auxiliary Experiment purposes exists.

    A common use case is to use auxiliary_experiment_purposes_to_include to transition
    to a node and auxiliary_experiment_purposes_to_exclude to transition away from it.

    Example usage: In Bayesian optimization with preference exploration (BOPE), we
    check if the preference exploration (PE) auxiliary experiment exists to indicate
    transition to the node that will generate candidates based on the learned
    objective.Since preference exploration is usually conducted after the exploratory
    batch is completed, we do not know at experiment creation time if the PE node
    should be used during the GenerationStrategy.

    Args:
        transition_to: The name of the GenerationNode the GenerationStrategy should
            transition to when this criterion is met, if it exists.
        auxiliary_experiment_purposes_to_include: Optional list of auxiliary experiment
            purposes we expect to have. This can be helpful when need to transition to
            a node based on AuxiliaryExperimentPurpose. Criterion is met when all
            inclusion and exclusion checks pass.
        auxiliary_experiment_purposes_to_exclude: Optional list of auxiliary experiment
            purpose we expect to not have. This can be helpful when need to transition
            out of a node based on AuxiliaryExperimentPurpose. Criterion is met when
            all inclusion and exclusion checks pass.
        block_gen_if_met: A flag to prevent continued generation from the
            associated GenerationNode if this criterion is met but other criterion
            remain unmet. Ex: ``MinTrials`` has not been met yet, but
            MinTrials has been reached. If this flag is set to true on MinTrials then
            we will raise an error, otherwise we will continue to generate trials
            until ``MinTrials`` is met (thus overriding MinTrials).
        block_transition_if_unmet: A flag to prevent the node from completing and
            being able to transition to another node. Ex: MaxGenerationParallelism
            defaults to setting this to False since we can complete and move on from
            this node without ever reaching its threshold.
        continue_trial_generation: A flag to indicate that all generation for a given
            trial is not completed, and thus even after transition, the next node will
            continue to generate arms for the same trial. Example usage: in
            ``BatchTrial``s we may  enable generation of arms within a batch from
            different ``GenerationNodes`` by setting this flag to True.
    """

    def __init__(
        self,
        transition_to: str,
        auxiliary_experiment_purposes_to_include: (
            list[AuxiliaryExperimentPurpose] | None
        ) = None,
        auxiliary_experiment_purposes_to_exclude: (
            list[AuxiliaryExperimentPurpose] | None
        ) = None,
        block_transition_if_unmet: bool | None = True,
        block_gen_if_met: bool | None = False,
        continue_trial_generation: bool | None = False,
    ) -> None:
        super().__init__(
            transition_to=transition_to,
            block_transition_if_unmet=block_transition_if_unmet,
            block_gen_if_met=block_gen_if_met,
            continue_trial_generation=continue_trial_generation,
        )

        if (
            auxiliary_experiment_purposes_to_include is None
            and auxiliary_experiment_purposes_to_exclude is None
        ):
            raise UserInputError(
                f"{self.__class__} cannot have both "
                "`auxiliary_experiment_purposes_to_include` and "
                "`auxiliary_experiment_purposes_to_exclude` be None."
            )
        self.auxiliary_experiment_purposes_to_include = (
            auxiliary_experiment_purposes_to_include
        )
        self.auxiliary_experiment_purposes_to_exclude = (
            auxiliary_experiment_purposes_to_exclude
        )

    def check_aux_exp_purposes(
        self,
        aux_exp_by_purposes: dict[
            AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]
        ],
        include: bool,
        expected_aux_exp_purposes: list[AuxiliaryExperimentPurpose] | None = None,
    ) -> bool:
        """Helper method to check if all elements in expected_aux_exp_purposes
        are in (or not in) aux_exp_purposes"""
        if expected_aux_exp_purposes is not None:
            for purpose in none_throws(expected_aux_exp_purposes):
                purpose_present = (
                    purpose in aux_exp_by_purposes
                    and len(aux_exp_by_purposes[purpose]) > 0
                )
                if purpose_present != include:
                    return False
        return True

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        """Check if the experiment has auxiliary experiments for certain purpose."""
        inclusion_check = self.check_aux_exp_purposes(
            aux_exp_by_purposes=experiment.auxiliary_experiments_by_purpose,
            include=True,
            expected_aux_exp_purposes=self.auxiliary_experiment_purposes_to_include,
        )
        exclusion_check = self.check_aux_exp_purposes(
            aux_exp_by_purposes=experiment.auxiliary_experiments_by_purpose,
            include=False,
            expected_aux_exp_purposes=self.auxiliary_experiment_purposes_to_exclude,
        )
        return inclusion_check and exclusion_check

    def block_continued_generation_error(
        self,
        node_name: str,
        experiment: Experiment,
        trials_from_node: set[int],
    ) -> None:
        """Raises the appropriate error (should only be called when the
        ``GenerationNode`` is blocked from continued generation). For this
        class, the exception is ``DataRequiredError``.
        """
        assert self.block_gen_if_met  # Sanity check.
        raise DataRequiredError(
            f"This criterion, {self.criterion_class} has been met but cannot "
            "continue generation from its associated GenerationNode."
        )


# TODO: Deprecate once legacy usecase is updated
class MinimumTrialsInStatus(TransitionCriterion):
    """
    Deprecated and replaced with more flexible MinTrials criterion.
    """

    def __init__(
        self,
        status: TrialStatus,
        threshold: int,
        transition_to: str | None = None,
    ) -> None:
        self.status = status
        self.threshold = threshold
        super().__init__(transition_to=transition_to)

    def is_met(
        self,
        experiment: Experiment,
        curr_node: GenerationNode,
    ) -> bool:
        return len(experiment.trial_indices_by_status[self.status]) >= self.threshold

    def block_continued_generation_error(
        self,
        node_name: str | None,
        experiment: Experiment | None,
        trials_from_node: set[int],
    ) -> None:
        pass
