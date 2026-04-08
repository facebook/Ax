#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Self, TYPE_CHECKING

from ax.utils.common.base import Base
from ax.utils.common.sentinel import Unset
from ax.utils.common.serialization import SerializationMixin


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class RunnerConfig:
    @dataclass(frozen=True)
    class SearchSpaceUpdateArguments:
        """Base arguments for search space updates. Override in RunnerConfig
        subclasses to add runner-specific fields."""

        pass

    @dataclass(frozen=True)
    class RunnerUpdateArguments:
        """Base arguments for general runner updates. Override in RunnerConfig
        subclasses to add runner-specific fields."""

        pass


class Runner(Base, SerializationMixin, ABC):
    """Abstract base class for custom runner classes"""

    config_type: ClassVar[type[RunnerConfig]] = RunnerConfig

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False

    @property
    def run_metadata_report_keys(self) -> list[str]:
        """A list of keys of the metadata dict returned by `run()` that are
        relevant outside the runner-internal impolementation. These can e.g.
        be reported in `orchestrator.report_results()`."""
        return []

    @abstractmethod
    def run(self, trial: core.base_trial.BaseTrial) -> dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass

    def run_multiple(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[int, dict[str, Any]]:
        """Runs a single evaluation for each of the given trials. Useful when deploying
        multiple trials at once is more efficient than deploying them one-by-one.
        Used in Ax ``Orchestrator``.

        NOTE: By default simply loops over `run_trial`. Should be overwritten
        if deploying multiple trials in batch is preferable.

        Args:
            trials: Iterable of trials to be deployed, each containing arms with
                parameterizations to be evaluated. Can be a `Trial`
                if contains only one arm or a `BatchTrial` if contains
                multiple arms.

        Returns:
            Dict of trial index to the run metadata of that trial from the deployment
            process.
        """
        return {trial.index: self.run(trial=trial) for trial in trials}

    def poll_available_capacity(self) -> int:
        """Checks how much available capacity there is to schedule trial evaluations.
        Required for runners used with Ax ``Orchestrator``.

        NOTE: This method might be difficult to implement in some systems. Returns -1
        if capacity of the system is "unlimited" or "unknown"
        (meaning that the ``Orchestrator`` should be trying to schedule as many trials
        as is possible without violating Orchestrator settings). There is no need to
        artificially force this method to limit capacity; ``Orchestrator`` has other
        limitations in place to limit number of trials running at once,
        like the ``OrchestratorOptions.max_pending_trials`` setting, or
        more granular control in the form of the `max_parallelism`
        setting in each of the `GenerationStep`s of a `GenerationStrategy`).

        Returns:
            An integer, representing how many trials there is available capacity for;
            -1 if capacity is "unlimited" or not possible to know in advance.
        """
        return -1

    def poll_trial_status(
        self, trials: Iterable[core.base_trial.BaseTrial]
    ) -> dict[core.base_trial.TrialStatus, set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Orchestrator``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `poll_trial_status` "
            "method."
        )

    def poll_exception(self, trial: core.base_trial.BaseTrial) -> str:
        """Returns the exception from a trial.

        Args:
            trial: Trial to get exception for.

        Returns:
            Exception string.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `poll_exception` method."
        )

    def stop(
        self, trial: core.base_trial.BaseTrial, reason: str | None = None
    ) -> dict[str, Any]:
        """Stop a trial based on custom runner subclass implementation.

        Optional method.

        Args:
            trial: The trial to stop.
            reason: A message containing information why the trial is to be stopped.

        Returns:
            A dictionary of run metadata from the stopping process.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a `stop` method."
        )

    def on_search_space_update(
        self,
        search_space: core.search_space.SearchSpace,
        arguments: RunnerConfig.SearchSpaceUpdateArguments | None = None,
    ) -> None:
        """Called after the experiment's search space has been updated.

        Validates the proposed runner-side changes, then applies them.
        Subclasses should override ``_validate_on_search_space_update``
        to add validation logic.

        Args:
            search_space: The updated search space.
            arguments: Optional typed arguments carrying runner-specific
                data. Subclasses should define a ``RunnerConfig`` subclass
                with a nested ``SearchSpaceUpdateArguments`` dataclass to
                declare supported fields.
        """
        if arguments is not None:
            UpdateArgsClass = type(self).config_type.SearchSpaceUpdateArguments
            if not isinstance(arguments, UpdateArgsClass):
                raise TypeError(
                    f"Expected {UpdateArgsClass.__name__}, "
                    f"got {type(arguments).__name__}."
                )
        self._validate_on_search_space_update(search_space, arguments)
        if arguments is not None:
            self._set_attributes(arguments)

    def _validate_on_search_space_update(
        self,
        search_space: core.search_space.SearchSpace,
        arguments: RunnerConfig.SearchSpaceUpdateArguments | None = None,
    ) -> None:
        """Override in subclasses to reject invalid search space updates
        before the runner's state is modified. The runner's attributes still
        hold their old values at this point; use the ``arguments`` to determine
        the proposed new state.

        Args:
            search_space: The already-updated search space.
            arguments: The proposed runner-side changes, if any.
        """
        pass

    def update(
        self,
        arguments: RunnerConfig.RunnerUpdateArguments,
        search_space: core.search_space.SearchSpace | None = None,
    ) -> None:
        """Update runner attributes at runtime.

        Validates that ``arguments`` is the correct type for this runner's
        ``config_type``, runs ``_validate_update`` to check that the
        update is permissible, then applies non-``_UNSET`` fields to the
        runner's instance attributes.

        Fields in ``arguments`` use ``_UNSET`` as their default to distinguish
        "not provided" from an explicit ``None``. Only fields whose value is
        not ``_UNSET`` are applied. The target attribute name defaults to the
        field name but can be overridden via
        ``metadata={"attr": "actual_attr_name"}`` in the dataclass field
        definition.

        Args:
            arguments: Typed arguments declaring which runner attributes
                to update.
            search_space: The experiment's current search space, if available.
                Forwarded to ``_validate_update`` for cross-validation.
        """
        expected_type = type(self).config_type.RunnerUpdateArguments
        if not isinstance(arguments, expected_type):
            raise TypeError(
                f"{type(self).__name__} expects "
                f"{expected_type.__qualname__}, "
                f"got {type(arguments).__name__}"
            )
        self._validate_update(arguments, search_space=search_space)
        self._set_attributes(arguments)

    def _validate_update(
        self,
        arguments: RunnerConfig.RunnerUpdateArguments,
        search_space: core.search_space.SearchSpace | None = None,
    ) -> None:
        """Override in subclasses to reject invalid updates before the runner's
        state is modified. The runner's attributes still hold their old values
        at this point; use the ``arguments`` to determine the proposed new
        state.

        Args:
            arguments: The proposed changes.
            search_space: The experiment's current search space.
        """
        pass

    def _set_attributes(
        self,
        arguments: (
            RunnerConfig.RunnerUpdateArguments | RunnerConfig.SearchSpaceUpdateArguments
        ),
    ) -> None:
        """Apply dataclass field values to self, skipping UNSET fields.

        Shared by ``update`` and ``on_search_space_update`` to ensure both
        follow the same validate-then-mutate pattern.
        """
        for field in fields(arguments):
            value = getattr(arguments, field.name)
            if isinstance(value, Unset):
                continue
            attr_name = field.metadata.get("attr", field.name)
            setattr(self, attr_name, value)

    def clone(self) -> Self:
        """Create a copy of this Runner."""
        cls = type(self)
        # pyre-ignore[45]: Cannot instantiate abstract class `Runner`.
        return cls(
            **cls.deserialize_init_args(args=cls.serialize_init_args(obj=self)),
        )

    def __eq__(self, other: Runner) -> bool:
        same_class = self.__class__ == other.__class__
        same_init_args = self.serialize_init_args(
            obj=self
        ) == other.serialize_init_args(obj=other)
        return same_class and same_init_args
