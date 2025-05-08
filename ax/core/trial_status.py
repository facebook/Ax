#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from enum import Enum


class TrialStatus(int, Enum):
    """Enum of trial status.

    General lifecycle of a trial is:::

        CANDIDATE --> STAGED --> RUNNING --> COMPLETED
                  ------------->         --> FAILED (retryable)
                                         --> EARLY_STOPPED (deemed unpromising)
                  -------------------------> ABANDONED (non-retryable)

    Trial is marked as a ``CANDIDATE`` immediately upon its creation.

    Trials may be abandoned at any time prior to completion or failure.
    The difference between abandonment and failure is that the ``FAILED`` state
    is meant to express a possibly transient or retryable error, so trials in
    that state may be re-run and arm(s) in them may be resuggested by Ax models
    to be added to new trials.

    ``ABANDONED`` trials on the other end, indicate
    that the trial (and arms(s) in it) should not be rerun or added to new
    trials. A trial might be marked ``ABANDONED`` as a result of human-initiated
    action (if some trial in experiment is poorly-performing, deterministically
    failing etc., and should not be run again in the experiment). It might also
    be marked ``ABANDONED`` in an automated way if the trial's execution
    encounters an error that indicates that the arm(s) in the trial should bot
    be evaluated in the experiment again (e.g. the parameterization in a given
    arm deterministically causes trial evaluation to fail). Note that it's also
    possible to abandon a single arm in a `BatchTrial` via
    ``batch.mark_arm_abandoned``.

    Early-stopped refers to trials that were deemed
    unpromising by an early-stopping strategy and therefore terminated.

    Additionally, when trials are deployed, they may be in an intermediate
    staged state (e.g. scheduled but waiting for resources) or immediately
    transition to running. Note that ``STAGED`` trial status is not always
    applicable and depends on the ``Runner`` trials are deployed with
    (and whether a ``Runner`` is present at all; for example, in Ax Service
    API, trials are marked as ``RUNNING`` immediately when generated from
    ``get_next_trial``, skipping the ``STAGED`` status).

    NOTE: Data for abandoned trials (or abandoned arms in batch trials) is
    not passed to the model as part of training data, unless ``fit_abandoned``
    option is specified to model bridge. Additionally, data from MapMetrics is
    typically excluded unless the corresponding trial is completed.
    """

    CANDIDATE = 0
    STAGED = 1
    FAILED = 2
    COMPLETED = 3
    RUNNING = 4
    ABANDONED = 5
    DISPATCHED = 6  # Deprecated.
    EARLY_STOPPED = 7

    @property
    def is_terminal(self) -> bool:
        """True if trial is completed."""
        return (
            self == TrialStatus.ABANDONED
            or self == TrialStatus.COMPLETED
            or self == TrialStatus.FAILED
            or self == TrialStatus.EARLY_STOPPED
        )

    @property
    def expecting_data(self) -> bool:
        """True if trial is expecting data."""
        return self in STATUSES_EXPECTING_DATA

    @property
    def is_deployed(self) -> bool:
        """True if trial has been deployed but not completed."""
        return self == TrialStatus.STAGED or self == TrialStatus.RUNNING

    @property
    def is_failed(self) -> bool:
        """True if this trial is a failed one."""
        return self == TrialStatus.FAILED

    @property
    def is_abandoned(self) -> bool:
        """True if this trial is an abandoned one."""
        return self == TrialStatus.ABANDONED

    @property
    def is_candidate(self) -> bool:
        """True if this trial is a candidate."""
        return self == TrialStatus.CANDIDATE

    @property
    def is_completed(self) -> bool:
        """True if this trial is a successfully completed one."""
        return self == TrialStatus.COMPLETED

    @property
    def is_running(self) -> bool:
        """True if this trial is a running one."""
        return self == TrialStatus.RUNNING

    @property
    def is_early_stopped(self) -> bool:
        """True if this trial is an early stopped one."""
        return self == TrialStatus.EARLY_STOPPED

    def __format__(self, fmt: str) -> str:
        """Define `__format__` to avoid pulling the `__format__` from the `int`
        mixin (since its better for statuses to show up as `RUNNING` than as
        just an int that is difficult to interpret).

        E.g. batch trial representation with the overridden method is:
        "BatchTrial(experiment_name='test', index=0, status=TrialStatus.CANDIDATE)".

        Docs on enum formatting: https://docs.python.org/3/library/enum.html#others.
        """
        return f"{self!s}"

    def __repr__(self) -> str:
        return f"{self.__class__}.{self.name}"


DEFAULT_STATUSES_TO_WARM_START: list[TrialStatus] = [
    TrialStatus.RUNNING,
    TrialStatus.COMPLETED,
    TrialStatus.ABANDONED,
    TrialStatus.EARLY_STOPPED,
]

NON_ABANDONED_STATUSES: set[TrialStatus] = set(TrialStatus) - {TrialStatus.ABANDONED}

STATUSES_EXPECTING_DATA: list[TrialStatus] = [
    TrialStatus.RUNNING,
    TrialStatus.COMPLETED,
    TrialStatus.EARLY_STOPPED,
]
