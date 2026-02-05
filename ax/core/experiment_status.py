#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from enum import Enum


class ExperimentStatus(int, Enum):
    """Enum of experiment status.

    General lifecycle of an experiment is:::

        DRAFT --> INITIALIZATION --> OPTIMIZATION --> COMPLETED

    Experiment is marked as ``DRAFT`` immediately upon its creation when
    the experiment is still being configured (search space, optimization config, etc.).

    Once the experiment is fully configured and begins initial exploration,
    it transitions to ``INITIALIZATION``. This is typically when the first trials
    are being generated to explore the search space.

    After initial exploration completes (typically after some data has been collected),
    the experiment transitions to ``OPTIMIZATION``, where Bayesian optimization or
    other adaptive methods are used to find optimal configurations.

    ``COMPLETED`` indicates the experiment has successfully finished its objectives.

    Note: This status tracks the high-level experiment lifecycle and is independent
    of individual trial statuses. An experiment in OPTIMIZATION status may have
    trials in various states (RUNNING, COMPLETED, FAILED, etc.).
    """

    DRAFT = 0
    INITIALIZATION = 1
    OPTIMIZATION = 2
    COMPLETED = 4

    @property
    def is_active(self) -> bool:
        """True if experiment is actively running trials."""
        return (
            self == ExperimentStatus.INITIALIZATION
            or self == ExperimentStatus.OPTIMIZATION
        )

    @property
    def is_draft(self) -> bool:
        """True if experiment is in draft phase."""
        return self == ExperimentStatus.DRAFT

    @property
    def is_initialization(self) -> bool:
        """True if experiment is in initialization phase."""
        return self == ExperimentStatus.INITIALIZATION

    @property
    def is_optimization(self) -> bool:
        """True if experiment is in optimization phase."""
        return self == ExperimentStatus.OPTIMIZATION

    @property
    def is_completed(self) -> bool:
        """True if experiment has successfully completed."""
        return self == ExperimentStatus.COMPLETED

    def __format__(self, fmt: str) -> str:
        """Define `__format__` to avoid pulling the `__format__` from the `int`
        mixin (since its better for statuses to show up as `DRAFT` than as
        just an int that is difficult to interpret).

        E.g. experiment representation with the overridden method is:
        "Experiment(name='test', status=ExperimentStatus.DRAFT)".

        Docs on enum formatting: https://docs.python.org/3/library/enum.html#others.
        """
        return f"{self!s}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"
