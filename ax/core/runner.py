#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from ax.utils.common.base import Base
from ax.utils.common.serialization import extract_init_args, serialize_init_args


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Runner(Base, ABC):
    """Abstract base class for custom runner classes"""

    @classmethod
    def serialize_init_args(cls, runner: Runner) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.
        """
        return serialize_init_args(object=runner)

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the runner.
        Used for storage.
        """
        return extract_init_args(args=args, class_=cls)

    @abstractmethod
    def run(self, trial: core.base_trial.BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass  # pragma: no cover

    def stop(
        self, trial: core.base_trial.BaseTrial, reason: Optional[str] = None
    ) -> Dict[str, Any]:
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

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False

    def clone(self) -> Runner:
        """Create a copy of this Runner."""
        cls = type(self)
        # pyre-ignore[45]: Cannot instantiate abstract class `Runner`.
        return cls(
            **serialize_init_args(self),
        )
