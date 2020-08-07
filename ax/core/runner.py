#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from ax.utils.common.equality import Base
from ax.utils.common.serialization import extract_init_args, serialize_init_args


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Runner(Base, ABC):
    """Abstract base class for custom runner classes"""

    @classmethod
    def serialize_init_args(cls, runner: "Runner") -> Dict[str, Any]:
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
    def run(self, trial: "core.base_trial.BaseTrial") -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass  # pragma: no cover

    def stop(self, trial: "core.base_trial.BaseTrial") -> None:
        """Stop a trial based on custom runner subclass implementation.

        Optional to implement

        Args:
            trial: The trial to deploy.
        """
        pass

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False
