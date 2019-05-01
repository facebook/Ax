#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from ax.core.base import Base


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Runner(Base, ABC):
    """Abstract base class for custom runner classes"""

    @abstractmethod
    def run(self, trial: "core.base_trial.BaseTrial") -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass  # pragma: no cover

    @property
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        return False
