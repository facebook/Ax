#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Dict

from ae.lazarus.ae.core.base import Base


if TYPE_CHECKING:  # pragma: no cover
    from ae.lazarus.ae.core.base_trial import BaseTrial  # noqa


class Runner(Base, ABC):
    """Abstract base class for custom runner classes"""

    @abstractmethod
    def run(self, trial: "BaseTrial") -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        pass  # pragma: no cover

    @abstractproperty
    def staging_required(self) -> bool:
        """Whether the trial goes to staged or running state once deployed."""
        pass  # pragma: no cover
