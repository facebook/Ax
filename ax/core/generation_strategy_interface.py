# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import AxError, UnsupportedError
from ax.utils.common.base import Base
from pyre_extensions import none_throws


class GenerationStrategyInterface(ABC, Base):
    """Interface for all generation strategies: standard Ax
    ``GenerationStrategy``, as well as non-standard (e.g. remote, external)
    generation strategies.

    NOTE: Currently in Beta; please do not use without discussion with the Ax
    developers.
    """

    _name: str
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Experiment | None = None

    # Constant for default number of arms to generate if `n` is not specified in
    # `gen` call and "total_concurrent_arms" is not set in experiment properties.
    DEFAULT_N: int = 1

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def gen_for_multiple_trials_with_multiple_models(
        self,
        experiment: Experiment,
        data: Data | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        n: int | None = None,
        num_trials: int = 1,
        arms_per_node: dict[str, int] | None = None,
    ) -> list[list[GeneratorRun]]:
        """Produce ``GeneratorRun``-s for multiple trials at once with the possibility
        of joining ``GeneratorRun``-s from multiple models into one ``BatchTrial``.

        Args:
            experiment: ``Experiment``, for which the generation strategy is producing
                a new generator run in the course of ``gen``, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's ``gen``, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the ``experiment``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            n: Integer representing how many total arms should be in the generator
                runs produced by this method. NOTE: Some underlying models may ignore
                the `n` and produce a model-determined number of arms. In that
                case this method will also output generator runs with number of
                arms that can differ from `n`.
            num_trials: Number of trials to generate generator runs for in this call.
                If not provided, defaults to 1.
            arms_per_node: An optional map from node name to the number of arms to
                generate from that node. If not provided, will default to the number
                of arms specified in the node's ``InputConstructors`` or n if no
                ``InputConstructors`` are defined on the node. We expect either n or
                arms_per_node to be provided, but not both, and this is an advanced
                argument that should only be used by advanced users.

        Returns:
            A list of lists of ``GeneratorRun``-s. Each outer list item represents
            a ``(Batch)Trial`` being suggested, with a list of ``GeneratorRun``-s for
            that trial.
        """
        # When implementing your subclass' override for this method, don't forget
        # to consider using "pending points", corresponding to arms in trials that
        # are currently running / being evaluated/
        ...

    def _gen_multiple(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Data | None = None,
        n: int = 1,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        **model_gen_kwargs: Any,
    ) -> list[GeneratorRun]:
        """Produce multiple generator runs at once, to be made into multiple
        trials on the experiment.

        NOTE: This is used to ensure that maximum parallelism and number
        of trials per node are not violated when producing many generator
        runs from this generation strategy in a row. Without this function,
        if one generates multiple generator runs without first making any
        of them into running trials, generation strategy cannot enforce that it only
        produces as many generator runs as are allowed by the parallelism
        limit and the limit on number of trials in current node.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment`.
            n: Integer representing how many arms should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            model_gen_kwargs: Keyword arguments that are passed through to
                ``GenerationNode.gen``, which will pass them through to
                ``ModelSpec.gen``, which will pass them to ``ModelBridge.gen``.
        """
        ...

    @abstractmethod
    def clone_reset(self) -> GenerationStrategyInterface:
        """Returns a clone of this generation strategy with all state reset."""
        ...

    @property
    def name(self) -> str:
        """Name of this generation strategy."""
        return self._name

    @property
    def experiment(self) -> Experiment:
        """Experiment, currently set on this generation strategy."""
        if self._experiment is None:
            raise AxError("No experiment set on generation strategy.")
        return none_throws(self._experiment)

    @experiment.setter
    def experiment(self, experiment: Experiment) -> None:
        """If there is an experiment set on this generation strategy as the
        experiment it has been generating generator runs for, check if the
        experiment passed in is the same as the one saved and log an information
        statement if its not. Set the new experiment on this generation strategy.
        """
        if self._experiment is not None and experiment._name != self.experiment._name:
            raise UnsupportedError(
                "This generation strategy has been used for experiment "
                f"{self.experiment._name} so far; cannot reset experiment"
                f" to {experiment._name}. If this is a new experiment, "
                "a new generation strategy should be created instead."
            )
        self._experiment = experiment
