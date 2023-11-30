# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.utils.common.base import Base
from ax.utils.common.typeutils import not_none


class GenerationStrategyInterface(ABC, Base):
    _name: Optional[str]
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment] = None

    @abstractmethod
    def gen_for_multiple_trials_with_multiple_models(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Optional[Data] = None,
        n: int = 1,
    ) -> List[List[GeneratorRun]]:
        """Produce GeneratorRuns for multiple trials at once with the possibility of
        ensembling, or using multiple models per trial, getting multiple
        GeneratorRuns per trial.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment`.
            n: Integer representing how many trials should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.

        Returns:
            A list of lists of lists generator runs. Each outer list represents
            a trial being suggested and  each inner list represents a generator
            run for that trial.
        """
        pass

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps.
        """
        if self._name is not None:
            return not_none(self._name)

        self._name = f"GenerationStrategy {self.db_id}"
        return not_none(self._name)

    @name.setter
    def name(self, name: str) -> None:
        """Set generation strategy name."""
        self._name = name

    @property
    def experiment(self) -> Experiment:
        """Experiment, currently set on this generation strategy."""
        if self._experiment is None:
            raise ValueError("No experiment set on generation strategy.")
        return not_none(self._experiment)

    @experiment.setter
    def experiment(self, experiment: Experiment) -> None:
        """If there is an experiment set on this generation strategy as the
        experiment it has been generating generator runs for, check if the
        experiment passed in is the same as the one saved and log an information
        statement if its not. Set the new experiment on this generation strategy.
        """
        if self._experiment is None or experiment._name == self.experiment._name:
            self._experiment = experiment
        else:
            raise ValueError(
                "This generation strategy has been used for experiment "
                f"{self.experiment._name} so far; cannot reset experiment"
                f" to {experiment._name}. If this is a new optimization, "
                "a new generation strategy should be created instead."
            )

    @property
    def last_generator_run(self) -> Optional[GeneratorRun]:
        """Latest generator run produced by this generation strategy.
        Returns None if no generator runs have been produced yet.
        """
        # Used to restore current model when decoding a serialized GS.
        return self._generator_runs[-1] if self._generator_runs else None
