#!/usr/bin/env python3

from typing import TYPE_CHECKING, Dict, List, Optional

from ae.lazarus.ae.core.base_trial import BaseTrial, immutable_once_run
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.utils.common.typeutils import not_none


if TYPE_CHECKING:
    from ae.lazarus.ae.core.experiment import (
        Experiment,
    )  # noqa F401  # pragma: no cover


class Trial(BaseTrial):
    """Trial that only has one attached condition and no condition weights.

    Args:
        experiment (Experiment): experiment, to which this trial is attached
        generator_run (GeneratorRun, optional): generator_run associated with
            this trial. Trial has only one generator run (and thus condition)
            attached to it. This can also be set later through `add_condition`
            or `add_generator_run`, but a trial's associated genetor run is
            immutable once set.
    """

    def __init__(
        self, experiment: "Experiment", generator_run: Optional[GeneratorRun] = None
    ) -> None:
        super().__init__(experiment)
        self._generator_run = None
        if generator_run is not None:
            self.add_generator_run(generator_run=generator_run)

    @property
    def generator_run(self) -> Optional[GeneratorRun]:
        """Generator run attached to this trial."""
        return self._generator_run

    @property
    def condition(self) -> Optional[Condition]:
        """The condition associated with this batch."""
        if self.generator_run is not None and len(self.generator_run.conditions) > 1:
            raise ValueError(  # pragma: no cover
                "Generator run associated with this trial included multiple "
                "conditions, but trial expects only one."
            )
        return (
            self.generator_run.conditions[0] if self.generator_run is not None else None
        )

    @immutable_once_run
    def add_condition(self, condition: Condition) -> "Trial":
        """Add condition to the trial.

        Returns:
            The trial instance.
        """

        return self.add_generator_run(
            generator_run=GeneratorRun(conditions=[condition])
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> "Trial":
        """Add a generator run to the trial.

        Note: since trial includes only one condition, this will raise a ValueError if
        the generator run includes multiple conditions.

        Returns:
            The trial instance.
        """

        if len(generator_run.conditions) > 1:
            raise ValueError(
                "Trial includes only one condition, but this generator run "
                "included multiple."
            )

        self._check_existing_and_name_condition(generator_run.conditions[0])

        # TODO validate that conditions belong to search space
        self._generator_run = generator_run
        return self

    @property
    def conditions(self) -> List[Condition]:
        """All conditions attached to this trial.

        Returns:
            conditions (List[Condition], optional): list of a single condition
                attached to this trial if there is one, else None.
        """
        return [self.condition] if self.condition is not None else []

    @property
    def conditions_by_name(self) -> Dict[str, Condition]:
        """Dictionary of all conditions attached to this trial with their names
        as keys.

        Returns:
            conditions (Dict[Condition], optional): dictionary of a single
                condition name to condition if one is attached to this trial,
                else None.
        """
        return (
            {self.condition.name: self.condition} if self.condition is not None else {}
        )

    @property
    def abandoned_conditions(self) -> List[Condition]:
        """Abandoned conditions attached to this trial."""
        return (
            [self.condition]
            if self.generator_run is not None
            and self.condition is not None
            and self.is_abandoned
            else []
        )

    @property
    def objective_mean(self) -> Optional[float]:
        """Objective mean for the condition attached to this trial."""
        # For SimpleExperiment, fetch_trial_data just executes eval_trial.
        df = self.experiment.fetch_trial_data(trial_index=self.index).df
        if df.empty or self.experiment.optimization_config is None:
            return None
        objective_name = not_none(
            self.experiment.optimization_config
        ).objective.metric.name
        return df.loc[df["metric_name"] == objective_name].iloc[0]["mean"]

    def __repr__(self) -> str:
        return (
            "Trial("
            f"experiment_name='{self._experiment.name}', "
            f"index={self._index}, "
            f"status={self._status})"
        )
