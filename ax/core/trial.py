#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import TYPE_CHECKING, Dict, List, Optional

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, immutable_once_run
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.utils.common.typeutils import not_none


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


class Trial(BaseTrial):
    """Trial that only has one attached arm and no arm weights.

    Args:
        experiment: experiment, to which this trial is attached
        generator_run: generator_run associated with
            this trial. Trial has only one generator run (and thus arm)
            attached to it. This can also be set later through `add_arm`
            or `add_generator_run`, but a trial's associated genetor run is
            immutable once set.
    """

    def __init__(
        self,
        experiment: "core.experiment.Experiment",
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
    ) -> None:
        super().__init__(experiment=experiment, trial_type=trial_type)
        self._generator_run = None
        if generator_run is not None:
            self.add_generator_run(generator_run=generator_run)

    @property
    def generator_run(self) -> Optional[GeneratorRun]:
        """Generator run attached to this trial."""
        return self._generator_run

    @property
    def arm(self) -> Optional[Arm]:
        """The arm associated with this batch."""
        if self.generator_run is not None and len(self.generator_run.arms) > 1:
            raise ValueError(  # pragma: no cover
                "Generator run associated with this trial included multiple "
                "arms, but trial expects only one."
            )
        return self.generator_run.arms[0] if self.generator_run is not None else None

    @immutable_once_run
    def add_arm(self, arm: Arm) -> "Trial":
        """Add arm to the trial.

        Returns:
            The trial instance.
        """

        return self.add_generator_run(
            generator_run=GeneratorRun(arms=[arm], type=GeneratorRunType.MANUAL.name)
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> "Trial":
        """Add a generator run to the trial.

        Note: since trial includes only one arm, this will raise a ValueError if
        the generator run includes multiple arms.

        Returns:
            The trial instance.
        """

        if len(generator_run.arms) > 1:
            raise ValueError(
                "Trial includes only one arm, but this generator run "
                "included multiple."
            )

        self.experiment.search_space.check_types(
            generator_run.arms[0].parameters, raise_error=True
        )

        self._check_existing_and_name_arm(generator_run.arms[0])

        self._generator_run = generator_run
        generator_run.index = 0
        return self

    @property
    def arms(self) -> List[Arm]:
        """All arms attached to this trial.

        Returns:
            arms: list of a single arm
                attached to this trial if there is one, else None.
        """
        return [self.arm] if self.arm is not None else []

    @property
    def arms_by_name(self) -> Dict[str, Arm]:
        """Dictionary of all arms attached to this trial with their names
        as keys.

        Returns:
            arms: dictionary of a single
                arm name to arm if one is attached to this trial,
                else None.
        """
        return {self.arm.name: self.arm} if self.arm is not None else {}

    @property
    def abandoned_arms(self) -> List[Arm]:
        """Abandoned arms attached to this trial."""
        return (
            [self.arm]
            if self.generator_run is not None
            and self.arm is not None
            and self.is_abandoned
            else []
        )

    @property
    def objective_mean(self) -> Optional[float]:
        """Objective mean for the arm attached to this trial."""
        # For SimpleExperiment, fetch_data just executes eval_trial.
        df = self.fetch_data().df
        if df.empty or self.experiment.optimization_config is None:
            return None
        objective_name = not_none(
            self.experiment.optimization_config
        ).objective.metric.name
        return df.loc[df["metric_name"] == objective_name].iloc[0]["mean"]

    def __repr__(self) -> str:
        return (
            "Trial("
            f"experiment_name='{self._experiment._name}', "
            f"index={self._index}, "
            f"status={self._status})"
        )
