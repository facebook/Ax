#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, immutable_once_run
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.types import TCandidateMetadata
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


class Trial(BaseTrial):
    """Trial that only has one attached arm and no arm weights.

    Args:
        experiment: Experiment, to which this trial is attached.
        generator_run: GeneratorRun, associated with this trial.
            Trial has only one generator run (of just one arm)
            attached to it. This can also be set later through `add_arm`
            or `add_generator_run`, but a trial's associated genetor run is
            immutable once set.
        trial_type: Type of this trial, if used in MultiTypeExperiment.
        ttl_seconds: If specified, trials will be considered failed after
            this many seconds since the time the trial was ran, unless the
            trial is completed before then. Meant to be used to detect
            'dead' trials, for which the evaluation process might have
            crashed etc., and which should be considered failed after
            their 'time to live' has passed.
        index: If specified, the trial's index will be set accordingly.
            This should generally not be specified, as in the index will be
            automatically determined based on the number of existing trials.
            This is only used for the purpose of loading from storage.
    """

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        index: Optional[int] = None,
    ) -> None:
        super().__init__(
            experiment=experiment,
            trial_type=trial_type,
            ttl_seconds=ttl_seconds,
            index=index,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self._generator_run = None
        if generator_run is not None:
            self.add_generator_run(generator_run=generator_run)

    @property
    def generator_run(self) -> Optional[GeneratorRun]:
        """Generator run attached to this trial."""
        return self._generator_run

    # pyre-ignore[6]: T77111662.
    @copy_doc(BaseTrial.generator_runs)
    @property
    def generator_runs(self) -> List[GeneratorRun]:
        gr = self._generator_run
        return [gr] if gr is not None else []

    @property
    def arm(self) -> Optional[Arm]:
        """The arm associated with this batch."""
        if self.generator_run is None:
            return None

        generator_run = not_none(self.generator_run)
        if len(generator_run.arms) == 0:
            return None
        elif len(generator_run.arms) > 1:
            raise ValueError(  # pragma: no cover
                "Generator run associated with this trial included multiple "
                "arms, but trial expects only one."
            )
        return generator_run.arms[0]

    @immutable_once_run
    def add_arm(
        self, arm: Arm, candidate_metadata: Optional[Dict[str, Any]] = None
    ) -> Trial:
        """Add arm to the trial.

        Returns:
            The trial instance.
        """

        return self.add_generator_run(
            generator_run=GeneratorRun(
                arms=[arm],
                type=GeneratorRunType.MANUAL.name,
                # pyre-ignore[6]: In call `GeneratorRun.__init__`, for 3rd parameter
                # `candidate_metadata_by_arm_signature`
                # expected `Optional[Dict[str, Optional[Dict[str, typing.Any]]]]`
                # but got `Optional[Dict[str, Dict[str, typing.Any]]]`
                candidate_metadata_by_arm_signature=None
                if candidate_metadata is None
                else {arm.signature: candidate_metadata.copy()},
            )
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> Trial:
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
        self._set_generation_step_index(
            generation_step_index=generator_run._generation_step_index
        )
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
            [not_none(self.arm)]
            if self.generator_run is not None
            and self.arm is not None
            and self.is_abandoned
            else []
        )

    @property
    def objective_mean(self) -> float:
        """Objective mean for the arm attached to this trial, retrieved from the
        latest data available for the objective for the trial.

        Note: the retrieved objective is the experiment-level objective at the
        time of the call to `objective_mean`, which is not necessarily the
        objective that was set at the time the trial was created or ran.
        """
        # For SimpleExperiment, fetch_data just executes eval_trial.

        opt_config = self.experiment.optimization_config
        if opt_config is None:
            raise ValueError(  # pragma: no cover
                "Experiment optimization config (and thus the objective) is not set."
            )
        return self.get_metric_mean(metric_name=opt_config.objective.metric.name)

    def get_metric_mean(self, metric_name: str) -> float:
        """Metric mean for the arm attached to this trial, retrieved from the
        latest data available for the metric for the trial.
        """

        fetch_result = self.lookup_data()

        try:
            df = fetch_result.df
            return df.loc[df["metric_name"] == metric_name].iloc[0]["mean"]
        except IndexError:  # pragma: no cover
            raise ValueError(f"Metric {metric_name} not yet in data for trial.")

    def __repr__(self) -> str:
        return (
            "Trial("
            f"experiment_name='{self._experiment._name}', "
            f"index={self._index}, "
            f"status={self._status}, "
            f"arm={self.arm})"
        )

    def _get_candidate_metadata_from_all_generator_runs(
        self,
    ) -> Dict[str, TCandidateMetadata]:
        """Retrieves candidate metadata from the generator run on this
        batch trial in the form of { arm name -> candidate metadata} mapping.
        """

        gr = self.generator_run
        if gr is None or gr.candidate_metadata_by_arm_signature is None:
            return {}

        cand_metadata = not_none(gr.candidate_metadata_by_arm_signature)
        return {a.name: cand_metadata.get(a.signature) for a in gr.arms}

    def _get_candidate_metadata(self, arm_name: str) -> TCandidateMetadata:
        """Retrieves candidate metadata for a specific arm."""

        gr = self.generator_run
        if gr is None or gr.arms[0].name != arm_name:
            raise ValueError(
                f"Arm by name {arm_name} is not part of trial #{self.index}."
            )

        if gr.candidate_metadata_by_arm_signature is None:
            return None

        arm = gr.arms[0]
        return not_none(gr.candidate_metadata_by_arm_signature).get(arm.signature)
