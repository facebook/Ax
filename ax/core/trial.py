#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from functools import partial

from logging import Logger

from typing import Any, TYPE_CHECKING

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, immutable_once_run
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.types import TCandidateMetadata, TEvaluationOutcome
from ax.exceptions.core import UnsupportedError
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from pyre_extensions import none_throws, override

logger: Logger = get_logger(__name__)

ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6

# pyre-fixme[5]: Global expression must be annotated.
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


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
        generator_run: GeneratorRun | None = None,
        trial_type: str | None = None,
        ttl_seconds: int | None = None,
        index: int | None = None,
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
    def generator_run(self) -> GeneratorRun | None:
        """Generator run attached to this trial."""
        return self._generator_run

    # pyre-ignore[6]: T77111662.
    @copy_doc(BaseTrial.generator_runs)
    @property
    def generator_runs(self) -> list[GeneratorRun]:
        gr = self._generator_run
        return [gr] if gr is not None else []

    @property
    def arm(self) -> Arm | None:
        """The arm associated with this batch."""
        if self.generator_run is None:
            return None

        generator_run = none_throws(self.generator_run)
        if len(generator_run.arms) == 0:
            return None
        elif len(generator_run.arms) > 1:
            raise ValueError(
                "Generator run associated with this trial included multiple "
                "arms, but trial expects only one."
            )
        return generator_run.arms[0]

    @immutable_once_run
    def add_arm(
        self, arm: Arm, candidate_metadata: dict[str, Any] | None = None
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
                candidate_metadata_by_arm_signature=(
                    None
                    if candidate_metadata is None
                    else {arm.signature: candidate_metadata.copy()}
                ),
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
    def arms(self) -> list[Arm]:
        """All arms attached to this trial.

        Returns:
            arms: list of a single arm
                attached to this trial if there is one, else None.
        """
        return [self.arm] if self.arm is not None else []

    @property
    def arms_by_name(self) -> dict[str, Arm]:
        """Dictionary of all arms attached to this trial with their names
        as keys.

        Returns:
            arms: dictionary of a single
                arm name to arm if one is attached to this trial,
                else None.
        """
        return {self.arm.name: self.arm} if self.arm is not None else {}

    @property
    def abandoned_arms(self) -> list[Arm]:
        """Abandoned arms attached to this trial."""
        return (
            [none_throws(self.arm)]
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
            raise ValueError(
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
        except IndexError:
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
    ) -> dict[str, TCandidateMetadata]:
        """Retrieves candidate metadata from the generator run on this
        batch trial in the form of { arm name -> candidate metadata} mapping.
        """

        gr = self.generator_run
        if gr is None or gr.candidate_metadata_by_arm_signature is None:
            return {}

        cand_metadata = none_throws(gr.candidate_metadata_by_arm_signature)
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
        return none_throws(gr.candidate_metadata_by_arm_signature).get(arm.signature)

    def validate_data_for_trial(self, data: Data) -> None:
        """Utility method to validate data before further processing."""
        for metric_name in data.df["metric_name"].values:
            if metric_name not in self.experiment.metrics:
                logger.info(
                    f"Data was logged for metric {metric_name} that was not yet "
                    "tracked on the experiment. Please specify `tracking_metric_"
                    "names` argument in AxClient.create_experiment to add tracking "
                    "metrics to the experiment. Without those, all data users "
                    "specify is still attached to the experiment, but will not be "
                    "fetched in `experiment.fetch_data()`, but you can still use "
                    "`experiment.lookup_data_for_trial` to get all attached data."
                )

    def update_trial_data(
        self,
        raw_data: TEvaluationOutcome,
        metadata: dict[str, str | int] | None = None,
        sample_size: int | None = None,
        combine_with_last_data: bool = False,
    ) -> str:
        """Utility method that attaches data to a trial and
        returns an update message.

        Args:
            raw_data: Evaluation data for the trial. Can be a mapping from
                metric name to a tuple of mean and SEM, just a tuple of mean and
                SEM if only one metric in optimization, or just the mean if SEM is
                unknown (then Ax will infer observation noise level).
                Can also be a list of (fidelities, mapping from
                metric name to a tuple of mean and SEM).
            metadata: Additional metadata to track about this run, optional.
            sample_size: Number of samples collected for the underlying arm,
                optional.
            combine_with_last_data: Whether to combine the given data with the
                data that was previously attached to the trial. See
                `Experiment.attach_data` for a detailed explanation.

        Returns:
            A string message summarizing the update.
        """
        arm_name = none_throws(self.arm).name
        sample_sizes = {arm_name: sample_size} if sample_size else {}
        raw_data_by_arm = {arm_name: raw_data}

        evaluations, data = self._make_evaluations_and_data(
            raw_data=raw_data_by_arm,
            metadata=metadata,
            sample_sizes=sample_sizes,
        )

        self.validate_data_for_trial(data=data)
        self.update_run_metadata(metadata=metadata or {})

        self.experiment.attach_data(
            data=data, combine_with_last_data=combine_with_last_data
        )

        return str(
            round_floats_for_logging(item=evaluations[next(iter(evaluations.keys()))])
        )

    def clone_to(
        self,
        experiment: core.experiment.Experiment | None = None,
        clear_trial_type: bool = False,
    ) -> Trial:
        """Clone the trial and attach it to the specified experiment.
        If no experiment is provided, the original experiment will be used.

        Args:
            experiment: The experiment to which the cloned trial will belong.
                If unspecified, uses the current experiment.
            clear_trial_type: If True, all cloned trials on the cloned experiment have
                `trial_type` set to `None`.

        Returns:
            A new instance of the trial.
        """
        experiment = self._experiment if experiment is None else experiment
        new_trial = experiment.new_trial(
            ttl_seconds=self.ttl_seconds,
            trial_type=None if clear_trial_type else self.trial_type,
        )
        if self.generator_run is not None:
            new_trial.add_generator_run(self.generator_run.clone())
        self._update_trial_attrs_on_clone(new_trial=new_trial)
        return new_trial

    @override
    def _raise_cant_attach_if_completed(self) -> None:
        """
        Helper method used by `validate_can_attach_data` to raise an error if
        the user tries to attach data to a completed trial. Subclasses such as
        `Trial` override this by suggesting a remediation.
        """
        raise UnsupportedError(
            f"Trial {self.index} has already been completed with data. "
            "To add more data to it (for example, for a different metric), "
            f"use `{self.__class__.__name__}.update_trial_data()`."
        )
