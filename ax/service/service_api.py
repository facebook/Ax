#!/usr/bin/env python3

from typing import Dict, NamedTuple, Optional, Tuple

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.core.types.types import TEvaluationOutcome, TModelPredictArm, TParameterization
from ax.modelbridge.generation_strategy import GenerationStrategy
# pyre-fixme[21]: Could not find `base_decoder`.
from ax.storage.sqa_store.base_decoder import Decoder
# pyre-fixme[21]: Could not find `base_encoder`.
from ax.storage.sqa_store.base_encoder import Encoder
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import _load_experiment
from ax.storage.sqa_store.save import _save_experiment
from libfb.py import db_locator


# DO NOT OPEN SOURCE UNTIL T40274901 is resolved2


class DBSettings(NamedTuple):
    """
    Defines behavior for loading/saving experiment to/from db.
    """

    decoder: Decoder
    encoder: Encoder
    tier_name: str


def initialize_db(db_settings: DBSettings) -> None:
    """
    Initialize db connections given settings.

    Args:
        db_settings: Optional[DBSettings] = None
    Returns:
        None
    """
    if db_settings.tier_name is not None:
        locator = db_locator.Locator(tier_name=db_settings.tier_name)
        init_engine_and_session_factory(creator=locator.get_creator())


def load_experiment(name: str, db_settings: DBSettings) -> Experiment:
    """
    Load experiment from the db. Service API only supports Experiment.

    Args:
        name: Experiment name.
        db_settings: Specifies decoder and xdb tier where experiment is stored.

    Returns:
        Experiment object.
    """
    initialize_db(db_settings)
    experiment = _load_experiment(name, db_settings.decoder)
    if not isinstance(experiment, Experiment):
        raise ValueError("Service API only supports Experiment")
    return experiment


def save_experiment(experiment: Experiment, db_settings: DBSettings) -> None:
    """
    Save experiment to db.

    Args:
        experiment: Experiment object.
        db_settings: Specifies decoder and xdb tier where experiment is stored.
    """
    initialize_db(db_settings)
    _save_experiment(experiment, db_settings.encoder)


class AELoopHandler:
    """
    Convenience handler for managing the experimentation loop process.
    """

    def __init__(
        self,
        generation_strategy: GenerationStrategy,
        db_settings: Optional[DBSettings] = None,
    ) -> None:
        self.generation_strategy = generation_strategy
        self.db_settings = db_settings
        self._experiment: Optional[SimpleExperiment] = None

    @property
    def experiment(self) -> SimpleExperiment:
        """Returns the experiment set on this loop handler"""
        if self._experiment is None:
            raise ValueError(
                "Experiment not set on loop handler. Must first "
                "call load_experiment or create_experiment to use handler functions."
            )
        return self._experiment

    # TODO (T40274901): Make a simpler API for doing this
    def create_experiment(self, experiment: SimpleExperiment) -> None:
        """Create and save a new experiment.

        Args:
            experiment: Experiment object.

        Returns:
            Experiment object.
        """
        if self.db_settings:
            save_experiment(experiment, self.db_settings)
        self._experiment = experiment

    def load_experiment(self, experiment_name: str) -> None:
        """Load an existing experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Experiment object.
        """
        if not self.db_settings:
            raise ValueError("Need to set db_settings on handler to load experiment.")

        # TODO (T40274901) be able to save and load simple experiment
        # pyre-ignore[8]: Should be fixed in above task
        self._experiment = load_experiment(experiment_name, self.db_settings)

    def _suggest_new_trial(self, n: int = 1) -> BaseTrial:
        """
        Suggest new candidate for this experiment.

        Uses data attached to the experiment and the given generator.

        Args:
            n: Number of candidates to generate.

        Returns:
            Trial with candidates.
        """
        generator = self.generation_strategy.get_model(
            self.experiment, data=self.experiment.eval()
        )
        generator_run = generator.gen(n=n)
        if n == 1:
            return self.experiment.new_trial(generator_run=generator_run)
        else:
            return self.experiment.new_batch_trial().add_generator_run(generator_run)

    def log_failure(
        self, trial_index: int, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark that the given trial has failed while running.

        Args:
            trial_index: Index of trial within the experiment.
            metadata: Additional metadata to track about this run.
        """
        trial = self.experiment.trials[trial_index]
        trial._status = TrialStatus.FAILED
        if metadata is not None:
            trial._run_metadata = metadata

        if self.db_settings is not None:
            save_experiment(self.experiment, self.db_settings)

    def create_trial(self, params: TParameterization) -> BaseTrial:
        """Create a new trial on the experiment with the given parameterization.

        Args:
            params: Parameterization of the new trial.

        Returns:
            Newly created trial.
        """
        new_trial = self.experiment.new_trial().add_arm(Arm(params=params))
        if self.db_settings is not None:
            save_experiment(self.experiment, self.db_settings)

        return new_trial

    def log_data(
        self,
        trial_index: int,
        raw_data: Dict[str, TEvaluationOutcome],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Stores given metric outcomes and metadata on trial.

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Map from condition name to metric outcomes.
            metadata: Additional metadata to track about this run.
        """
        trial = self.experiment.trials[trial_index]

        trial._status = TrialStatus.COMPLETED
        if metadata is not None:
            trial._run_metadata = metadata

        data = SimpleExperiment.data_from_evaluations(raw_data, trial.index)
        self.experiment.attach_data(data)

        if self.db_settings is not None:
            save_experiment(self.experiment, self.db_settings)

    def should_stop_early(
        self, params: TParameterization, data: TEvaluationOutcome
    ) -> bool:
        """Whether to stop the given parameterization given early data."""
        raise NotImplementedError

    def get_next_trial(self, n: int = 1) -> BaseTrial:
        """
        Generate trial with the next set of parameters to try in the iteration process.

        Args:
            n: Number of candidates to generate.

        Returns:
            The new experimental trial.
        """
        # Potentially move this into log_data to save latency on this call
        trial = self._suggest_new_trial(n)
        if self.db_settings is not None:
            save_experiment(self.experiment, self.db_settings)
        return trial

    def get_optimized_arm(self) -> Optional[Tuple[Arm, Optional[TModelPredictArm]]]:
        """
        Return the best set of parameters the experiment has knowledge of.

        If experiment is in the optimization phase, return the best point
        determined by the model used in the latest optimization round.

        Otherwise return none.

        Returns:
            Tuple of (best arm, model predictions for best arm). None if no data.
        """

        # Find latest trial which has a generator_run attached and get its predictions
        for _, trial in sorted(
            list(self.experiment.trials.items()), key=lambda x: x[0], reverse=True
        ):
            gr = None
            if isinstance(trial, Trial):
                gr = trial.generator_run
            elif isinstance(trial, BatchTrial):
                if len(trial.generator_run_structs) > 0:
                    # In theory batch_trial can have >1 gr, grab the first
                    gr = trial.generator_run_structs[0].generator_run

            if gr is not None and gr.best_arm_predictions is not None:
                return gr.best_arm_predictions

        # TODO[speculative] Maybe pick best of existing data.
        return None

    def get_report(self) -> str:
        """Returns HTML of a generated report containing vizualizations."""
        raise NotImplementedError
