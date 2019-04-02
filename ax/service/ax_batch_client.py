#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.core.types import TModelPredictArm, TParameterization
from ax.service.ax_client import AxClient
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none


class AxBatchClient(AxClient):
    """An extension of AxClient, capable of handling `BatchTrial`."""

    def get_next_batch_trial(self, n: int) -> Tuple[Dict[str, TParameterization], int]:
        """
        Generate trial with the next sets of parameters to try in the optimization
        process.

        Args:
            n: Number of candidates to generate.

        Returns:
            Tuple of trial parameterization dict, trial index
        """
        # Potentially move this into log_data to save latency on this call
        trial = self._suggest_new_batch_trial(n)
        self._save_experiment_if_possible()
        return ({arm.name: arm.params for arm in trial.arms}, trial.index)

    def complete_batch_trial(
        self,
        trial_index: int,
        # `raw_data` format: {arm_name -> {metric_name -> (mean, standard error)}}
        raw_data: Dict[str, Dict[str, Tuple[float, float]]],
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
        self._save_experiment_if_possible()

    # TODO: this is currently only compatible with some models. T42389552
    @copy_doc(AxClient.get_best_parameters)
    def get_best_parameters(
        self
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
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
                best_arm, best_arm_predictions = gr.best_arm_predictions
                return best_arm.params, best_arm_predictions

        return None

    # ---------------------- Private helper methods. ---------------------

    def _suggest_new_batch_trial(self, n: int) -> BatchTrial:
        """
        Suggest new set of candidates for this experiment.

        Uses data attached to the experiment and the given generator.

        Args:
            n: Number of candidates to generate.

        Returns:
            BatchTrial with candidates.
        """
        generator = not_none(self.generation_strategy).get_model(
            self.experiment, data=self.experiment.eval()
        )
        generator_run = generator.gen(n=n)
        return self.experiment.new_batch_trial().add_generator_run(generator_run)
