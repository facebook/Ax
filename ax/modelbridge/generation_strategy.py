#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from inspect import signature
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

import pandas as pd
from ax.core.base import Base
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models, get_model_from_generator_run
from ax.utils.common.kwargs import consolidate_kwargs, get_function_argument_names
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger = get_logger(__name__)


TModelFactory = Callable[..., ModelBridge]
MAX_CONDITIONS_GENERATED = 10000


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


class GenerationStep(NamedTuple):
    """One step in the generation strategy, corresponds to a single model.
    Describes the model, how many trials will be generated with this model, what
    minimum number of observations is required to proceed to the next model, etc.

    Model can be specified either from the model registry
    (`ax.modelbridge.registry.Models` or using a callable model constructor. Only
    models from the registry can be saved, and thus optimization can only be
    resumed if interrupted when using models from the registry.
    """

    model: Union[Models, Callable[..., ModelBridge]]
    num_trials: int
    min_trials_observed: int = 0
    max_parallelism: Optional[int] = None
    enforce_num_trials: bool = True
    # Kwargs to pass into the Models constructor (or factory function).
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass into the Model's `.gen` function.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    # pyre-ignore[15]: inconsistent override
    index: int = -1  # Index of this step, set internally.

    @property
    def model_name(self) -> str:
        # Model can be defined as member of Models enum or as a factory function,
        # so we use Models member (str) value if former and function name if latter.
        if isinstance(self.model, Models):
            return checked_cast(str, checked_cast(Models, self.model).value)
        if callable(self.model):
            return self.model.__name__  # pyre-fixme[16]: union has no attr __name__
        raise TypeError(  # pragma: no cover
            "`model` was not a member of `Models` or a callable."
        )


class MaxParallelismReachedException(Exception):
    """Special exception indicating that maximum number of trials running in
    parallel set on a given step (as `GenerationStep.max_parallelism`) has been
    reached. Upon getting this exception, users should wait until more trials
    are completed with data, to generate new trials.
    """

    def __init__(self, step: GenerationStep, num_running: int) -> None:
        super().__init__(
            f"Maximum parallelism for generation step #{step.index} ({step.model_name})"
            f" has been reached: {num_running} trials are currently 'running'. Some "
            "trials need to be completed before more trials can be generated. See "
            "https://ax.dev/docs/bayesopt.html to understand why limited parallelism "
            "improves performance of Bayesian optimization."
        )


class GenerationStrategy(Base):
    """GenerationStrategy describes which model should be used to generate new
    points for which trials, enabling and automating use of different models
    throughout the optimization process. For instance, it allows to use one
    model for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of models to generate an arbitrary numbers of trials
    described in the `trials_per_model` argument.
    """

    _name: Optional[str]
    _steps: List[GenerationStep]
    _model: Optional[ModelBridge]  # Current model.
    _curr: GenerationStep  # Current step in the strategy.
    # Whether all models in this GS are in Models registry enum.
    _uses_registered_models: bool
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment]
    _db_id: Optional[int]  # Used when storing to DB.

    def __init__(self, steps: List[GenerationStep], name: Optional[str] = None) -> None:
        assert isinstance(steps, list) and all(
            isinstance(s, GenerationStep) for s in steps
        ), "Steps must be a GenerationStep list."
        self._db_id = None
        self._name = name
        self._steps = steps
        self._uses_registered_models = True
        self._generator_runs = []
        self._model = None
        self._experiment = None
        for idx, step in enumerate(self._steps):
            if step.num_trials == -1:
                if idx < len(self._steps) - 1:
                    raise ValueError(  # pragma: no cover
                        "Only last step in generation strategy can have `num_trials` "
                        "set to -1 to indicate that the model in the step should "
                        "be used to generate new trials indefinitely."
                    )
            elif step.num_trials < 1:  # pragma: no cover
                raise ValueError("`num_trials` must be positive or -1 for all models.")
            self._steps[idx] = step._replace(index=idx)
            if not isinstance(step.model, Models):
                self._uses_registered_models = False
        if not self._uses_registered_models:
            logger.info(
                "Using model via callable function, "
                "so optimization is not resumable if interrupted."
            )
        self._curr = steps[0]

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps."""
        if self._name is not None:
            return not_none(self._name)

        factory_names = (step.model_name for step in self._steps)
        # Trim the "get_" beginning of the factory function if it's there.
        factory_names = (n[4:] if n[:4] == "get_" else n for n in factory_names)
        self._name = "+".join(factory_names)
        return not_none(self._name)

    @property
    def model_transitions(self) -> List[int]:
        """List of trial indices where a transition happened from one model to
        another."""
        gen_changes = [step.num_trials for step in self._steps]
        return [sum(gen_changes[: i + 1]) for i in range(len(gen_changes))][:-1]

    @property
    def model(self) -> Optional[ModelBridge]:
        """Current model in this strategy. Returns None if no model has been set
        yet (i.e., if no generator runs have been produced from this GS).
        """
        return self._model  # pragma: no cover

    @property
    def experiment(self) -> Experiment:
        """Experiment, currently set on this generation strategy."""
        if self._experiment is None:  # pragma: no cover
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
        else:  # pragma: no cover
            raise ValueError(
                "This generation strategy has been used for experiment "
                f"{self.experiment._name} so far; cannot reset experiment"
                f" to {experiment._name}. If this is a new optimization, "
                "a new generation strategy should be created instead."
            )

    @property
    def uses_non_registered_models(self) -> bool:
        """Whether this generation strategy involves models that are not
        registered and therefore cannot be stored."""
        return not self._uses_registered_models

    @property
    def last_generator_run(self) -> Optional[GeneratorRun]:
        """Latest generator run produced by this generation strategy.
        Returns None if no generator runs have been produced yet.
        """
        # Used to restore current model when decoding a serialized GS.
        return self._generator_runs[-1] if self._generator_runs else None

    @property
    def trial_indices_by_step(self) -> Dict[int, List[int]]:
        """Find trials in experiment that are not mapped to a generation step yet
        and add them to the mapping of trials by generation step.
        """
        trial_indices_by_step = defaultdict(list)
        for trial_index, trial in self.experiment.trials.items():
            if (
                trial._generation_step_index is not None
                and trial._generation_step_index <= self._curr.index
            ):
                trial_indices_by_step[trial._generation_step_index].append(trial_index)

        return trial_indices_by_step

    @property
    def trials_as_df(self) -> Optional[pd.DataFrame]:
        """Puts information on individual trials into a data frame for easy
        viewing. For example:
        Gen. Step | Model | Trial Index | Trial Status | Arm Parameterizations
        0         | Sobol | 0           | RUNNING      | {"0_0":{"x":9.17...}}
        """
        logger.info(
            "Note that parameter values in dataframe are rounded to 2 decimal "
            "points; the values in the dataframe are thus not the exact ones "
            "suggested by Ax in trials."
        )
        if self._experiment is None or all(
            len(l) == 0 for l in self.trial_indices_by_step.values()
        ):
            return None
        records = [
            {
                "Generation Step": step_idx,
                "Generation Model": self._steps[step_idx].model_name,
                "Trial Index": trial_idx,
                "Trial Status": self.experiment.trials[trial_idx].status.name,
                "Arm Parameterizations": {
                    arm.name: _round_floats_for_logging(arm.parameters)
                    for arm in self.experiment.trials[trial_idx].arms
                },
            }
            for step_idx, trials in self.trial_indices_by_step.items()
            for trial_idx in trials
        ]
        return pd.DataFrame.from_records(records).reindex(
            columns=[
                "Generation Step",
                "Generation Model",
                "Trial Index",
                "Trial Status",
                "Arm Parameterizations",
            ]
        )

    @property
    def num_running_trials_for_current_step(self) -> int:
        """Number of trials in status `RUNNING` for the current generation step
        of this strategy.
        """
        num_running = 0
        for trial in self.experiment.trials.values():
            if (
                trial._generation_step_index == self._curr.index
                and trial.status.is_running
            ):
                num_running += 1
        return num_running

    def gen(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        self.experiment = experiment
        self._set_model(experiment=experiment, data=data or experiment.fetch_data())
        max_parallelism = self._curr.max_parallelism
        num_running = self.num_running_trials_for_current_step
        if max_parallelism is not None and num_running >= max_parallelism:
            raise MaxParallelismReachedException(
                step=self._curr, num_running=num_running
            )
        model = not_none(self.model)
        generator_run = model.gen(
            n=n,
            **consolidate_kwargs(
                kwargs_iterable=[self._curr.model_gen_kwargs, kwargs],
                keywords=get_function_argument_names(model.gen),
            ),
        )
        generator_run._generation_step_index = self._curr.index
        self._generator_runs.append(generator_run)
        return generator_run

    def clone_reset(self) -> "GenerationStrategy":
        """Copy this generation strategy without it's state."""
        return GenerationStrategy(name=self.name, steps=self._steps)

    def __repr__(self) -> str:
        """String representation of this generation strategy."""
        repr = f"GenerationStrategy(name='{self.name}', steps=["
        remaining_trials = "subsequent" if len(self._steps) > 1 else "all"
        for step in self._steps:
            num_trials = (
                f"{step.num_trials}" if step.num_trials != -1 else remaining_trials
            )
            if isinstance(step.model, Models):
                # pyre-ignore[16]: `Union` has no attribute `value`.
                repr += f"{step.model.value} for {num_trials} trials, "
        repr = repr[:-2]
        repr += f"])"
        return repr

    # ------------------------- Model selection logic helpers. -------------------------

    def _set_model(self, experiment: Experiment, data: Data) -> None:
        model_state = {}
        lgr = self.last_generator_run
        if lgr is not None and lgr._model_state_after_gen is not None:
            model_state = not_none(lgr._model_state_after_gen)

        if self._curr.num_trials == -1:  # Unlimited trials, just use curr. model.
            self._set_current_model(experiment=experiment, data=data, **model_state)
            return

        # Not unlimited trials => determine whether to transition to next model.
        step_trials = self.trial_indices_by_step[self._curr.index]
        all_trials = experiment.trials
        completed = sum(1 for i in step_trials if all_trials[i].completed_successfully)
        did_not_complete = sum(1 for i in step_trials if all_trials[i].did_not_complete)

        enough_observed = completed >= self._curr.min_trials_observed
        enough_generated = len(step_trials) - did_not_complete >= self._curr.num_trials

        # Check that minimum observed_trials is satisfied if it's enforced.
        if self._curr.enforce_num_trials and enough_generated and not enough_observed:
            raise DataRequiredError(
                "All trials for current model have been generated, but not enough "
                "data has been observed to fit next model. Try again when more data "
                "are available."
            )

        if enough_generated and enough_observed:
            # Change to the next model.
            if len(self._steps) == self._curr.index + 1:
                raise ValueError(f"Generation strategy {self.name} is completed.")
            self._curr = self._steps[self._curr.index + 1]
            self._set_current_model(experiment=experiment, data=data)
        else:
            # Continue generating from the current model.
            self._set_current_model(experiment=experiment, data=data, **model_state)

    def _set_current_model(
        self, experiment: Experiment, data: Data, **kwargs: Any
    ) -> None:
        """Instantiate the current model with all available data.
        """
        kwargs = kwargs or {}
        if isinstance(self._curr.model, Models):
            self._set_current_model_from_models_enum(
                experiment=experiment, data=data, **kwargs
            )
        else:
            # If model was not specified as Models member, it was specified as a
            # factory function.
            self._set_current_model_from_factory_function(
                experiment=experiment, data=data, **kwargs
            )

    def _set_current_model_from_models_enum(
        self, experiment: Experiment, data: Data, **kwargs: Any
    ) -> None:
        """Instantiate the current model, provided through a Models enum member
        function, with all available data."""
        self._model = self._curr.model(
            experiment=experiment,
            data=data,
            search_space=experiment.search_space,
            **(self._curr.model_kwargs or {}),
            **kwargs,
        )

    def _set_current_model_from_factory_function(
        self, experiment: Experiment, data: Data, **kwargs: Any
    ) -> None:
        """Instantiate the current model, provided through a callable factory
        function, with all available data."""
        model = self._curr.model
        assert not isinstance(model, Models) and callable(model)
        self._model = self._curr.model(
            **_filter_kwargs(
                self._curr.model,
                experiment=experiment,
                data=data,
                search_space=experiment.search_space,
                **(self._curr.model_kwargs or {}),
                **kwargs,
            )
        )

    def _restore_model_from_generator_run(
        self, models_enum: Optional[Type[Models]] = None
    ) -> None:
        generator_run = self.last_generator_run
        if generator_run is None:
            raise ValueError("No generator run was stored on generation strategy.")
        if self._experiment is None:  # pragma: no cover
            raise ValueError("No experiment was set on this generation strategy.")
        self._model = get_model_from_generator_run(
            generator_run=generator_run,
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            models_enum=models_enum,
        )
