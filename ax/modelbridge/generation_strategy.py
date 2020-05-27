#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Type, Union

import pandas as pd
from ax.core.base import Base
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.exceptions.core import DataRequiredError
from ax.exceptions.generation_strategy import (
    GenerationStrategyCompleted,
    MaxParallelismReachedException,
)
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models, get_model_from_generator_run
from ax.utils.common.equality import equality_typechecker, object_attribute_dicts_equal
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

    NOTE: Model can be specified either from the model registry
    (`ax.modelbridge.registry.Models` or using a callable model constructor. Only
    models from the registry can be saved, and thus optimization can only be
    resumed if interrupted when using models from the registry.

    Args:
        model: A member of `Models` enum or a callable returning an instance of
            `ModelBridge` with an instantiated underlying `Model`. Refer to
            `ax/modelbridge/factory.py` for examples of such callables.
        num_trials: How many trials to generate with the model from this step.
            If set to -1, trials will continue to be generated from this model
            as long as `generation_strategy.gen` is called (available only for
            the last of the generation steps).
        min_trials_observed: How many trials must be completed before the
            generation strategy can proceed to the next step. Defaults to 0.
            If `num_trials` of a given step have been generated but `min_trials_
            observed` have not been completed, a call to `generation_strategy.gen`
            will fail with a `DataRequiredError`.
        max_parallelism: How many trials generated in the course of this step are
            allowed to be run (i.e. have `trial.status` of `RUNNING`) simultaneously.
            If `max_parallelism` trials from this step are already running, a call
            to `generation_strategy.gen` will fail with a `MaxParallelismReached
            Exception`, indicating that more trials need to be completed before
            generating and running next trials.
        use_update: Whether to use `model_bridge.update` instead or reinstantiating
            model + bridge on every call to `gen` within a single generation step.
            NOTE: use of `update` on stateful models that do not implement `_get_state`
            may result in inability to correctly resume a generation strategy from
            a serialized state.
        enforce_num_trials: Whether to enforce that only `num_trials` are generated
            from the given step. If False and `num_trials` have been generated, but
            `min_trials_observed` have not been completed, `generation_strategy.gen`
            will continue generating trials from the current step, exceeding `num_
            trials` for it. Allows to avoid `DataRequiredError`, but delays
            proceeding to next generation step.
        model_kwargs: Dictionary of kwargs to pass into the model constructor on
            instantiation. E.g. if `model` is `Models.SOBOL`, kwargs will be applied
            as `Models.SOBOL(**model_kwargs)`; if `model` is `get_sobol`, `get_sobol(
            **model_kwargs)`. NOTE: if generation strategy is interrupted and
            resumed from a stored snapshot and its last used model has state saved on
            its generator runs, `model_kwargs` is updated with the state dict of the
            model, retrieved from the last generator run of this generation strategy.
        model_gen_kwargs: Each call to `generation_strategy.gen` performs a call to the
            step's model's `gen` under the hood; `model_gen_kwargs` will be passed to
            the model's `gen` like so: `model.gen(**model_gen_kwargs)`.
        index: Index of this generation step, for use internally in `Generation
            Strategy`. Do not assign as it will be reassigned when instantiating
            `GenerationStrategy` with a list of its steps.

    """

    model: Union[Models, Callable[..., ModelBridge]]
    num_trials: int
    min_trials_observed: int = 0
    max_parallelism: Optional[int] = None
    use_update: bool = False
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

    @equality_typechecker
    def __eq__(self, other: GenerationStep) -> bool:
        return object_attribute_dicts_equal(
            one_dict=self._asdict(), other_dict=other._asdict()
        )


class GenerationStrategy(Base):
    """GenerationStrategy describes which model should be used to generate new
    points for which trials, enabling and automating use of different models
    throughout the optimization process. For instance, it allows to use one
    model for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of models to generate an arbitrary numbers of trials
    described in the `trials_per_model` argument.

    Args:
        steps: A list of `GenerationStep` describing steps of this strategy.
        name: An optional name for this generaiton strategy. If not specified,
            strategy's name will be names of its steps' models joined with '+'.
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
        self._seen_trial_indices_by_status = None

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
    def trial_indices_by_step(self) -> Dict[int, Set[int]]:
        """Find trials in experiment that are not mapped to a generation step yet
        and add them to the mapping of trials by generation step.
        """
        trial_indices_by_step = defaultdict(set)
        for trial_index, trial in self.experiment.trials.items():
            if (
                trial._generation_step_index is not None
                and trial._generation_step_index <= self._curr.index
            ):
                trial_indices_by_step[trial._generation_step_index].add(trial_index)

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
            len(trials) == 0 for trials in self.trial_indices_by_step.values()
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
            # pyre-fixme[10]: Name `trials` is used but not defined.
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
        """Produce the next points in the experiment. Additional kwargs passed to
        this method are propagated directly to the underlying model's `gen`, along
        with the `model_gen_kwargs` set on the current generation step.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment` if
                `use_update` is False and only the new data since the last call to
                this method if `use_update` is True.
            n: Integer representing how many arms should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the `n` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from `n`.
        """
        self.experiment = experiment
        self._set_or_update_model(data=data)
        self._seen_trial_indices_by_status = deepcopy(
            experiment.trial_indices_by_status
        )
        max_parallelism = self._curr.max_parallelism
        num_running = self.num_running_trials_for_current_step
        if max_parallelism is not None and num_running >= max_parallelism:
            raise MaxParallelismReachedException(
                step_index=self._curr.index,
                model_name=self._curr.model_name,
                num_running=num_running,
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
                repr += f"{step.model.value} for {num_trials} trials, "
        repr = repr[:-2]
        repr += "])"
        return repr

    # ------------------------- Model selection logic helpers. -------------------------

    def _set_or_update_model(self, data: Optional[Data]) -> None:
        if self._curr.num_trials == -1:  # Unlimited trials, just use curr. model.
            self._set_or_update_current_model(data=data)
            return

        # Not unlimited trials => determine whether to transition to next model.
        step_trials = self.trial_indices_by_step[self._curr.index]
        by_status = self.experiment.trial_indices_by_status
        num_completed = len(step_trials.intersection(by_status[TrialStatus.COMPLETED]))
        # Number of trials that will not be `COMPLETED`, used to avoid counting
        # unsuccessfully terminated trials against the number of generated trials
        # during determination of whether enough trials have been generated and
        # completed to proceed to the next generation step.
        num_will_not_complete = len(
            step_trials.intersection(
                by_status[TrialStatus.FAILED].union(by_status[TrialStatus.ABANDONED])
            )
        )

        enough_observed = num_completed >= self._curr.min_trials_observed
        enough_generated = (
            len(step_trials) - num_will_not_complete >= self._curr.num_trials
        )

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
                raise GenerationStrategyCompleted(
                    f"Generation strategy {self} generated all the trials as "
                    "specified in its steps."
                )
            self._curr = self._steps[self._curr.index + 1]
            # This is the first time this step's model is initialized, so we don't
            # try to `update` it but rather initialize with all the data even if
            # `use_update` is true for the now-current generation step.
            self._set_current_model(data=data)
        else:
            # Continue generating from the current model.
            self._set_or_update_current_model(data=data)

    def _set_or_update_current_model(self, data: Optional[Data]) -> None:
        if self._model is not None and self._curr.use_update:
            self._update_current_model(data=data)
        else:
            self._set_current_model(data=data)

    def _set_current_model(self, data: Optional[Data]) -> None:
        """Instantiate the current model with all available data.
        """
        model_kwargs = self._curr.model_kwargs or {}

        # If last generator run's index matches the current step, extract
        # model state from last generator run and pass it to the model
        # being instantiated in this function.
        lgr = self.last_generator_run
        if (
            lgr is not None
            and lgr._generation_step_index == self._curr.index
            and lgr._model_state_after_gen
        ):
            serialized_model_state = not_none(lgr._model_state_after_gen)
            model_state = not_none(self.model)._deserialize_model_state(
                serialized_model_state
            )
            model_kwargs.update(model_state)

        # TODO[T65857344]: move from fetching all data to using cached data
        if data is None:
            if self._curr.use_update:
                # If the new step is using `update`, it's important to instantiate
                # the model with data for completed trials only, so later we can
                # update it with data for new trials as they become completed.
                # `experiment.fetch_data` can fetch all available data, including
                # for non-completed trials (depending on how the experiment's metrics
                # implement `fetch_experiment_data`). We avoid fetching data for
                # trials with statuses other than `COMPLETED`, by fetching specifically
                # for `COMPLETED` trials.
                data = self.experiment.fetch_trials_data(
                    self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
                )
            else:
                data = self.experiment.fetch_data()
        if isinstance(self._curr.model, Models):
            self._set_current_model_from_models_enum(data=data, **model_kwargs)
        else:
            # If model was not specified as Models member, it was specified as a
            # factory function.
            self._set_current_model_from_factory_function(data=data, **model_kwargs)

    def _update_current_model(self, data: Optional[Data]) -> None:
        """Update the current model with new data (data for trials that have been
        completed since the last call to `GenerationStrategy.gen`).
        """
        if self._model is None:
            raise ValueError("Cannot update if no model instantiated.")
        # Should only pass data that is new since last call to `gen`, to the
        # underlying model's `update`.
        newly_completed_trials = self._find_trials_completed_since_last_gen()
        if len(newly_completed_trials) == 0:
            logger.info("There were no newly completed trials since last model update.")
            return
        if data is None:
            new_data = self.experiment.fetch_trials_data(
                trial_indices=newly_completed_trials
            )
            if new_data.df.empty:
                logger.info("Skipping model update as there is no new data.")
                return
        elif data.df.empty:
            logger.info("Skipping model update as data supplied to `gen` is empty.")
            return
        else:
            new_data = Data(
                df=data.df[data.df.trial_index.isin(newly_completed_trials)]
            )
        # We definitely have non-empty new data by now.
        trial_indices_in_new_data = sorted(new_data.df["trial_index"].unique())
        logger.info(f"Updating model with data for trials: {trial_indices_in_new_data}")
        not_none(self._model).update(experiment=self.experiment, new_data=new_data)

    def _set_current_model_from_models_enum(self, data: Data, **kwargs: Any) -> None:
        """Instantiate the current model, provided through a Models enum member
        function, with the provided data and kwargs."""
        self._model = self._curr.model(experiment=self.experiment, data=data, **kwargs)

    def _set_current_model_from_factory_function(
        self, data: Data, **kwargs: Any
    ) -> None:
        """Instantiate the current model, provided through a callable factory
        function, with the provided data and kwargs."""
        model = self._curr.model
        assert not isinstance(model, Models) and callable(model)
        self._model = self._curr.model(
            **_filter_kwargs(
                self._curr.model,
                experiment=self.experiment,
                data=data,
                # Some factory functions (like `get_sobol`) require search space
                # instead of experiment.
                search_space=self.experiment.search_space,
                **kwargs,
            )
        )

    def _restore_model_from_generator_run(
        self, models_enum: Optional[Type[Models]] = None
    ) -> None:
        """Reinstantiates the most recent model on this generation strategy
        from the last generator run it produced.

        NOTE: Uses model and model bridge kwargs stored on the generator run, as well
        as the model state attributes stored on the generator run.
        """
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

    def _find_trials_completed_since_last_gen(self) -> Set[int]:
        """Retrieves indices of trials that have been completed or updated with data
        since the last call to `GenerationStrategy.gen`.
        """
        completed_now = self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        if self._seen_trial_indices_by_status is None:
            return completed_now

        completed_before = not_none(self._seen_trial_indices_by_status)[
            TrialStatus.COMPLETED
        ]
        return completed_now.difference(completed_before)

    def _register_trial_data_update(self, trial: BaseTrial, data: Data) -> None:
        """Registers that a given trial has new data even though it's a trial that has
        been completed before. Useful only for generation steps that have `use_update=
        True`, as the information registered by this function is used for identifying
        new data since last call to `GenerationStrategy.gen`.
        """
        # TODO[T65857344]: store information about trial update to pass with `new_data`
        # to `model_update`. This information does not need to be stored, since when
        # restoring generation strategy from serialized form, all data will is
        # refetched and the underlying model is re-fit.
        if any(s.use_update for s in self._steps):
            raise NotImplementedError(
                "Updating completed trials with new data is not yet supported for "
                "generation strategies that leverage `model.update` functionality."
            )
