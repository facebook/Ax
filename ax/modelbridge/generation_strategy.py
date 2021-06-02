#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import pandas as pd
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.exceptions.core import DataRequiredError, NoDataError
from ax.exceptions.generation_strategy import (
    GenerationStrategyCompleted,
    MaxParallelismReachedException,
)
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import (
    ModelRegistryBase,
    _combine_model_kwargs_and_state,
    get_model_from_generator_run,
)
from ax.utils.common.base import Base, SortableBase
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


@dataclass
class GenerationStep(SortableBase):
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

    model: Union[ModelRegistryBase, Callable[..., ModelBridge]]
    num_trials: int
    min_trials_observed: int = 0
    max_parallelism: Optional[int] = None
    use_update: bool = False
    enforce_num_trials: bool = True
    # Kwargs to pass into the Models constructor (or factory function).
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass into the Model's `.gen` function.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    index: int = -1  # Index of this step, set internally.

    @property
    def model_name(self) -> str:
        # Model can be defined as member of Models enum or as a factory function,
        # so we use Models member (str) value if former and function name if latter.
        if isinstance(self.model, ModelRegistryBase):
            return checked_cast(str, checked_cast(ModelRegistryBase, self.model).value)
        try:
            # `model` is defined via a factory function.
            return self.model.__name__  # pyre-fixme[16]: union has no attr __name__
        except Exception:
            raise TypeError(  # pragma: no cover
                f"`model` {self.model} was not a member of `Models` or a function."
            )

    @property
    def _unique_id(self) -> str:
        return str(self.index)


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
    _curr: GenerationStep  # Current step in the strategy.
    # Whether all models in this GS are in Models registry enum.
    _uses_registered_models: bool
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment] = None
    # Trial indices as last seen by the model; updated in `_model` property setter.
    _seen_trial_indices_by_status = None
    _model: Optional[ModelBridge] = None  # Current model.

    def __init__(self, steps: List[GenerationStep], name: Optional[str] = None) -> None:
        assert isinstance(steps, list) and all(
            isinstance(s, GenerationStep) for s in steps
        ), "Steps must be a GenerationStep list."
        self._name = name
        self._steps = steps
        self._uses_registered_models = True
        self._generator_runs = []
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
            step.index = idx
            if not isinstance(step.model, ModelRegistryBase):
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

    @name.setter
    def name(self, name: str) -> None:
        """Set generation strategy name."""
        self._name = name

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
    def num_running_trials_this_step(self) -> int:
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

    @property
    def num_can_complete_this_step(self) -> int:
        """Number of trials for the current step in generation strategy that can
        be completed (so are not in status `FAILED` or `ABANDONED`). Used to keep
        track of how many generator runs (that become trials) can be produced
        from the current generation step.

        NOTE: This includes `COMPLETED` trials.
        """
        step_trials = self.trial_indices_by_step[self._curr.index]
        by_status = self.experiment.trial_indices_by_status
        # Number of trials that will not be `COMPLETED`, used to avoid counting
        # unsuccessfully terminated trials against the number of generated trials
        # during determination of whether enough trials have been generated and
        # completed to proceed to the next generation step.
        num_will_not_complete = len(
            step_trials.intersection(
                by_status[TrialStatus.FAILED].union(by_status[TrialStatus.ABANDONED])
            )
        )
        return len(step_trials) - num_will_not_complete

    @property
    def num_completed_this_step(self) -> int:
        """Number of trials in status `COMPLETD` for the current generation step
        of this strategy.
        """
        step_trials = self.trial_indices_by_step[self._curr.index]
        by_status = self.experiment.trial_indices_by_status
        return len(step_trials.intersection(by_status[TrialStatus.COMPLETED]))

    def gen(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        n: int = 1,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment. Additional kwargs passed to
        this method are propagated directly to the underlying model's `gen`, along
        with the `model_gen_kwargs` set on the current generation step.

        NOTE: Each generator run returned from this function must become a single
        trial on the experiment to comply with assumptions made in generation
        strategy. Do not split one generator run produced from generation strategy
        into multiple trials (never making a generator run into a trial is allowed).

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
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
        """
        return self._gen_multiple(
            experiment=experiment,
            num_generator_runs=1,
            data=data,
            n=n,
            pending_observations=pending_observations,
            **kwargs,
        )[0]

    def clone_reset(self) -> GenerationStrategy:
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
            try:
                model_name = step.model_name
            except TypeError:
                model_name = "model with unknown name"

            repr += f"{model_name} for {num_trials} trials, "
        repr = repr[:-2]
        repr += "])"
        return repr

    # ------------------------- Candidate generation helpers. -------------------------

    def _gen_multiple(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Optional[Data] = None,
        n: int = 1,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        **kwargs: Any,
    ) -> List[GeneratorRun]:
        """Produce multiple generator runs at once, to be made into multiple
        trials on the experiment.

        NOTE: This is used to ensure that maximum paralellism and number
        of trials per step are not violated when producing many generator
        runs from this generation strategy in a row. Without this function,
        if one generates multiple generator runs without first making any
        of them into running trials, generation strategy cannot enforce that it only
        produces as many generator runs as are allowed by the paralellism
        limit and the limit on number of trials in current step.

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
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
        """
        self.experiment = experiment
        self._set_or_update_model(data=data)
        self._save_seen_trial_indices()
        max_parallelism = self._curr.max_parallelism
        num_running = self.num_running_trials_this_step

        # Make sure to not make too many generator runs and
        # exceed maximum allowed paralellism for the step.
        if max_parallelism is not None:
            if num_running >= max_parallelism:
                raise MaxParallelismReachedException(
                    step_index=self._curr.index,
                    model_name=self._curr.model_name,
                    num_running=num_running,
                )
            else:
                num_generator_runs = min(
                    num_generator_runs, max_parallelism - num_running
                )

        # Make sure not to extend number of trials expected in step.
        if self._curr.enforce_num_trials and self._curr.num_trials > 0:
            num_generator_runs = min(
                num_generator_runs,
                self._curr.num_trials - self.num_can_complete_this_step,
            )

        model = not_none(self.model)
        generator_runs = []
        for _ in range(num_generator_runs):
            try:
                generator_run = model.gen(
                    n=n,
                    pending_observations=pending_observations,
                    **consolidate_kwargs(
                        kwargs_iterable=[self._curr.model_gen_kwargs, kwargs],
                        keywords=get_function_argument_names(model.gen),
                    ),
                )
                generator_run._generation_step_index = self._curr.index
                self._generator_runs.append(generator_run)
                generator_runs.append(generator_run)
            except DataRequiredError as err:
                # Model needs more data, so we log the error and return
                # as many generator runs as we were able to produce, unless
                # no trials were produced at all (in which case its safe to raise).
                if len(generator_runs) == 0:
                    raise
                logger.debug(f"Model required more data: {err}.")
                break

        return generator_runs

    # ------------------------- Model selection logic helpers. -------------------------

    def _set_or_update_model(self, data: Optional[Data]) -> None:
        if self._curr.num_trials == -1:  # Unlimited trials, just use curr. model.
            self._set_or_update_current_model(data=data)
            return

        # Not unlimited trials => determine whether to transition to next model.
        enough_generated = self.num_can_complete_this_step >= self._curr.num_trials
        enough_observed = self.num_completed_this_step >= self._curr.min_trials_observed

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
        """Instantiate the current model with all available data."""
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
            model_kwargs = _combine_model_kwargs_and_state(
                model_kwargs=model_kwargs,
                generator_run=lgr,
                model_class=not_none(not_none(self.model).model.__class__),
            )

        if data is None:
            if self._curr.use_update:
                # If the new step is using `update`, it's important to instantiate
                # the model with data for completed trials only, so later we can
                # update it with data for new trials as they become completed.
                # `experiment.lookup_data` can lookup all available data, including
                # for non-completed trials (depending on how the experiment's metrics
                # implement `fetch_experiment_data`). We avoid fetching data for
                # trials with statuses other than `COMPLETED`, by fetching specifically
                # for `COMPLETED` trials.
                avail_while_running_metrics = {
                    m.name
                    for m in self.experiment.metrics.values()
                    if m.is_available_while_running()
                }
                if avail_while_running_metrics:
                    raise NotImplementedError(
                        f"Metrics {avail_while_running_metrics} are available while "
                        "trial is running, but use of `update` functionality in "
                        "generation strategy relies on new data being available upon "
                        "trial completion."
                    )
                data = self.experiment.lookup_data(
                    trial_indices=self.experiment.trial_indices_by_status[
                        TrialStatus.COMPLETED
                    ]
                )
            else:
                data = self.experiment.lookup_data()
        # By the time we get here, we will have already transitioned
        # to a new step, but if previou step required observed data,
        # we should raise an error even if enough trials were completed.
        # Such an empty data case does indicate an invalid state; this
        # check is to improve the experience of detecting and debugging
        # the invalid state that led to this.
        previous_step_required_observations = (
            self._curr.index > 0
            and self._steps[self._curr.index - 1].min_trials_observed > 0
        )
        if data.df.empty and previous_step_required_observations:
            raise NoDataError(
                f"Observed data is required for generation step #{self._curr.index} "
                f"(model {self._curr.model_name}), but fetched data was empty. "
                "Something is wrong with experiment setup -- likely metrics do not "
                "implement fetching logic (check your metrics) or no data was "
                "attached to experiment for completed trials."
            )
        # TODO(jej)[T87591836] Support non-`Data` data types.
        if isinstance(self._curr.model, ModelRegistryBase):
            # pyre-fixme [6]: Incompat param: Expect `Data` got `AbstractDataFrameData`
            self._set_current_model_from_models_enum(data=data, **model_kwargs)
        else:
            # If model was not specified as Models member, it was specified as a
            # factory function.
            # pyre-fixme [6]: Incompat param: Expect `Data` got `AbstractDataFrameData`
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
            logger.debug(
                "There were no newly completed trials since last model update."
            )
            return
        if data is None:
            new_data = self.experiment.lookup_data(trial_indices=newly_completed_trials)
            if new_data.df.empty:
                logger.info("Skipping model update as there is no new data.")
                return
        elif data.df.empty:
            logger.info("Skipping model update as data supplied to `gen` is empty.")
            return
        else:
            new_data = Data(
                # pyre-fixme[6]: Expected `Optional[pd.core.frame.DataFrame]` for
                #  1st param but got `Series`.
                df=data.df[data.df.trial_index.isin(newly_completed_trials)]
            )
        # We definitely have non-empty new data by now.
        trial_indices_in_new_data = sorted(new_data.df["trial_index"].unique())
        logger.info(f"Updating model with data for trials: {trial_indices_in_new_data}")
        # pyre-fixme [6]: Incompat param: Expected `Data` got `AbstractDataFrameData`
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
        assert not isinstance(model, ModelRegistryBase) and callable(model)
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
        self, models_enum: Optional[Type[ModelRegistryBase]] = None
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
        data = self.experiment.lookup_data()
        self._model = get_model_from_generator_run(
            generator_run=generator_run,
            experiment=self.experiment,
            # pyre-fixme [6]: Incompat param: Expect `Data` got `AbstractDataFrameData`
            data=data,
            models_enum=models_enum,
        )
        self._save_seen_trial_indices()

    # ------------------------- State-tracking helpers. -------------------------

    def _save_seen_trial_indices(self) -> None:
        """Saves Experiment's `trial_indices_by_status` at the time of the model's
        last `gen` (so these `trial_indices_by_status` reflect which trials model
        has seen the data for). Useful when `use_update=True` for a given
        generation step.
        """
        self._seen_trial_indices_by_status = deepcopy(
            self.experiment.trial_indices_by_status
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

    def _register_trial_data_update(self, trial: BaseTrial) -> None:
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
