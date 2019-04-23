#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from inspect import signature
from typing import Any, Callable, List, NamedTuple, Optional

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.factory import Models
from ax.utils.common.typeutils import not_none


TModelFactory = Callable[..., ModelBridge]
MAX_CONDITIONS_GENERATED = 10000


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


class GenerationStep(NamedTuple):
    """One step in the generation strategy, corresponds to a single model.
    Describes the model, how many arms will be generated with this model, what
    minimum number of observations is required to proceed to the next model, etc.
    """

    model: Models
    num_arms: int
    min_arms_observed: int = 0
    max_async_parallelism: Optional[int] = None  # TODO[drfreund, T41983558]: use
    enforce_num_arms: bool = True
    model_kwargs: Any = None  # Kwargs to pass into the Models factory function.
    index: Optional[int] = None  # Index of this step, set internally.


class GenerationStrategy:
    """GenerationStrategy describes which model should be used to generate new
    points for which trials, enabling and automating use of different models
    throughout the optimization process. For instance, it allows to use one
    model for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of models to generate an arbitrary numbers of arms
    described in the `arms_per_model` argument.
    """

    _name: Optional[str]
    _steps: List[GenerationStep]
    _generated: List[str]  # Arms generated in the current generation step.
    _observed: List[str]  # Arms in the current step for which we observed data.
    _model: Optional[ModelBridge]  # Current model.
    _data: Data  # All data this strategy has been updated with.
    _curr: GenerationStep  # Current step in the strategy.

    def __init__(self, steps: List[GenerationStep], name: Optional[str] = None) -> None:
        self._name = name
        self._steps = steps
        assert isinstance(self._steps, list), "Steps must be a GenerationStep list."
        for idx, step in enumerate(self._steps):
            if step.num_arms == -1:
                if idx < len(self._steps) - 1:
                    raise ValueError(
                        "Only last step in generation strategy can have num_arms "
                        "set to -1 to indicate that the model in the step should "
                        "be used to generate new arms indefinitely."
                    )
            elif step.num_arms < 1:
                raise ValueError("`num_arms` must be positive or -1 for all models.")
            self._steps[idx] = step._replace(index=idx)
        self._generated = []
        self._observed = []
        self._model = None
        self._data = Data()
        self._curr = steps[0]

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps."""
        if self._name:
            return self._name

        factory_names = (
            step.model.__name__[4:]  # pyre-ignore[16] T41922457
            # Trim the "get_" beginning of the factory function if it's there.
            # pyre-ignore[16] T41922457
            if step.model.__name__[:4] == "get_" else step.model.__name__
            for step in self._steps
        )
        return "+".join(factory_names)  # pyre-ignore[6]

    @property
    def generator_changes(self) -> List[int]:
        """List of arm indices where a transition happened from one model to
        another."""
        gen_changes = [step.num_arms for step in self._steps]
        return [sum(gen_changes[: i + 1]) for i in range(len(gen_changes))][:-1]

    def gen(
        self,
        experiment: Experiment,
        new_data: Optional[Data] = None,  # Take in just the new data.
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        # TODO[drfreund, T41983558]: add `n` arg. and handle generating batches.

        # Add new data to data we already have.
        if new_data is not None:
            self._data = Data.from_multiple_data(data=[self._data, new_data])
            for _, row in new_data.df.iterrows():
                if (
                    row["arm_name"] in experiment.arms_by_name
                    and not experiment.trials.get(row["trial_index"]).status.is_failed
                ):
                    # NOTE: if this arm was suggested multiple times, in the
                    # current step and before, and the data passed includes its
                    # evaluation not for the current step, it will still be
                    # counted as observed for the current step.
                    self._observed.append(
                        experiment.arms_by_name.get(row["arm_name"]).signature
                    )

        enough_observed = len(self._observed) >= self._curr.min_arms_observed
        enough_generated = (  # If num arms is -1, model should be used indefinitely.
            self._curr.num_arms != -1 and len(self._generated) >= self._curr.num_arms
        )

        # Check that minimum observed_arms is satisfied if it's enforced.
        if self._curr.enforce_num_arms and enough_generated and not enough_observed:
            raise ValueError(
                "All trials for current model have been generated, but not enough "
                "data has been observed to fit next model. Try again when more data "
                "are available."
            )

        if self._model is None:
            # Instantiate the first model.
            self._set_current_model(experiment=experiment, **kwargs)
        elif enough_generated and enough_observed:
            # Change to the next model.
            self._change_model(experiment=experiment, **kwargs)
        elif new_data is not None:
            # We're sticking with the current model, but update with new data
            self._model.update(experiment=experiment, data=new_data)

        assert self._model is not None  # guaranteed but typecheck doesn't know
        generator_run = self._model.gen(1)

        self._generated.extend(a.signature for a in generator_run.arms)
        return generator_run

    def _set_current_model(self, experiment: Experiment, **kwargs: Any) -> None:
        """Instantiate the current model with all available data.
        """
        self._model = self._curr.model(  # pyre-ignore[29] T41922457
            **_filter_kwargs(
                self._curr.model,
                experiment=experiment,
                data=self._data,
                search_space=experiment.search_space,
                **(self._curr.model_kwargs or {}),
                **kwargs,
            )
        )

    def _change_model(self, experiment: Experiment, **kwargs: Any) -> None:
        """Get a new model for the next step.
        """
        # Increment the model
        if len(self._steps) == not_none(self._curr.index) + 1:
            raise ValueError(f"Generation strategy {self.name} is completed.")
        self._curr = self._steps[not_none(self._curr.index) + 1]
        # New step => reset _generated and _observed.
        self._generated, self._observed = [], []
        self._set_current_model(experiment=experiment, **kwargs)

    def clone_reset(self) -> "GenerationStrategy":
        """Copy this generation strategy without it's state."""
        return GenerationStrategy(name=self.name, steps=self._steps)
