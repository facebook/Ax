#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from inspect import signature
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import pandas as pd
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger = get_logger(__name__)


TModelFactory = Callable[..., ModelBridge]
MAX_CONDITIONS_GENERATED = 10000


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


def get_model_from_generator_run(
    generator_run: GeneratorRun, experiment: Experiment, data: Data
) -> ModelBridge:
    """Reinstantiate a model from model key and kwargs stored on a given generator
    run, with the given experiment and the data to initialize the model with.
    """
    if not generator_run._model_key:  # pragma: no cover
        raise ValueError(
            "Cannot restore model from generator run as no model key was "
            "on the generator run stored."
        )
    model = Models(generator_run._model_key)
    return model(
        experiment=experiment,
        data=data,
        **(generator_run._model_kwargs or {}),
        **(generator_run._bridge_kwargs or {}),
    )


class GenerationStep(NamedTuple):
    """One step in the generation strategy, corresponds to a single model.
    Describes the model, how many arms will be generated with this model, what
    minimum number of observations is required to proceed to the next model, etc.
    """

    model: Union[Models, Callable[..., ModelBridge]]
    num_arms: int
    min_arms_observed: int = 0
    recommended_max_parallelism: Optional[int] = None
    enforce_num_arms: bool = True
    # Kwargs to pass into the Models constructor (or factory function).
    model_kwargs: Dict[str, Any] = None
    # Kwargs to pass into the Model's `.gen` function.
    model_gen_kwargs: Dict[str, Any] = None
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
    # Whether all models in this GS are in Models registry enum.
    _uses_registered_models: bool
    # Latest generator run produced by this GS (used to restore current model
    # when decoding a serialized GS).
    _last_generator_run: Optional[GeneratorRun]

    def __init__(self, steps: List[GenerationStep], name: Optional[str] = None) -> None:
        self._name = name
        self._steps = steps
        assert isinstance(self._steps, list), "Steps must be a GenerationStep list."
        self._uses_registered_models = True
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
            if not isinstance(step.model, Models):
                self._uses_registered_models = False
        self._generated = []
        self._observed = []
        self._model = None
        self._data = Data()
        self._curr = steps[0]
        self._last_generator_run = None

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps."""
        if self._name:
            return self._name

        # Model can be defined as member of Models enum or as a factory function,
        # so we use Models member (str) value if former and function name if latter.
        factory_names = (
            checked_cast(str, step.model.value)
            if isinstance(step.model, Models)
            else step.model.__name__  # pyre-ignore[16]
            for step in self._steps
        )
        # Trim the "get_" beginning of the factory function if it's there.
        factory_names = (n[4:] if n[:4] == "get_" else n for n in factory_names)

        return "+".join(factory_names)

    @property
    def generator_changes(self) -> List[int]:
        """List of arm indices where a transition happened from one model to
        another."""
        gen_changes = [step.num_arms for step in self._steps]
        return [sum(gen_changes[: i + 1]) for i in range(len(gen_changes))][:-1]

    @property
    def model(self) -> Optional[ModelBridge]:
        """Current model in this strategy."""
        return self._model  # pragma: no cover

    @property
    def uses_non_registered_models(self) -> bool:
        """Whether this generation strategy involves models that are not
        registered and therefore cannot be stored."""
        return not self._uses_registered_models

    def gen(
        self,
        experiment: Experiment,
        new_data: Optional[Data] = None,  # Take in just the new data.
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        # Get arm signatures for each entry in new_data that is indeed new.
        new_arms = self._get_new_arm_signatures(
            experiment=experiment, new_data=new_data
        )
        enough_observed = (
            len(self._observed) + len(new_arms)
        ) >= self._curr.min_arms_observed
        unlimited_arms = self._curr.num_arms == -1
        enough_generated = (
            not unlimited_arms and len(self._generated) >= self._curr.num_arms
        )
        remaining_arms = self._curr.num_arms - len(self._generated)

        # Check that minimum observed_arms is satisfied if it's enforced.
        if self._curr.enforce_num_arms and enough_generated and not enough_observed:
            raise ValueError(
                "All trials for current model have been generated, but not enough "
                "data has been observed to fit next model. Try again when more data "
                "are available."
            )
            # TODO[Lena, T44021164]: take into account failed trials. Potentially
            # reduce `_generated` count when a trial mentioned in new data failed.
        if (
            self._curr.enforce_num_arms
            and not unlimited_arms
            and 0 < remaining_arms < n
        ):
            raise ValueError(
                f"Cannot generate {n} new arms as there are only {remaining_arms} "
                "remaining arms to generate using the current model."
            )

        all_data = (
            Data.from_multiple_data(data=[self._data, new_data])
            if new_data
            else self._data
        )

        if self._model is None:
            # Instantiate the first model.
            self._set_current_model(experiment=experiment, data=all_data, **kwargs)
        elif enough_generated and enough_observed:
            # Change to the next model.
            self._change_model(experiment=experiment, data=all_data, **kwargs)
        elif new_data is not None:
            # We're sticking with the current model, but update with new data
            self._model.update(experiment=experiment, data=new_data)

        gen_run = not_none(self._model).gen(n=n, **(self._curr.model_gen_kwargs or {}))

        # If nothing failed, update known data, _generated, and _observed.
        self._data = all_data
        self._observed.extend(new_arms)
        self._generated.extend(a.signature for a in gen_run.arms)
        self._last_generator_run = gen_run
        return gen_run

    def clone_reset(self) -> "GenerationStrategy":
        """Copy this generation strategy without it's state."""
        return GenerationStrategy(name=self.name, steps=self._steps)

    @equality_typechecker
    def __eq__(self, other: "GenerationStrategy") -> bool:
        """Need to override the default __eq__ method, because the default
        checks equality of memory addresses of the objects.
        """
        data_equals = (
            self._data.df.empty and other._data.df.empty
        ) or self._data.df.sort_index(axis=1).equals(other._data.df.sort_index(axis=1))
        return (
            self._generated == other._generated
            and self._observed == other._observed
            and data_equals
            and self._curr == other._curr
        )

    def __repr__(self) -> str:
        """String representation of this generation strategy."""
        repr = f"GenerationStrategy(name='{self.name}', steps=["
        for step in self._steps:
            num_arms = f"{step.num_arms}" if step.num_arms != -1 else "subsequent"
            if isinstance(step.model, Models):
                repr += f"{step.model.value} for {num_arms} arms, "
        repr = repr[:-2]
        repr += f"], generated {len(self._generated)} arm(s))"
        return repr

    def _set_current_model(
        self, experiment: Experiment, data: Data, **kwargs: Any
    ) -> None:
        """Instantiate the current model with all available data.
        """
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
        logger.info(
            f"Using a custom model provided through a callable function "
            "`{self._curr.model.__name__}`"
            ". Note that Ax cannot save models provided through functions, "
            "so this optimization will not be resumable if interrupted."
        )
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

    def _restore_model_from_generator_run(self, experiment: Experiment) -> None:
        generator_run = self._last_generator_run
        if generator_run is None:
            raise ValueError("No generator run was stored on generation strategy.")
        self._model = get_model_from_generator_run(
            generator_run=generator_run, experiment=experiment, data=self._data
        )

    def _change_model(self, experiment: Experiment, data: Data, **kwargs: Any) -> None:
        """Get a new model for the next step.
        """
        # Increment the model
        if len(self._steps) == not_none(self._curr.index) + 1:
            raise ValueError(f"Generation strategy {self.name} is completed.")
        self._curr = self._steps[not_none(self._curr.index) + 1]
        # New step => reset _generated and _observed.
        self._generated, self._observed = [], []
        self._set_current_model(experiment=experiment, data=data, **kwargs)

    def _get_new_arm_signatures(
        self, experiment: Experiment, new_data: Optional[Data]
    ) -> List[str]:
        new_signatures = []
        if new_data is not None:
            for _, row in new_data.df.iterrows():
                # If a row with the same trial index, arm name, and metric name
                # has already been seen in this generation strategy, the
                # data passed into this function is not entirely new.
                if not self._data.df.empty:
                    if not pd.merge(
                        new_data.df,
                        self._data.df,
                        on=["arm_name", "metric_name", "trial_index"],
                    ).empty:
                        arm = row["arm_name"]
                        trial = row["trial_index"]
                        metric = row["metric_name"]
                        raise ValueError(
                            f"Data for arm {arm} in trial {trial} for metric "
                            f"{metric} has already been seen. Please only pass "
                            "new data to `GenerationStrategy.gen`."
                        )
                if (
                    row["arm_name"] in experiment.arms_by_name
                    # pyre-fixme[16]: `Optional` has no attribute `status`.
                    and not experiment.trials.get(row["trial_index"]).status.is_failed
                ):
                    new_signatures.append(
                        # pyre-fixme[16]: `Optional` has no attribute `signature`.
                        experiment.arms_by_name.get(row["arm_name"]).signature
                    )
        return new_signatures
