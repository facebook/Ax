#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from inspect import signature
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Type, Union

import pandas as pd
from ax.core.base import Base
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models, get_model_from_generator_run
from ax.utils.common.kwargs import consolidate_kwargs, get_function_argument_names
from ax.utils.common.logger import get_logger
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
    Describes the model, how many arms will be generated with this model, what
    minimum number of observations is required to proceed to the next model, etc.
    """

    model: Union[Models, Callable[..., ModelBridge]]
    num_arms: int
    min_arms_observed: int = 0
    recommended_max_parallelism: Optional[int] = None
    enforce_num_arms: bool = True
    # Kwargs to pass into the Models constructor (or factory function).
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass into the Model's `.gen` function.
    model_gen_kwargs: Optional[Dict[str, Any]] = None
    # pyre-fixme[15]: `index` overrides attribute defined in `tuple` inconsistently.
    index: Optional[int] = None  # Index of this step, set internally.


class GenerationStrategy(Base):
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
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment]
    _db_id: Optional[int]  # Used when storing to DB.

    def __init__(self, steps: List[GenerationStep], name: Optional[str] = None) -> None:
        self._db_id = None
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
        self._generator_runs = []
        self._experiment = None

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps."""
        if self._name:
            # pyre-fixme[7]: Expected `str` but got `Optional[str]`.
            return self._name

        # Model can be defined as member of Models enum or as a factory function,
        # so we use Models member (str) value if former and function name if latter.
        factory_names = (
            # pyre-fixme[16]: `Union` has no attribute `value`.
            checked_cast(str, step.model.value)
            if isinstance(step.model, Models)
            else step.model.__name__  # pyre-ignore[16]
            for step in self._steps
        )
        # Trim the "get_" beginning of the factory function if it's there.
        factory_names = (n[4:] if n[:4] == "get_" else n for n in factory_names)
        self._name = "+".join(factory_names)
        # pyre-fixme[7]: Expected `str` but got `Optional[str]`.
        return self._name

    @property
    def model_transitions(self) -> List[int]:
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

    @property
    def last_generator_run(self) -> Optional[GeneratorRun]:
        """Latest generator run produced by this generation strategy.
        Returns None if no generator runs have been produced yet.
        """
        # Used to restore current model when decoding a serialized GS.
        return self._generator_runs[-1] if self._generator_runs else None

    def gen(
        self,
        experiment: Experiment,
        new_data: Optional[Data] = None,  # Take in just the new data.
        n: int = 1,
        **kwargs: Any,
    ) -> GeneratorRun:
        """Produce the next points in the experiment."""
        self._set_experiment(experiment=experiment)

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

        # Check that minimum observed_arms is satisfied if it's enforced.
        if self._curr.enforce_num_arms and enough_generated and not enough_observed:
            raise ValueError(
                "All trials for current model have been generated, but not enough "
                "data has been observed to fit next model. Try again when more data "
                "are available."
            )
            # TODO[Lena, T44021164]: take into account failed trials. Potentially
            # reduce `_generated` count when a trial mentioned in new data failed.

        all_data = (
            Data.from_multiple_data(data=[self._data, new_data])
            if new_data
            else self._data
        )

        if self._model is None:
            # Instantiate the first model.
            self._set_current_model(experiment=experiment, data=all_data)
        elif enough_generated and enough_observed:
            # Change to the next model.
            self._change_model(experiment=experiment, data=all_data)
        elif new_data is not None:
            # We're sticking with the curr. model, but should update with new data.
            # pyre-fixme[16]: `Optional` has no attribute `update`.
            self._model.update(experiment=experiment, data=new_data)

        kwargs = consolidate_kwargs(
            kwargs_iterable=[self._curr.model_gen_kwargs, kwargs],
            keywords=get_function_argument_names(not_none(self._model).gen),
        )
        gen_run = not_none(self._model).gen(n=n, **kwargs)

        # If nothing failed, update known data, _generated, and _observed.
        self._data = all_data
        self._generated.extend([arm.signature for arm in gen_run.arms])
        self._observed.extend(new_arms)
        self._generator_runs.append(gen_run)
        return gen_run

    def clone_reset(self) -> "GenerationStrategy":
        """Copy this generation strategy without it's state."""
        return GenerationStrategy(name=self.name, steps=self._steps)

    def __repr__(self) -> str:
        """String representation of this generation strategy."""
        repr = f"GenerationStrategy(name='{self.name}', steps=["
        remaining_arms = "subsequent" if len(self._steps) > 1 else "all"
        for step in self._steps:
            num_arms = f"{step.num_arms}" if step.num_arms != -1 else remaining_arms
            if isinstance(step.model, Models):
                # pyre-fixme[16]: `Union` has no attribute `value`.
                repr += f"{step.model.value} for {num_arms} arms, "
        repr = repr[:-2]
        repr += f"], generated {len(self._generated)} arm(s) so far)"
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
        assert not isinstance(self._curr.model, Models)
        fxn_name = (  # Only grab the name when available; without this ternary
            # pyre-ignore[16]
            f"` {self._curr.model.__name__}`"  # operator, grabbing the __name__
            if hasattr(self._curr.model, "__name__")  # will error for mocks.
            else ""
        )
        logger.info(
            f"Using a custom model provided through a callable function {fxn_name}"
            ". Note that Ax cannot save models provided through functions, "
            "so this optimization will not be resumable if interrupted. For "
            "resumable optimization, use models, registered in the `Models` "
            "registry enum (`ax.modelbridge.registry.Models`)."
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
            experiment=not_none(self._experiment),
            data=self._data,
            models_enum=models_enum,
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

    def _set_experiment(self, experiment: Experiment) -> None:
        """If there is an experiment set on this generation strategy as the
        experiment it has been generating generator runs for, check if the
        experiment passed in is the same as the one saved and log an information
        statement if its not. Set the new experiment on this generation strategy.
        """
        if (
            self._experiment is not None
            and experiment._name is not not_none(self._experiment)._name
        ):  # pragma: no cover
            logger.info(
                "This generation strategy has been used for experiment "
                f"{not_none(self._experiment)._name} so far; generating trials for "
                f"{experiment._name} from now on. If this is a new optimization, "
                "a new generation strategy should be created instead."
            )
        self._experiment = experiment
