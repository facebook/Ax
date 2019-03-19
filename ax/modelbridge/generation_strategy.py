#!/usr/bin/env python3

from functools import wraps
from inspect import signature
from typing import Any, Callable, List, Optional

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge


TModelFactory = Callable[..., ModelBridge]
MAX_CONDITIONS_GENERATED = 10000


def _from_generation_strategy(f: Callable) -> Callable:
    """Set 'from_generator_strategy' attribute on the wrapped function to true."""

    @wraps(f)
    def wrapped_get_model(*args: Any, **kwargs: Any) -> ModelBridge:
        return f(*args, **kwargs)

    wrapped_get_model.from_generation_strategy = True
    return wrapped_get_model


def _filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}


class GenerationStrategy:
    """GenerationStrategy describes which model should be used to generate new
    points for which trials, enabling and automating use of different models
    throughout the optimization process. For instance, it allows to use one
    model for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of models to generate an arbitrary numbers of arms
    described in the `arms_per_model` argument.

    Note: if model returned from `GenerationStrategy.get_model` is
    used to generate more than one arm, it is possible that a model
    actually generating the N-th arm using `GenerationStrategy` is not
    the one designated in the strategy. This is because each model is created
    during the execution of `get_model` and not changed until `get_model`
    is executed again. For instance:

    >>  strategy = GenerationStrategy(
    ...     model_factories=[get_sobol, get_GPEI],
    ...     arms_per_model=[1, 25]
    ... )
    ... exp = make_my_experiment()  # Some function that creates an Experiment.
    ... sobol = strategy.get_model(experiment=exp)
    ... # All 5 arms generated in line below will be generated through
    ... # Sobol, even though GenerationStrategy designates Sobol model only
    ... # for the 1st arm in the experiment.
    ... generator_run = sobol.gen(5)
    ... exp.new_batch_trial().add_generator_run(generator_run=generator_run)
    ... # Now that the experiment includes at least one arm, `get_model`
    ... # returns the next model in the strategy after Sobol, GP+EI model.
    ... gpei = strategy.get_model(experiment=exp, data=exp.fetch_data())

    Args:
        model_factories: functions that return a
            single model. Index of a factory function in this list will
            correspond to the ordering of models in a ``GenerationStrategy``.
            This list is expected to have more than one model factory function.

        arms_per_model: number of arms for each
            of the models in ``model_factories`` to generate.
    """

    _model_factories: List[TModelFactory]
    _arms_per_model: List[int]
    _generator_changes: List[int] = None
    _last_used_model: Optional[ModelBridge] = None

    def __init__(
        self, model_factories: List[TModelFactory], arms_per_model: List[int]
    ) -> None:
        if len(arms_per_model) != len(model_factories):
            raise ValueError(
                "GenerationStrategy expects to include as many designated "
                "number of arms per model as models. "
            )
        if min(arms_per_model) <= 0:
            raise ValueError("arms_per_model must all be greater than 0.")
        self._model_factories = model_factories
        self._arms_per_model = arms_per_model
        # Record at what iteration in the experiment models changed
        # (used to indicate generator changes in plotting).
        gen_changes = self._arms_per_model
        self._generator_changes = [
            sum(gen_changes[: i + 1]) for i in range(len(gen_changes))
        ][:-1]
        # TODO[drfreund]: possibly add 'bind_kwargs' argument to pass kwargs
        # needed for instantiation of different models in the strategy.

    @property
    def name(self) -> str:
        factory_names = (
            factory.__name__[4:]
            # Trim the "get_" beginning of the factory function if it's there.
            if factory.__name__[:4] == "get_" else factory.__name__
            for factory in self._model_factories
        )
        return "+".join(factory_names)

    @property
    def generator_changes(self) -> List[int]:
        return self._generator_changes

    @_from_generation_strategy
    def get_model(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        search_space: Optional[SearchSpace] = None,
        exclude_abandoned: bool = False,
        **kwargs: Any,
    ) -> ModelBridge:
        """Obtains a model from the factory corresponding to the current
        trial in the strategy. This function filters out keyword arguments
        that are not applicable to the model factory function to be used
        for this trial.

        Note: if the experiment passed as argument has 0 trials, number of
        arms generated in this strategy resets to 0, which means that the
        strategy starts with the first provided model.

        Args:
            experiment: experiment, for which this generation
                strategy will be generating arms.
            data: data, on which to train the model, defaults
                to None.
            search_space: search space for this experiment.
            exclude_abandoned: whether we should exclude abandoned
                arms in the experiment when determining which model
                to return (e.g., if this generator strategy uses model A for
                first 5 arms, and then model B, if one of the first 5
                arms is abandoned, model A will be returned from this
                function if exclude_abandoned is True and model B if its
                False). Defaults to False.
            **kwargs: any other arguments to be passed into the model (e.g.,
                'min_weight' for Thompson Sampler).
        """
        # Determine how many arms are already attached to this experiment.
        arms_ran = (
            experiment.sum_trial_sizes - experiment.num_abandoned_arms
            if exclude_abandoned
            else experiment.sum_trial_sizes
        )

        if arms_ran >= sum(self._arms_per_model):
            raise ValueError(
                "This generation strategy expected to generate only "
                f"{sum(self._arms_per_model)} arms, "
                f"but experiment includes {arms_ran} arms already."
            )

        # Find index of model to use for this trial.
        idx = 0
        while sum(self._arms_per_model[: idx + 1]) <= arms_ran and idx + 1 < len(
            self._model_factories
        ):
            idx += 1
        # Is this model the same as the one that would've been returned for
        # the previous trial:
        same_model = sum(self._arms_per_model[:idx]) < arms_ran

        factory = self._model_factories[idx]

        # Filter out kwargs that the specific chosen factory function does not
        # require.
        factory_kwargs = _filter_kwargs(
            factory,
            experiment=experiment,
            data=data if data is not None else Data(),
            search_space=search_space
            if search_space is not None
            else experiment.search_space,
            **kwargs,
        )
        if (
            "data" not in factory_kwargs.keys()
            and same_model
            and self._last_used_model is not None
        ):
            current_model = self._last_used_model
        else:
            current_model = factory(**factory_kwargs)
        self._last_used_model = current_model
        return current_model
