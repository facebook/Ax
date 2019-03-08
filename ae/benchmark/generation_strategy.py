#!/usr/bin/env python3

from functools import wraps
from typing import Any, Callable, List, Optional

from ae.lazarus.ae.benchmark.utils import filter_kwargs
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.generator.base import Generator


TGeneratorFactory = Callable[..., Generator]
MAX_CONDITIONS_GENERATED = 10000


def _from_generation_strategy(f: Callable) -> Callable:
    """Set 'from_generator_strategy' attribute on the wrapped function to true."""

    @wraps(f)
    def wrapped_get_generator(*args: Any, **kwargs: Any) -> Generator:
        return f(*args, **kwargs)

    wrapped_get_generator.from_generation_strategy = True
    return wrapped_get_generator


class GenerationStrategy:
    """GenerationStrategy describes which generator should be used for which
    trials of data, enabling and automating use of different generators
    throughout the optimization process. For instance, it allows to use one
    generator for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of generators to generate an arbitrary numbers of arms
    described in the `arms_per_generator` argument.

    Note: if generator returned from `GenerationStrategy.get_generator` is
    used to generate more than one arm, it is possible that a generator
    actually generating the N-th arm using `GenerationStrategy` is not
    the one designated in the strategy. This is because each generator is created
    during the execution of `get_generator` and not changed until `get_generator`
    is executed again. For instance:

    >>  strategy = GenerationStrategy(
    ...     generator_factories=[get_sobol, get_GPEI],
    ...     arms_per_generator=[1, 25]
    ... )
    ... exp = make_my_experiment()  # Some function that creates an Experiment.
    ... sobol = strategy.get_generator(experiment=exp)
    ... # All 5 arms generated in line below will be generated through
    ... # Sobol, even though GenerationStrategy designates Sobol generator only
    ... # for the 1st arm in the experiment.
    ... generator_run = sobol.gen(5)
    ... exp.new_batch_trial().add_generator_run(generator_run=generator_run)
    ... # Now that the experiment includes at least one arm, `get_generator`
    ... # returns the next generator in the strategy after Sobol, GP+EI generator.
    ... gpei = strategy.get_generator(experiment=exp, data=exp.fetch_data())

    Args:
        generator_factories: functions that return a
            single generator. Index of a factory function in this list will
            correspond to the ordering of generators in a ``GenerationStrategy``.
            This list is expected to have more than one generator factory function.

        arms_per_generator: number of arms for each
            of the generators in ``generator_factories`` to generate.
    """

    _generator_factories: List[TGeneratorFactory]
    _arms_per_generator: List[int]
    _generator_changes: List[int] = None
    _last_used_generator: Optional[Generator] = None

    def __init__(
        self,
        generator_factories: List[TGeneratorFactory],
        arms_per_generator: List[int],
    ) -> None:
        if len(arms_per_generator) != len(generator_factories):
            raise ValueError(
                "GenerationStrategy expects to include as many designated "
                "number of arms per generator as generators. "
            )
        if min(arms_per_generator) <= 0:
            raise ValueError("arms_per_generator must all be greater than 0.")
        self._generator_factories = generator_factories
        self._arms_per_generator = arms_per_generator
        # Record at what iteration in the experiment generators changed
        # (used to indicate generator changes in plotting).
        gen_changes = self._arms_per_generator
        self._generator_changes = [
            sum(gen_changes[: i + 1]) for i in range(len(gen_changes))
        ][:-1]
        # TODO[drfreund]: possibly add 'bind_kwargs' argument to pass kwargs
        # needed for instantiation of different generators in the strategy.

    @property
    def name(self) -> str:
        factory_names = (
            factory.__name__[4:]
            # Trim the "get_" beginning of the factory function if it's there.
            if factory.__name__[:4] == "get_" else factory.__name__
            for factory in self._generator_factories
        )
        return "+".join(factory_names)

    @property
    def generator_changes(self) -> List[int]:
        return self._generator_changes

    @_from_generation_strategy
    def get_generator(
        self,
        experiment: Experiment,
        data: Optional[Data] = None,
        search_space: Optional[SearchSpace] = None,
        exclude_abandoned: bool = False,
        **kwargs: Any,
    ) -> Generator:
        """Obtains a generator from the factory corresponding to the current
        trial in the strategy. This function filters out keyword arguments
        that are not applicable to the generator factory function to be used
        for this trial.

        Note: if the experiment passed as argument has 0 trials, number of
        arms generated in this strategy resets to 0, which means that the
        strategy starts with the first provided generator.

        Args:
            experiment: experiment, for which this generation
                strategy will be generating arms.
            data: data, on which to train the generator, defaults
                to None.
            search_space: search space for this experiment.
            exclude_abandoned: whether we should exclude abandoned
                arms in the experiment when determining which generator
                to return (e.g., if this generator strategy uses generator A for
                first 5 arms, and then generator B, if one of the first 5
                arms is abandoned, generator A will be returned from this
                function if exclude_abandoned is True and generator B if its
                False). Defaults to False.
            **kwargs: any other arguments to be passed into the generator (e.g.,
                'min_weight' for Thompson Sampler).
        """
        # Determine how many arms are already attached to this experiment.
        arms_ran = (
            experiment.sum_trial_sizes - experiment.num_abandoned_arms
            if exclude_abandoned
            else experiment.sum_trial_sizes
        )

        if arms_ran >= sum(self._arms_per_generator):
            raise ValueError(
                "This generation strategy expected to generate only "
                f"{sum(self._arms_per_generator)} arms, "
                f"but experiment includes {arms_ran} arms already."
            )

        # Find index of generator to use for this trial.
        idx = 0
        while sum(self._arms_per_generator[: idx + 1]) <= arms_ran and idx + 1 < len(
            self._generator_factories
        ):
            idx += 1
        # Is this generator the same as the one that would've been returned for
        # the previous trial:
        same_generator = sum(self._arms_per_generator[:idx]) < arms_ran

        factory = self._generator_factories[idx]

        # Filter out kwargs that the specific chosen factory function does not
        # require.
        factory_kwargs = filter_kwargs(
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
            and same_generator
            and self._last_used_generator is not None
        ):
            current_generator = self._last_used_generator
        else:
            current_generator = factory(**factory_kwargs)
        self._last_used_generator = current_generator
        return current_generator
