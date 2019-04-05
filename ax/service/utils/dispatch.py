#!/usr/bin/env python3
from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.factory import get_GPEI, get_sobol
from ax.modelbridge.generation_strategy import GenerationStrategy


def choose_generation_strategy(search_space: SearchSpace) -> GenerationStrategy:
    """Select an appropriate generation strategy based on the properties of
    the search space."""
    num_continuous_parameters, num_discrete_choices = 0, 0
    for parameter in search_space.parameters:
        if isinstance(parameter, ChoiceParameter):
            num_discrete_choices += len(parameter.values)
        if isinstance(parameter, RangeParameter):
            num_continuous_parameters += 1
    # If there are more discrete choices than continuous parameters, Sobol
    # will do better than GP+EI.
    if num_continuous_parameters >= num_discrete_choices:
        return GenerationStrategy(
            model_factories=[get_sobol, get_GPEI],
            arms_per_model=[max(5, len(search_space.parameters)), -1],
        )
    else:
        # Expected `List[typing.Callable[..., ax.modelbridge.base.ModelBridge]]`
        # for 1st parameter `model_factories` to call `GenerationStrategy.__init__`
        # but got `List[typing.Callable(ax.modelbridge.factory.get_sobol)
        # [[Named(search_space, SearchSpace), Keywords(kwargs,
        # typing.Union[bool, int])], ax.modelbridge.random.RandomModelBridge]]`.
        # pyre-fixme[6]:
        return GenerationStrategy(model_factories=[get_sobol], arms_per_model=[-1])
