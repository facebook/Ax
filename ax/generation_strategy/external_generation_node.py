#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.model_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import TransitionCriterion


# TODO[drfreund]: Introduce a `GenerationNodeInterface` to
# make inheritance/overriding of `GenNode` methods cleaner.
class ExternalGenerationNode(GenerationNode, ABC):
    """A generation node intended to be used with non-Ax methods for
    candidate generation.

    To leverage external methods for candidate generation, the user must
    create a subclass that implements ``update_generator_state`` and
    ``get_next_candidate`` methods. This can then be provided
    as a node into a ``GenerationStrategy``, either as standalone or as
    part of a larger generation strategy with other generation nodes,
    e.g., with a Sobol node for initialization.

    Example:
    >>> class MyExternalGenerationNode(ExternalGenerationNode):
    >>>     ...
    >>> generation_strategy = GenerationStrategy(
    >>>     nodes = [MyExternalGenerationNode(...)]
    >>> )
    >>> ax_client = AxClient(generation_strategy=generation_strategy)
    >>> ax_client.create_experiment(...)
    >>> ax_client.get_next_trial()  # Generates trials using the new generation node.
    """

    def __init__(
        self,
        node_name: str,
        should_deduplicate: bool = True,
        transition_criteria: Sequence[TransitionCriterion] | None = None,
    ) -> None:
        """Initialize an external generation node.

        NOTE: The runtime accounting in this method should be replicated by the
        subclasses. This will ensure accurate comparison of runtimes between
        methods, in case a non-significant compute is spent in the constructor.

        Args:
            node_name: Name of the generation node.
            should_deduplicate: Whether to deduplicate the generated points against
                the existing trials on the experiment. If True, the duplicate points
                will be discarded and re-generated up to 5 times, after which a
                `GenerationStrategyRepeatedPoints` exception will be raised.
                NOTE: For this to work, the generator must be able to produce a
                different parameterization when called again with the same state.
            transition_criteria: Criteria for determining whether to move to the next
                node in the generation strategy. This is an advanced option that is
                only relevant if the generation strategy consists of multiple nodes.
        """
        t_init_start = time.monotonic()
        super().__init__(
            node_name=node_name,
            model_specs=[],
            best_model_selector=None,
            should_deduplicate=should_deduplicate,
            transition_criteria=transition_criteria,
        )
        self.fit_time_since_gen: float = time.monotonic() - t_init_start

    @abstractmethod
    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """A method used to update the state of the generator. This includes any
        models, predictors or any other custom state used by the generation node.
        This method will be called with the up-to-date experiment and data before
        ``get_next_candidate`` is called to generate the next trial(s). Note
        that ``get_next_candidate`` may be called multiple times (to generate
        multiple candidates) after a call to  ``update_generator_state``.

        Args:
            experiment: The ``Experiment`` object representing the current state of the
                experiment. The key properties includes ``trials``, ``search_space``,
                and ``optimization_config``. The data is provided as a separate arg.
            data: The data / metrics collected on the experiment so far.
        """

    @abstractmethod
    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """Get the parameters for the next candidate configuration to evaluate.

        Args:
            pending_parameters: A list of parameters of the candidates pending
                evaluation. This is often used to avoid generating duplicate candidates.

        Returns:
            A dictionary mapping parameter names to parameter values for the next
            candidate suggested by the method.
        """

    @property
    def _fitted_model(self) -> None:
        return None

    @property
    def model_spec_to_gen_from(self) -> GeneratorSpec | None:
        return self._model_spec_to_gen_from

    def _fit(
        self,
        experiment: Experiment,
        data: Data | None = None,
        **kwargs: Any,
    ) -> None:
        """A method used to initialize or update the experiment state / data
        on any surrogate models or predictors used during candidate generation.

        This method records the time spent during the update and defers to
        `update_generator_state` for the actual work.

        Args:
            experiment: The experiment to fit the surrogate model / predictor to.
            data: The experiment data used to fit the model.
            search_space: UNSUPPORTED. An optional override for the experiment
                search space.
            kwargs: UNSUPPORTED. Additional keyword arguments for model fitting.
        """
        if kwargs:
            raise UnsupportedError(
                "Unexpected arguments encountered. `ExternalGenerationNode._fit` only "
                "supports `experiment` and `data` arguments. "
                f"Each of the following arguments should be None / empty. {kwargs=}."
            )
        t_fit_start = time.monotonic()
        self.update_generator_state(
            experiment=experiment,
            data=data if data is not None else experiment.lookup_data(),
        )
        self.fit_time_since_gen += time.monotonic() - t_fit_start

    def _gen(
        self,
        experiment: Experiment,
        data: Data | None = None,
        n: int | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        **model_gen_kwargs: Any,
    ) -> GeneratorRun:
        """Generate new candidates for evaluation.

        This method calls `get_next_trial_parameterizations` to get the parameters
        for the next trial(s), and packages it as needed for higher level Ax APIs.
        If `should_deduplicate=True`, this also checks for duplicates and re-generates
        the parameters as needed.

        Args:
            n: Optional integer representing how many arms should be in the generator
                run produced by this method. Defaults to 1.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some methods to avoid
                re-suggesting candidates that are currently being evaluated.
            model_gen_kwargs: Keyword arguments, passed through to
                ``GeneratorSpec.gen``; these override any pre-specified in
                ``GeneratorSpec.model_gen_kwargs``.

        Returns:
            A ``GeneratorRun`` containing the newly generated candidates.
        """
        if self._model_spec_to_gen_from is not None:
            # This is the fallback case. Generate using base GNode logic.
            gr = super()._gen(
                experiment=experiment,
                data=data,
                n=n,
                pending_observations=pending_observations,
                **model_gen_kwargs,
            )
            # Unset self._model_spec_to_gen_from before returning.
            self._model_spec_to_gen_from = None
            return gr
        t_gen_start = time.monotonic()
        n = 1 if n is None else n
        pending_parameters: list[TParameterization] = []
        if pending_observations:
            for obs in pending_observations.values():
                for o in obs:
                    if o not in pending_parameters:
                        pending_parameters.append(o.parameters)
        generated_params: list[TParameterization] = []
        for _ in range(n):
            # NOTE: We could pass `experiment` and `data` to `get_next_candidate`
            # to make it possible for it to be stateless in more cases.
            params = self.get_next_candidate(pending_parameters=pending_parameters)
            generated_params.append(params)
            pending_parameters.append(params)
        # Return the parameterizations as a generator run.
        generator_run = GeneratorRun(
            arms=[Arm(parameters=params) for params in generated_params],
            fit_time=self.fit_time_since_gen,
            gen_time=time.monotonic() - t_gen_start,
            model_key=self.node_name,
        )
        # TODO: This shares the same bug as Adapter.gen. In both cases, after
        # deduplication, the generator run will record fit_time as 0.
        self.fit_time_since_gen = 0
        return generator_run
