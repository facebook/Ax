#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from logging import Logger
from typing import Any, cast, TYPE_CHECKING, Union

# TODO[@mgarrard, @drfreund]: Remove this when we streamline `apply_input_
# constructors`, such that they no longer need to access individual
# input constructor purposes.
import ax.generation_strategy as gs_module  # @manual

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import AxError, UserInputError
from ax.exceptions.generation_strategy import GenerationStrategyRepeatedPoints
from ax.exceptions.model import ModelError
from ax.generation_strategy.best_model_selector import BestModelSelector

if TYPE_CHECKING:
    from ax.generation_strategy.generation_node_input_constructors import (
        InputConstructorPurpose,
        NodeInputConstructors,
    )
    from ax.generation_strategy.generation_strategy import GenerationStrategy

from ax.adapter.base import Adapter
from ax.adapter.registry import (
    _extract_generator_state_after_gen,
    GeneratorRegistryBase,
    Generators,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGen,
    MaxGenerationParallelism,
    MinTrials,
    TransitionCriterion,
    TrialBasedCriterion,
)
from ax.utils.common.base import SortableBase
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import SerializationMixin
from pyre_extensions import none_throws


logger: Logger = get_logger(__name__)

MISSING_MODEL_SELECTOR_MESSAGE = (
    "A `BestModelSelector` must be provided when using multiple "
    "`GeneratorSpec`s in a `GenerationNode`. After fitting all `GeneratorSpec`s, "
    "the `BestModelSelector` will be used to select the `GeneratorSpec` to "
    "use for candidate generation."
)
MAX_GEN_ATTEMPTS = 5
MAX_GEN_ATTEMPTS_EXCEEDED_MESSAGE = (
    f"GenerationStrategy exceeded `MAX_GEN_ATTEMPTS` of {MAX_GEN_ATTEMPTS} while "
    "trying to generate a unique parameterization. This indicates that the search "
    "space has likely been fully explored, or that the sweep has converged."
)
DEFAULT_FALLBACK: dict[type[Exception], GeneratorSpec] = {
    cast(type[Exception], GenerationStrategyRepeatedPoints): GeneratorSpec(
        generator_enum=Generators.SOBOL, generator_key_override="Fallback_Sobol"
    )
}


class GenerationNode(SerializationMixin, SortableBase):
    """Base class for GenerationNode, capable of fitting one or more model specs under
    the hood and generating candidates from them.

    Args:
        name: A unique name for the GenerationNode. Used for storage purposes.
        generator_specs: A list of GeneratorSpecs to be selected from for generation
            in this GenerationNode.
        best_model_selector: A ``BestModelSelector`` used to select the
            ``GeneratorSpec`` to generate from in ``GenerationNode`` with
            multiple ``GeneratorSpec``s.
        should_deduplicate: Whether to deduplicate the parameters of proposed arms
            against those of previous arms via rejection sampling. If this is True,
            the GenerationStrategy will discard generator runs produced from the
            GenerationNode that has `should_deduplicate=True` if they contain arms
            already present on the experiment and replace them with new generator runs.
            If no generator run with entirely unique arms could be produced in 5
            attempts, a `GenerationStrategyRepeatedPoints` error will be raised, as we
            assume that the optimization converged when the model can no longer suggest
            unique arms.
        transition_criteria: List of TransitionCriterion, each of which describes a
            condition that must be met before completing a GenerationNode. All `is_met`
            must evaluate True for the GenerationStrategy to move on to the next
            GenerationNode.
        input_constructors: A dictionary mapping input constructor purpose enum to the
            input constructor enum. Each input constructor maps to a method which
            encodes the logic for determining dynamic inputs to the ``GenerationNode``
        trial_type: Specifies the type of trial to generate, is limited to either
            ``Keys.SHORT_RUN`` or ``Keys.LONG_RUN`` for now. If not specified, will
            default to None and not be used during generation.
        previous_node_name: The previous ``GenerationNode`` name in the
            ``GenerationStrategy``, if any. Initialized to None for all nodes, and is
            set during transition from one ``GenerationNode`` to the next. Can be
            overwritten if multiple transitions occur between nodes, and will always
            store the most recent previous ``GenerationNode`` name.
        should_skip: Whether to skip this node during generation time. Defaults to
            False, and can only currently be set to True via ``NodeInputConstructors``
        fallback_specs: Optional dict mapping expected exception types to `ModelSpec`
            fallbacks used when gen fails.

    Note for developers: by "model" here we really mean an Ax Adapter object, which
    contains an Ax Model under the hood. We call it "model" here to simplify and focus
    on explaining the logic of GenerationStep and GenerationStrategy.
    """

    # Required options:
    generator_specs: list[GeneratorSpec]
    # TODO: Move `should_deduplicate` to `GeneratorSpec` if possible, and make optional
    should_deduplicate: bool
    _name: str

    # Optional specifications
    _generator_spec_to_gen_from: GeneratorSpec | None = None
    # TODO: @mgarrard should this be a dict criterion_class name -> criterion mapping?
    _transition_criteria: Sequence[TransitionCriterion]
    _input_constructors: dict[
        InputConstructorPurpose,
        NodeInputConstructors,
    ]
    _previous_node_name: str | None = None
    _trial_type: str | None = None
    _should_skip: bool = False
    fallback_specs: dict[type[Exception], GeneratorSpec]

    # [TODO] Handle experiment passing more eloquently by enforcing experiment
    # attribute is set in generation strategies class
    _generation_strategy: None | GenerationStrategy = None

    def __init__(
        self,
        name: str,
        generator_specs: list[GeneratorSpec],
        best_model_selector: BestModelSelector | None = None,
        should_deduplicate: bool = False,
        transition_criteria: Sequence[TransitionCriterion] | None = None,
        input_constructors: None
        | (
            dict[
                InputConstructorPurpose,
                NodeInputConstructors,
            ]
        ) = None,
        previous_node_name: str | None = None,
        trial_type: str | None = None,
        should_skip: bool = False,
        fallback_specs: dict[type[Exception], GeneratorSpec] | None = None,
    ) -> None:
        self._name = name
        # Check that the model specs have unique model keys.
        generator_keys = {
            generator_spec.generator_key for generator_spec in generator_specs
        }
        if len(generator_keys) != len(generator_specs):
            raise UserInputError(
                "Model keys must be unique across all model specs in a GenerationNode."
            )
        if len(generator_specs) > 1 and best_model_selector is None:
            raise UserInputError(MISSING_MODEL_SELECTOR_MESSAGE)
        if trial_type is not None and (
            trial_type != Keys.SHORT_RUN and trial_type != Keys.LONG_RUN
        ):
            raise NotImplementedError(
                f"Trial type must be either {Keys.SHORT_RUN} or {Keys.LONG_RUN},"
                f" got {trial_type}."
            )
        # If possible, assign `_generator_spec_to_gen_from` right away, for use in
        # `__repr__`
        if len(generator_specs) == 1:
            self._generator_spec_to_gen_from = generator_specs[0]
        self.generator_specs = generator_specs
        self.best_model_selector = best_model_selector
        self.should_deduplicate = should_deduplicate
        self._transition_criteria = (
            transition_criteria if transition_criteria is not None else []
        )
        self._input_constructors = (
            input_constructors if input_constructors is not None else {}
        )
        self._previous_node_name = previous_node_name
        self._trial_type = trial_type
        self._should_skip = should_skip
        self.fallback_specs = (
            fallback_specs if fallback_specs is not None else DEFAULT_FALLBACK
        )

    def __eq__(self, other: SortableBase) -> bool:
        return SortableBase.__eq__(self, other)

    @property
    def name(self) -> str:
        """Returns the unique name of this GenerationNode"""
        return self._name

    @property
    def generator_spec_to_gen_from(self) -> GeneratorSpec:
        """Returns the cached `_generator_spec_to_gen_from` or gets it from
        `_pick_fitted_adapter_to_gen_from` and then caches and returns it
        """
        if self._generator_spec_to_gen_from is not None:
            return self._generator_spec_to_gen_from

        self._generator_spec_to_gen_from = self._pick_fitted_adapter_to_gen_from()
        return self._generator_spec_to_gen_from

    @property
    def generator_to_gen_from_name(self) -> str | None:
        """Returns the name of the generator that will be used for gen, if there is one.
        Otherwise, returns None.
        """
        if self._generator_spec_to_gen_from is not None:
            return self._generator_spec_to_gen_from.generator_key
        else:
            return None

    @property
    def generation_strategy(self) -> GenerationStrategy:
        """Returns a backpointer to the GenerationStrategy, useful for obtaining the
        experiment associated with this GenerationStrategy"""
        # TODO: @mgarrard remove this property once we make experiment a required
        # argument on GenerationStrategy
        if self._generation_strategy is None:
            raise ValueError(
                "Generation strategy has not been initialized on this node."
            )
        return none_throws(self._generation_strategy)

    @property
    def transition_criteria(self) -> Sequence[TransitionCriterion]:
        """Returns the sequence of TransitionCriteria that will be used to determine
        if this GenerationNode is complete and should transition to the next node.
        """
        return [] if self._transition_criteria is None else self._transition_criteria

    @property
    def input_constructors(
        self,
    ) -> dict[
        InputConstructorPurpose,
        NodeInputConstructors,
    ]:
        """Returns the input constructors that will be used to determine any dynamic
        inputs to this ``GenerationNode``.
        """
        return self._input_constructors if self._input_constructors is not None else {}

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment associated with this GenerationStrategy"""
        return self.generation_strategy.experiment

    @property
    def is_completed(self) -> bool:
        """Returns True if this GenerationNode is complete and should transition to
        the next node.
        """
        # TODO: @mgarrard make this logic more robust and general
        # We won't mark a node completed if it has an AutoTransitionAfterGen criterion
        # as this is typically used in cyclic generation strategies
        return self.should_transition_to_next_node(raise_data_required_error=False)[
            0
        ] and not any(
            isinstance(tc, AutoTransitionAfterGen) for tc in self.transition_criteria
        )

    @property
    def previous_node(self) -> GenerationNode | None:
        """Returns the previous ``GenerationNode``, if any."""
        return (
            self.generation_strategy.nodes_dict[self._previous_node_name]
            if self._previous_node_name is not None
            else None
        )

    @property
    def _unique_id(self) -> str:
        """Returns a unique (w.r.t. parent class: ``GenerationStrategy``) id
        for this GenerationNode. Used for SQL storage.
        """
        return self.name

    @property
    def _fitted_adapter(self) -> Adapter | None:
        """Private property to return optional fitted_adapter from
        self.generator_spec_to_gen_from for convenience. If no model is fit,
        this will return None.
        """
        try:
            # Using the private attribute since using the non-private `fitted_adapter`
            # property will raise a UserInputError if there is no fitted model.
            return self.generator_spec_to_gen_from._fitted_adapter
        except ModelError:
            # ModelError is raised if there are no fitted adapters to select from.
            return None

    def __repr__(self) -> str:
        """String representation of this ``GenerationNode`` (note that it
        will abridge some aspects of ``TransitionCriterion`` and
        ``GeneratorSpec`` attributes).
        """
        str_rep = f"{self.__class__.__name__}"
        str_rep += f"(name='{self.name}'"
        str_rep += ", generator_specs="
        generator_spec_str = (
            ", ".join([spec._brief_repr() for spec in self.generator_specs])
            .replace("\n", " ")
            .replace("\t", "")
        )
        str_rep += f"[{generator_spec_str}]"
        str_rep += (
            f", transition_criteria={str(self._brief_transition_criteria_repr())}"
        )
        return f"{str_rep})"

    def _fit(
        self,
        experiment: Experiment,
        data: Data | None = None,
        **kwargs: Any,
    ) -> None:
        """Fits the specified models to the given experiment + data using
        the model kwargs set on each corresponding model spec and the kwargs
        passed to this method.

        NOTE: During fitting of the ``GeneratorSpec``, state of this ``GeneratorSpec``
        after its last candidate generation is extracted from the last
        ``GeneratorRun`` it produced (if any was captured in
        ``GeneratorRun.generator_state_after_gen``) and passed into
        ``GeneratorSpec.fit`` as keyword arguments.

        Args:
            experiment: The experiment to fit the model to.
            data: The experiment data used to fit the model, optional (if not specified
                will use ``experiment.lookup_data()``, extracted in ``Adapter``).
            kwargs: Additional keyword arguments to pass to the model's
                ``fit`` method. NOTE: Local kwargs take precedence over the ones
                stored in ``GeneratorSpec.generator_kwargs``.
        """
        self._generator_spec_to_gen_from = None
        for generator_spec in self.generator_specs:
            try:
                # Stores the fitted model as `generator_spec._fitted_adapter`.
                generator_spec.fit(
                    experiment=experiment,
                    data=data,
                    **{
                        **self._get_model_state_from_last_generator_run(
                            generator_spec=generator_spec
                        ),
                        **kwargs,
                    },
                )
            except Exception as e:
                if len(self.generator_specs) == 1:
                    # If no other model specs, raise.
                    raise
                # If there are other model specs, try to handle gracefully.
                logger.exception(
                    "Model fitting failed for `GeneratorSpec` "
                    f"{generator_spec.generator_key}. Original error message: {e}."
                )
                # Discard any previously fitted models for this spec.
                generator_spec._fitted_adapter = None

    def gen(
        self,
        *,
        experiment: Experiment,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        skip_fit: bool = False,
        data: Data | None = None,
        **gs_gen_kwargs: Any,
    ) -> GeneratorRun | None:
        """This method generates candidates using `self._gen` and handles deduplication
        of generated candidates if `self.should_deduplicate=True`.

        NOTE: This method returning ``None`` indicates that this node should be skipped
            by its generation strategy.
        NOTE: ``n`` argument will be passed through this node's input constructors,
            which might modify the number of arms this node will produce. Also, some
            underlying generators  may ignore the ``n`` argument and produce a
            model-determined number of arms. In that case this method will also output
            a generator run with number of arms that may differ from ``n``.

        Args:
            experiment: The experiment to generate candidates for.
            data: Optional override for the experiment data used to generate candidates;
                if not specified, will use ``experiment.lookup_data()`` (extracted in
                ``Adapter``).
            pending_observations: A map from metric signature to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            generator_gen_kwargs: Keyword arguments, passed through to
                ``ModelSpec.gen``; these override any pre-specified in
                ``ModelSpec.generator_gen_kwargs``. Often will contain ``n``.

        Returns:
            A ``GeneratorRun`` containing the newly generated candidates or ``None``
            if this node is not in a correct state to generate candidates and should
            be skipped (e.g. if its input constructor for the ``n`` argument specifies
            that it should generate 0 candidate arms given the current experiment
            state and user inputs).
        """
        # TODO: Consider returning "should skip" from apply input constructors
        input_constructor_values = self.apply_input_constructors(
            experiment=experiment,
            gen_kwargs=gs_gen_kwargs,
        )
        # If during input constructor application we determined that we should skip
        # this node, return early.
        if self._should_skip:
            logger.debug(f"Skipping generation for node {self.name}.")
            return None

        if not skip_fit:
            self._fit(experiment=experiment, data=data)
        generator_gen_kwargs = gs_gen_kwargs.copy()
        generator_gen_kwargs.update(input_constructor_values)
        try:
            # Generate from the main generator on this node. If deduplicating,
            # keep generating until each of `generator_run.arms` is not a
            # duplicate of a previous active arm (e.g. not from a failed trial)
            # on the experiment.
            gr = self._gen_maybe_deduplicate(
                experiment=experiment,
                data=data,
                pending_observations=pending_observations,
                **generator_gen_kwargs,
            )
        except Exception as e:
            gr = self._try_gen_with_fallback(
                exception=e,
                experiment=experiment,
                data=data,
                pending_observations=pending_observations,
                **generator_gen_kwargs,
            )

        gr._generation_node_name = self.name
        # TODO: @mgarrard determine a more refined way to indicate trial type
        if self._trial_type is not None:
            gen_metadata = gr.gen_metadata if gr.gen_metadata is not None else {}
            gen_metadata["trial_type"] = self._trial_type
            gr._gen_metadata = gen_metadata
        return gr

    def _gen(
        self,
        experiment: Experiment,
        n: int | None,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        data: Data | None,
        **generator_gen_kwargs: Any,
    ) -> GeneratorRun:
        """Picks a fitted model, from which to generate candidates (via
        ``self._pick_fitted_adapter_to_gen_from``) and generates candidates
        from it. Uses the ``generator_gen_kwargs`` set on the selected ``GeneratorSpec``
        alongside any kwargs passed in to this function (with local kwargs)
        taking precedent.

        Args:
            experiment: The experiment to generate candidates for.
            data: The experiment data used to generate candidates (usually determined
                via ``experiment.lookup_data()`` in ``GenerationStrategy.gen``, but
                could be manually specified by a user).
            n: Optional integer representing how many arms should be in the generator
                run produced by this method. When this is ``None``, ``n`` will be
                determined by the ``GeneratorSpec`` that we are generating from.
            pending_observations: A map from metric signature to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            generator_gen_kwargs: Keyword arguments, passed through to
                ``GeneratorSpec.gen``;these override any pre-specified in
                ``GeneratorSpec.generator_gen_kwargs``.

        Returns:
            A ``GeneratorRun`` containing the newly generated candidates.
        """
        generator_spec = self.generator_spec_to_gen_from
        if n is None and generator_spec.generator_gen_kwargs:
            # If `n` is not specified, ensure that the `None` value does not
            # override the one set in `generator_spec.generator_gen_kwargs`.
            n = generator_spec.generator_gen_kwargs.get("n", None)
        return generator_spec.gen(
            experiment=experiment,
            data=data,
            n=n,
            # For `pending_observations`, prefer the input to this function, as
            # `pending_observations` are dynamic throughout the experiment and thus
            # unlikely to be specified in `generator_spec.generator_gen_kwargs`.
            pending_observations=pending_observations,
            **generator_gen_kwargs,
        )

    def _gen_maybe_deduplicate(
        self,
        experiment: Experiment,
        n: int | None,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        data: Data | None,
        **generator_gen_kwargs: Any,
    ) -> GeneratorRun:
        """Attempts to generate candidates from the main ``GeneratorSpec``
        on this ``GenerationNode``, with deduplication if
        ``self.should_deduplicate=True``. If maximum number of deduplication
        attempts is exceeded, raises ``GenerationStrategyRepeatedPoints``.

        NOTE: Should only ever be called from ``GenerationNode.gen``.
        """
        n_gen_draws = 0
        dedup_against_arms = experiment.arms_by_signature_for_deduplication
        # Keep generating until each of `generator_run.arms` is not a duplicate
        # of a previous arm, if `should_deduplicate is True`
        while n_gen_draws < MAX_GEN_ATTEMPTS:
            n_gen_draws += 1
            gr = self._gen(
                experiment=experiment,
                data=data,
                n=n,
                pending_observations=pending_observations,
                **generator_gen_kwargs,
            )
            if not self.should_deduplicate or not dedup_against_arms:
                return gr  # Not deduplicating.
            if all(arm.signature not in dedup_against_arms for arm in gr.arms):
                return gr  # Generated a set of all-non-duplicate arms.
            logger.debug(
                "The generator run produced duplicate arms. Re-running the "
                "generation step in an attempt to deduplicate. Candidates "
                f"produced in the last generator run: {gr.arms}."
            )

        raise GenerationStrategyRepeatedPoints(MAX_GEN_ATTEMPTS_EXCEEDED_MESSAGE)

    def _try_gen_with_fallback(
        self,
        exception: Exception,
        experiment: Experiment,
        n: int | None,
        data: Data | None,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        **generator_gen_kwargs: Any,
    ) -> GeneratorRun:
        """Attempts to generate candidates from the fallback ``GeneratorSpec``
        on this ``GenerationNode``. Identifies the correct fallback based on the
        type of ``Exception`` thrown by ``_gen_maybe_deduplicate``.

        NOTE: Should only ever be called from ``GenerationNode.gen``.
        """
        error_type = type(exception)
        if error_type not in self.fallback_specs:
            raise exception

        # identify fallback model to use
        fallback_model = self.fallback_specs[error_type]
        logger.warning(
            f"gen failed with error {exception}, "
            "switching to fallback model with generator_enum "
            f"{fallback_model.generator_enum}"
        )

        # Fit fallback model using information from the experiment as ground truth.
        fallback_model.fit(
            experiment=experiment,
            **self._get_model_state_from_last_generator_run(
                generator_spec=fallback_model
            ),
        )
        # Switch _generator_spec_to_gen_from to a fallback spec
        self._generator_spec_to_gen_from = fallback_model
        gr = self._gen(
            experiment=experiment,
            data=data,
            n=n,
            pending_observations=pending_observations,
            **generator_gen_kwargs,
        )
        return gr

    def _get_model_state_from_last_generator_run(
        self, generator_spec: GeneratorSpec
    ) -> dict[str, Any]:
        """Get the fit args from the last generator run for the model being fit.

        Args:
            generator_spec: The model spec to get the fit args for.

        Returns:
            A dictionary of fit args extracted from the last generator run
            that was generated by the model being fit.
        """
        if self._generation_strategy is None:
            # If there is no GS, we cannot access the previous GRs.
            return {}
        curr_model = generator_spec.generator_enum
        # Find the last GR that was generated by the model being fit.
        grs = self.generation_strategy._generator_runs
        for gr in reversed(grs):
            if (
                gr._generation_node_name == self.name
                and gr._generator_key == generator_spec.generator_key
            ):
                # Extract the fit args from the GR.
                return _extract_generator_state_after_gen(
                    generator_run=gr,
                    generator_class=curr_model.generator_class,
                )
        # No previous GR from this model.
        return {}

    # ------------------------- Model selection logic helpers. -------------------------

    def _pick_fitted_adapter_to_gen_from(self) -> GeneratorSpec:
        """Select one model to generate from among the fitted models on this
        generation node.

        NOTE: In base ``GenerationNode`` class, this method does the following:
          1. if this ``GenerationNode`` has an associated ``BestModelSelector``,
             use it to select one model to generate from among the fitted models
             on this generation node.
          2. otherwise, ensure that this ``GenerationNode`` only contains one
             `GeneratorSpec` and select it.
        """
        if self.best_model_selector is None:
            if (
                len(self.generator_specs) != 1
            ):  # pragma: no cover -- raised in __init__.
                raise UserInputError(MISSING_MODEL_SELECTOR_MESSAGE)
            return self.generator_specs[0]

        fitted_specs = [
            # Only select between models that were successfully fit.
            spec
            for spec in self.generator_specs
            if spec._fitted_adapter is not None
        ]
        if len(fitted_specs) == 0:
            raise ModelError(
                "No fitted models were found on the `GeneratorSpec`s. "
                "This can be caused by model fitting errors, which should be "
                "diagnosed by following the exception logs produced earlier."
            )
        try:
            best_model = none_throws(self.best_model_selector).best_model(
                generator_specs=fitted_specs,
            )
            return best_model
        except Exception as e:
            logger.warning(
                "The `BestModelSelector` raised an error when selecting the best "
                "generator. This can happen if the generator ran into issues during "
                "computing the relevant diagnostics, such as insufficient training "
                "data. Returning the first generator that was successfully fit. "
                f"Original error message: {e}."
            )
            return fitted_specs[0]

    # ------------------------- Trial logic helpers. -------------------------
    @property
    def trials_from_node(self) -> set[int]:
        """Returns a set containing the indices of trials generated by this node.

        Returns:
            Set[int]: A set containing all the indices of trials generated by this node.
        """
        trials_from_node = set()
        for _idx, trial in self.experiment.trials.items():
            for gr in trial.generator_runs:
                if (
                    gr._generation_node_name is not None
                    and gr._generation_node_name == self.name
                ):
                    trials_from_node.add(trial.index)
        return trials_from_node

    @property
    def node_that_generated_last_gr(self) -> str | None:
        """Returns the name of the node that generated the last generator run.

        Returns:
            str: The name of the node that generated the last generator run.
        """
        return (
            self.generation_strategy.last_generator_run._generation_node_name
            if self.generation_strategy.last_generator_run
            else None
        )

    @property
    def transition_edges(self) -> dict[str, list[TransitionCriterion]]:
        """Returns a dictionary mapping the next ``GenerationNode`` to the
        TransitionCriteria that define the transition that that node.

        Ex: if the transition from the current node to node `x` is defined by
        IsSingleObjective and MinTrials criterion then the return would be
        {'x': [IsSingleObjective, MinTrials]}.

        Returns:
            Dict[str, List[TransitionCriterion]]: A dictionary mapping the next
            ``GenerationNode`` to the ``TransitionCriterion`` that are associated
            with it.
        """
        if self.transition_criteria is None:
            return {}

        tc_edges = defaultdict(list)
        for tc in self.transition_criteria:
            tc_edges[tc.transition_to].append(tc)
        return tc_edges

    def should_transition_to_next_node(
        self, raise_data_required_error: bool = True
    ) -> tuple[bool, str]:
        """Checks whether we should transition to the next node based on this node's
        TransitionCriterion.

        Important: This method relies on the ``transition_criterion`` of this node to
        be listed in order of importance. Ex: a fallback transition should come after
        the primary transition in the transition criterion list.

        Args:
            raise_data_required_error: Whether to raise ``DataRequiredError`` in the
                case detailed above. Not raising the error is useful if just looking to
                check how many generator runs (to be made into trials) can be produced,
                but not actually producing them yet.
        Returns:
            Tuple[bool, str]: Whether we should transition to the next node
                and the name of the node to gen from (either the current or next node)
        """
        # if no transition criteria are defined, this node can generate unlimited trials
        if len(self.transition_criteria) == 0:
            return False, self.name

        # For each "transition edge" (set of all transition criteria that lead from
        # current node (e.g. "node A") to another specific node ("e.g. "node B")
        # in the node DAG:
        # I. Check if all of the transition criteria along that edge are met; if so,
        # transition to the next node defined by that edge.
        # II. If we did not transition along this edge, but the edge has some
        # "generation blocking" transition criteria (ex `MaxGenerationParallelism`)
        # that are met, raise the error associated with that criterion.
        for next_node, all_tc in self.transition_edges.items():
            # I. Check if there are any TCs that block transition and whether all
            # of them are met. If all of them are met, then we should transition.
            transition_blocking = [tc for tc in all_tc if tc.block_transition_if_unmet]
            all_transition_blocking_met_should_transition = transition_blocking and all(
                tc.is_met(
                    experiment=self.experiment,
                    curr_node=self,
                )
                for tc in transition_blocking
            )
            if all_transition_blocking_met_should_transition:
                return True, next_node

            # II. Raise any necessary generation errors: for any met criterion,
            # call its `block_continued_generation_error` method if not all
            # transition-blocking criteria are met. The method might not raise an
            # error, depending on its implementation on given criterion, so the error
            # from the first met one that does block continued generation, will raise.
            if raise_data_required_error:
                generation_blocking = [tc for tc in all_tc if tc.block_gen_if_met]
                for tc in generation_blocking:
                    if tc.is_met(self.experiment, curr_node=self):
                        tc.block_continued_generation_error(
                            node_name=self.name,
                            experiment=self.experiment,
                            trials_from_node=self.trials_from_node,
                        )
                # TODO[@mgarrard, @drfreund] Try replacing `block_gen_if_met` with
                # a self-transition and rework this error block.

        return False, self.name

    def new_trial_limit(self, raise_generation_errors: bool = False) -> int:
        """How many trials can this generation strategy can currently produce
        ``GeneratorRun``-s for (with potentially multiple generator runs produced for
        each intended trial).

        NOTE: Only considers transition criteria that inherit from
        ``TrialBasedCriterion``.

        Returns:
            The number of generator runs that can currently be produced, with -1
            meaning unlimited generator runs.
        """
        # TODO: @mgarrard Should we consider returning `None` if there is no limit?
        trial_based_gen_blocking_criteria = [
            criterion
            for criterion in self.transition_criteria
            if criterion.block_gen_if_met and isinstance(criterion, TrialBasedCriterion)
        ]
        # Cache trials_from_node to avoid repeated computation.
        trials_from_node = self.trials_from_node
        gen_blocking_criterion_delta_from_threshold = [
            criterion.num_till_threshold(
                experiment=self.experiment, trials_from_node=trials_from_node
            )
            for criterion in trial_based_gen_blocking_criteria
        ]

        # Raise any necessary generation errors: for any met criterion,
        # call its `block_continued_generation_error` method The method might not
        # raise an error, depending on its implementation on given criterion, so the
        # error from the first met one that does block continued generation, will be
        # raised.
        if raise_generation_errors:
            for criterion in trial_based_gen_blocking_criteria:
                # TODO[mgarrard]: Raise a group of all the errors, from each gen-
                # blocking transition criterion.
                if criterion.is_met(
                    self.experiment,
                    curr_node=self,
                ):
                    criterion.block_continued_generation_error(
                        node_name=self.name,
                        experiment=self.experiment,
                        trials_from_node=self.trials_from_node,
                    )
        if len(gen_blocking_criterion_delta_from_threshold) == 0:
            return -1
        return max(min(gen_blocking_criterion_delta_from_threshold), -1)

    def _brief_transition_criteria_repr(self) -> str:
        """Returns a brief string representation of the
        transition criteria for this node.

        Returns:
            str: A string representation of the transition criteria for this node.
        """
        if self.transition_criteria is None:
            return "None"
        tc_list = ", ".join(
            [
                f"{tc.__class__.__name__}(transition_to='{str(tc.transition_to)}')"
                for tc in self.transition_criteria
            ]
        )
        return f"[{tc_list}]"

    def apply_input_constructors(
        self,
        experiment: Experiment,
        gen_kwargs: dict[str, Any],
    ) -> dict[str, Union[int, ObservationFeatures | None]]:
        # NOTE: In the future we might have to add new types to the `Union` above
        # or allow `Any` for the value type, but until we have more different types
        # of input constructors, this provides a bit of additional type checking.
        return {
            "n": self._determine_arms_from_node(
                experiment=experiment,
                gen_kwargs=gen_kwargs,
            ),
            "fixed_features": self._determine_fixed_features_from_node(
                experiment=experiment,
                gen_kwargs=gen_kwargs,
            ),
        }

    def _determine_arms_from_node(
        self,
        experiment: Experiment,
        gen_kwargs: dict[str, Any],
    ) -> int:
        """Calculates the number of arms to generate from the node that will be used
        during generation.

        Args:
            gen_kwargs: The kwargs passed to the ``GenerationStrategy``'s
                gen call, including arms_per_node: an optional map from node name to
                the number of arms to generate from that node. If not provided, will
                default to the number of arms specified in the node's
                ``InputConstructors`` or n if no``InputConstructors`` are defined on
                the node.

        Returns:
            The number of arms to generate from the node that will be used during this
            generation via ``_gen_multiple``.
        """
        arms_per_node = gen_kwargs.get("arms_per_node")
        purpose_N = (
            gs_module.generation_node_input_constructors.InputConstructorPurpose.N
        )
        if arms_per_node is not None:
            # arms_per_node provides a way to manually override input
            # constructors. This should be used with caution, and only
            # if you really know what you're doing. :)
            arms_from_node = arms_per_node[self.name]
        elif purpose_N not in self.input_constructors:
            # if the node does not have an input constructor for N, we first
            # use the n passed to method if it exists, then check if the model gen
            # kwargs define an n, then fallback to default values of n
            arms_from_node = gen_kwargs.get("n")
            if arms_from_node is None and self.generator_spec_to_gen_from is not None:
                arms_from_node = (
                    self.generator_spec_to_gen_from.generator_gen_kwargs.get("n", None)
                )
            if arms_from_node is None:
                # TODO[@mgarrard, @drfreund]: We can remove this check if we
                # decide that generation nodes can only be used within a
                # generation strategy; then we will need to refactor some tests.
                if self._generation_strategy is not None:
                    arms_from_node = self._generation_strategy.DEFAULT_N
                else:
                    arms_from_node = 1
        else:
            arms_from_node = self.input_constructors[purpose_N](
                previous_node=self.previous_node,
                next_node=self,
                gs_gen_call_kwargs=gen_kwargs,
                experiment=experiment,
            )

        return arms_from_node

    def _determine_fixed_features_from_node(
        self,
        experiment: Experiment,
        gen_kwargs: dict[str, Any],
    ) -> ObservationFeatures | None:
        """Uses the ``InputConstructors`` on the node to determine the fixed features
        to pass into the model. If fixed_features are provided, they will take
        precedence over the fixed_features from the node.

        Args:
            node_to_gen_from: The node from which to generate from
            gen_kwargs: The kwargs passed to the ``GenerationStrategy``'s
                gen call, including the fixed features passed to the ``gen`` method if
                any.

        Returns:
            An object of ObservationFeatures that represents the fixed features to
            pass into the model.
        """
        node_fixed_features = None
        # passed_fixed_features represents the fixed features that were passed by the
        # user to the gen method as overrides.
        passed_fixed_features = gen_kwargs.get("fixed_features")
        if passed_fixed_features is not None:
            node_fixed_features = passed_fixed_features
        else:
            input_constructors_module = gs_module.generation_node_input_constructors
            purpose_fixed_features = (
                input_constructors_module.InputConstructorPurpose.FIXED_FEATURES
            )
            if purpose_fixed_features in self.input_constructors:
                node_fixed_features = self.input_constructors[purpose_fixed_features](
                    previous_node=self.previous_node,
                    next_node=self,
                    gs_gen_call_kwargs=gen_kwargs,
                    experiment=experiment,
                )
        # also pass default parameter values as fixed features for disabled parameters
        disabled_parameters_parameterization = {
            name: parameter.default_value
            for name, parameter in experiment.search_space.parameters.items()
            if parameter.is_disabled
        }
        if len(disabled_parameters_parameterization) == 0:
            return node_fixed_features

        if node_fixed_features is None:
            return ObservationFeatures(parameters=disabled_parameters_parameterization)

        return node_fixed_features.clone(
            replace_parameters={
                **disabled_parameters_parameterization,
                **node_fixed_features.parameters,
            }
        )


class GenerationStep(GenerationNode, SortableBase):
    """One step in the generation strategy, corresponds to a single generator.
    Describes the generator, how many trials will be generated with this generator, what
    minimum number of observations is required to proceed to the next generator, etc.

    Args:
        generator: A member of `Generators` enum returning an instance of
            `Adapter` with an instantiated underlying `Generator`.
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
        use_update: DEPRECATED.
        enforce_num_trials: Whether to enforce that only `num_trials` are generated
            from the given step. If False and `num_trials` have been generated, but
            `min_trials_observed` have not been completed, `generation_strategy.gen`
            will continue generating trials from the current step, exceeding `num_
            trials` for it. Allows to avoid `DataRequiredError`, but delays
            proceeding to next generation step.
        generator_kwargs: Dictionary of kwargs to pass into the adapter and generator
            constructors on instantiation. E.g. if `generator` is `Generators.SOBOL`,
            kwargs will be applied as `Generators.SOBOL(**generator_kwargs)`.
            NOTE: If generation strategy is interrupted and resumed from a stored
            snapshot and its last used generator has state saved on its generator runs,
            `generator_kwargs` is updated with the state dict of the generator,
            retrieved from the last generator run of this generation strategy.
        generator_gen_kwargs: Each call to `generation_strategy.gen` performs a call
            to the step's adapter's `gen` under the hood; `generator_gen_kwargs` will be
            passed to the adapter's `gen` like: `adapter.gen(**generator_gen_kwargs)`.
        completion_criteria: List of TransitionCriterion. All `is_met` must evaluate
            True for the GenerationStrategy to move on to the next Step
        index: Index of this generation step, for use internally in `Generation
            Strategy`. Do not assign as it will be reassigned when instantiating
            `GenerationStrategy` with a list of its steps.
        should_deduplicate: Whether to deduplicate the parameters of proposed arms
            against those of previous arms via rejection sampling. If this is True,
            the generation strategy will discard generator runs produced from the
            generation step that has `should_deduplicate=True` if they contain arms
            already present on the experiment and replace them with new generator runs.
            If no generator run with entirely unique arms could be produced in 5
            attempts, a `GenerationStrategyRepeatedPoints` error will be raised, as we
            assume that the optimization converged when the model can no longer suggest
            unique arms.
        generator_name: Optional name of the generator. If not specified, defaults to
            the `model_key` of the generator spec.
        use_all_trials_in_exp: Whether to use all trials in the experiment to determine
            whether to transition to the next step. If False, `num_trials` and
            `min_trials_observed` will only count trials generatd by this step. If True,
            they will count all trials in the experiment (of corresponding statuses).

    Note for developers: by "generator" here we really mean an ``Adapter`` object, which
    contains a ``Generator`` under the hood. We call it "generator" here to simplify and
    focus on explaining the logic of GenerationStep and GenerationStrategy.
    """

    def __init__(
        self,
        generator: GeneratorRegistryBase,
        num_trials: int,
        generator_kwargs: dict[str, Any] | None = None,
        generator_gen_kwargs: dict[str, Any] | None = None,
        completion_criteria: Sequence[TransitionCriterion] | None = None,
        min_trials_observed: int = 0,
        max_parallelism: int | None = None,
        enforce_num_trials: bool = True,
        should_deduplicate: bool = False,
        generator_name: str | None = None,
        use_all_trials_in_exp: bool = False,
        use_update: bool = False,  # DEPRECATED.
        index: int = -1,  # Index of this step, set internally.
        # Deprecated arguments for backwards compatibility.
        model_kwargs: dict[str, Any] | None = None,
        model_gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        r"""Initializes a single-model GenerationNode, a.k.a. a GenerationStep.

        See the class docstring for argument descriptions.
        """
        if use_update:
            raise DeprecationWarning("`GenerationStep.use_update` is deprecated.")
        # These are here for backwards compatibility. Prior to implementation of
        # the __init__ method, these were the fields of the dataclass. GenerationStep
        # storage utilizes these attributes, so we need to store them. Once we start
        # using GenerationNode storage, we can clean up these attributes.
        self.index = index
        self.generator = generator
        self.num_trials = num_trials
        self.completion_criteria: Sequence[TransitionCriterion] = (
            completion_criteria or []
        )
        self.min_trials_observed = min_trials_observed
        self.max_parallelism = max_parallelism
        self.enforce_num_trials = enforce_num_trials
        self.use_update = use_update

        generator_kwargs = generator_kwargs or {}
        generator_gen_kwargs = generator_gen_kwargs or {}

        if (
            self.enforce_num_trials
            and (self.num_trials >= 0)
            and (self.min_trials_observed > self.num_trials)
        ):
            raise UserInputError(
                "`GenerationStep` received `min_trials_observed > num_trials` "
                f"(`min_trials_observed = {self.min_trials_observed}`, `num_trials = "
                f"{self.num_trials}`), making completion of this step impossible. "
                "Please alter inputs so that `min_trials_observed <= num_trials`."
            )
        if not isinstance(self.generator, GeneratorRegistryBase):
            raise UserInputError(
                "`generator` in generation step must be a `GeneratorRegistryBase` "
                "enum subclass entry. Support for alternative callables "
                f"has been deprecated. Got {self.generator=}."
            )
        else:
            # Pass deprecated arguments to GeneratorSpec which handles them.
            generator_spec = GeneratorSpec(
                generator_enum=self.generator,
                generator_kwargs=generator_kwargs,
                generator_gen_kwargs=generator_gen_kwargs,
                model_kwargs=model_kwargs,
                model_gen_kwargs=model_gen_kwargs,
            )
        if not generator_name:
            generator_name = generator_spec.generator_key
        self.generator_name: str = generator_name

        # Create transition criteria for this step. If num_trials is provided to
        # this `GenerationStep`, then we create a `MinTrials` criterion which ensures
        # at least that many trials in good status are generated. `MinTrials` can also
        # enforce the min_trials_observed requirement. The `transition_to` argument
        # is set in `GenerationStrategy` constructor, because only then is the order
        # of the generation steps actually known.
        transition_criteria = []
        if self.num_trials != -1:
            transition_criteria.append(
                MinTrials(
                    threshold=self.num_trials,
                    not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                    block_gen_if_met=self.enforce_num_trials,
                    block_transition_if_unmet=True,
                    use_all_trials_in_exp=use_all_trials_in_exp,
                )
            )

        if self.min_trials_observed > 0:
            transition_criteria.append(
                MinTrials(
                    only_in_statuses=[
                        TrialStatus.COMPLETED,
                        TrialStatus.EARLY_STOPPED,
                    ],
                    threshold=self.min_trials_observed,
                    block_gen_if_met=False,
                    block_transition_if_unmet=True,
                    use_all_trials_in_exp=use_all_trials_in_exp,
                )
            )
        max_parallelism = self.max_parallelism
        if max_parallelism is not None:
            transition_criteria.append(
                MaxGenerationParallelism(
                    threshold=max_parallelism,
                    only_in_statuses=[TrialStatus.RUNNING],
                    block_gen_if_met=True,
                    block_transition_if_unmet=False,
                    transition_to=None,
                )
            )

        transition_criteria += self.completion_criteria
        super().__init__(
            name=f"GenerationStep_{str(self.index)}",
            generator_specs=[generator_spec],
            should_deduplicate=should_deduplicate,
            transition_criteria=transition_criteria,
        )

    @property
    def generator_kwargs(self) -> dict[str, Any]:
        """Returns the generator kwargs of the underlying ``GeneratorSpec``."""
        return self.generator_spec.generator_kwargs

    @property
    def generator_gen_kwargs(self) -> dict[str, Any]:
        """Returns the model gen kwargs of the underlying ``GeneratorSpec``."""
        return self.generator_spec.generator_gen_kwargs

    @property
    def generator_spec(self) -> GeneratorSpec:
        """Returns the first generator_spec from the generator_specs attribute."""
        return self.generator_specs[0]

    @property
    def _unique_id(self) -> str:
        """Returns the unique ID of this generation step, which is the index."""
        return str(self.index)

    def gen(
        self,
        *,
        experiment: Experiment,
        n: int | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        skip_fit: bool = False,
        data: Data | None = None,
        **gs_gen_kwargs: Any,
    ) -> GeneratorRun | None:
        gr = super().gen(
            experiment=experiment,
            n=n,
            pending_observations=pending_observations,
            data=data,
            skip_fit=skip_fit,
            **gs_gen_kwargs,
        )
        if gr is None:
            raise AxError(
                "This `GenerationStep`'s underlying `GenerationNode` returned "
                "`None` from `gen`. This is an unexpected state for a "
                "`GenerationStep`-based generation strategy."
            )
        gr._generation_step_index = self.index
        return gr
