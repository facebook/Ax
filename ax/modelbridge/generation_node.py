#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Module-level import to avoid circular dependency b/w this file and
# generation_strategy.py
from ax import modelbridge
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UserInputError
from ax.exceptions.generation_strategy import GenerationStrategyRepeatedPoints
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.cross_validation import BestModelSelector
from ax.modelbridge.model_spec import FactoryFunctionModelSpec, ModelSpec
from ax.modelbridge.registry import ModelRegistryBase
from ax.modelbridge.transition_criterion import (
    MaxGenerationParallelism,
    MaxTrials,
    MinTrials,
    TransitionCriterion,
    TrialBasedCriterion,
)
from ax.utils.common.base import Base, SortableBase
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import SerializationMixin
from ax.utils.common.typeutils import not_none


logger: Logger = get_logger(__name__)

TModelFactory = Callable[..., ModelBridge]
CANNOT_SELECT_ONE_MODEL_MSG = """\
Base `GenerationNode` does not implement selection among fitted \
models, so exactly one `ModelSpec` must be specified when using \
`GenerationNode._pick_fitted_model_to_gen_from` (usually called \
by `GenerationNode.gen`.
"""
MAX_GEN_DRAWS = 5
MAX_GEN_DRAWS_EXCEEDED_MESSAGE = (
    f"GenerationStrategy exceeded `MAX_GEN_DRAWS` of {MAX_GEN_DRAWS} while trying to "
    "generate a unique parameterization. This indicates that the search space has "
    "likely been fully explored, or that the sweep has converged."
)


class GenerationNode(SerializationMixin, SortableBase):
    """Base class for GenerationNode, capable of fitting one or more model specs under
    the hood and generating candidates from them.

    Args:
        model_specs: A list of ModelSpecs to be selected from for generation in this
            GenerationNode
        should_deduplicate: Whether to deduplicate the parameters of proposed arms
            against those of previous arms via rejection sampling. If this is True,
            the GenerationStrategy will discard generator runs produced from the
            GenerationNode that has `should_deduplicate=True` if they contain arms
            already present on the experiment and replace them with new generator runs.
            If no generator run with entirely unique arms could be produced in 5
            attempts, a `GenerationStrategyRepeatedPoints` error will be raised, as we
            assume that the optimization converged when the model can no longer suggest
            unique arms.
        node_name: A unique name for the GenerationNode. Used for storage purposes.
        transition_criteria: List of TransitionCriterion, each of which describes a
            condition that must be met before completing a GenerationNode. All `is_met`
            must evaluateTrue for the GenerationStrategy to move on to the next
            GenerationNode.

    Note for developers: by "model" here we really mean an Ax ModelBridge object, which
    contains an Ax Model under the hood. We call it "model" here to simplify and focus
    on explaining the logic of GenerationStep and GenerationStrategy.
    """

    # Required options:
    model_specs: List[ModelSpec]
    # TODO: Move `should_deduplicate` to `ModelSpec` if possible, and make optional
    should_deduplicate: bool
    _node_name: str

    # Optional specifications
    _model_spec_to_gen_from: Optional[ModelSpec] = None
    # TODO: @mgarrard should this be a dict criterion_class name -> criterion mapping?
    _transition_criteria: Optional[Sequence[TransitionCriterion]]

    # [TODO] Handle experiment passing more eloquently by enforcing experiment
    # attribute is set in generation strategies class
    _generation_strategy: Optional[
        modelbridge.generation_strategy.GenerationStrategy
    ] = None

    def __init__(
        self,
        node_name: str,
        model_specs: List[ModelSpec],
        best_model_selector: Optional[BestModelSelector] = None,
        should_deduplicate: bool = False,
        transition_criteria: Optional[Sequence[TransitionCriterion]] = None,
    ) -> None:
        self._node_name = node_name
        # While `GenerationNode` only handles a single `ModelSpec` in the `gen`
        # and `_pick_fitted_model_to_gen_from` methods, we validate the
        # length of `model_specs` in `_pick_fitted_model_to_gen_from` in order
        # to not require all `GenerationNode` subclasses to override an `__init__`
        # method to bypass that validation.
        self.model_specs = model_specs
        self.best_model_selector = best_model_selector
        self.should_deduplicate = should_deduplicate
        self._transition_criteria = transition_criteria

    @property
    def node_name(self) -> str:
        """Returns the unique name of this GenerationNode"""
        return self._node_name

    @property
    def model_spec_to_gen_from(self) -> ModelSpec:
        """Returns the cached `_model_spec_to_gen_from` or gets it from
        `_pick_fitted_model_to_gen_from` and then caches and returns it
        """
        if self._model_spec_to_gen_from is not None:
            return self._model_spec_to_gen_from

        self._model_spec_to_gen_from = self._pick_fitted_model_to_gen_from()
        return self._model_spec_to_gen_from

    @property
    def model_to_gen_from_name(self) -> Optional[str]:
        """Returns the name of the model that will be used for gen, if there is one.
        Otherwise, returns None.
        """
        if self._model_spec_to_gen_from is not None:
            return self._model_spec_to_gen_from.model_key
        else:
            return None

    @property
    def generation_strategy(self) -> modelbridge.generation_strategy.GenerationStrategy:
        """Returns a backpointer to the GenerationStrategy, useful for obtaining the
        experiment associated with this GenerationStrategy"""
        # TODO: @mgarrard remove this property once we make experiment a required
        # arguement on GenerationStrategy
        if self._generation_strategy is None:
            raise ValueError(
                "Generation strategy has not been initialized on this node."
            )
        return not_none(self._generation_strategy)

    @property
    def transition_criteria(self) -> Sequence[TransitionCriterion]:
        """Returns the sequence of TransitionCriteria that will be used to determine
        if this GenerationNode is complete and should transition to the next node.
        """
        return [] if self._transition_criteria is None else self._transition_criteria

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment associated with this GenerationStrategy"""
        return self.generation_strategy.experiment

    @property
    def is_completed(self) -> bool:
        """Returns True if this GenerationNode is complete and should transition to
        the next node.
        """
        return self.should_transition_to_next_node(raise_data_required_error=False)[0]

    @property
    def _unique_id(self) -> str:
        """Returns a unique (w.r.t. parent class: ``GenerationStrategy``) id
        for this GenerationNode. Used for SQL storage.
        """
        return self.node_name

    @property
    def _fitted_model(self) -> Optional[ModelBridge]:
        """Private property to return optional fitted_model from
        self.model_spec_to_gen_from for convenience. If no model is fit,
        will return None. If using the non-private `fitted_model` property,
        and no model is fit, a UserInput error will be raised.
        """
        return self.model_spec_to_gen_from._fitted_model

    def fit(
        self,
        experiment: Experiment,
        data: Data,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Fits the specified models to the given experiment + data using
        the model kwargs set on each corresponding model spec and the kwargs
        passed to this method.

        Args:
            experiment: The experiment to fit the model to.
            data: The experiment data used to fit the model.
            search_space: An optional overwrite for the experiment search space.
            optimization_config: An optional overwrite for the experiment
                optimization config.
            kwargs: Additional keyword arguments to pass to the model's
                ``fit`` method. NOTE: Local kwargs take precedence over the ones
                stored in ``ModelSpec.model_kwargs``.
        """
        self._model_spec_to_gen_from = None
        for model_spec in self.model_specs:
            model_spec.fit(  # Stores the fitted model as `model_spec._fitted_model`
                experiment=experiment,
                data=data,
                search_space=search_space,
                optimization_config=optimization_config,
                **kwargs,
            )

    # TODO [drfreund]: Move this up to `GenerationNodeInterface` once implemented.
    def gen(
        self,
        n: Optional[int] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        max_gen_draws_for_deduplication: int = MAX_GEN_DRAWS,
        arms_by_signature_for_deduplication: Optional[Dict[str, Arm]] = None,
        **model_gen_kwargs: Any,
    ) -> GeneratorRun:
        """This method generates candidates using `self._gen` and handles deduplication
        of generated candidates if `self.should_deduplicate=True`.

        NOTE: Models must have been fit prior to calling ``gen``.
        NOTE: Some underlying models may ignore the ``n`` argument and produce a
            model-determined number of arms. In that case this method will also output
            a generator run with number of arms that may differ from ``n``.

        Args:
            n: Optional integer representing how many arms should be in the generator
                run produced by this method. When this is ``None``, ``n`` will be
                determined by the ``ModelSpec`` that we are generating from.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            max_gen_draws_for_deduplication: Maximum number of attempts for generating
                new candidates without duplicates. If non-duplicate candidates are not
                generated with these attempts, a ``GenerationStrategyRepeatedPoints``
                exception will be raised.
            arms_by_signature_for_deduplication: A dictionary mapping arm signatures to
                the arms, to be used for deduplicating newly generated arms.
            model_gen_kwargs: Keyword arguments, passed through to ``ModelSpec.gen``;
                these override any pre-specified in ``ModelSpec.model_gen_kwargs``.

        Returns:
            A ``GeneratorRun`` containing the newly generated candidates.
        """
        should_generate_run = True
        generator_run = None
        n_gen_draws = 0
        # Keep generating until each of `generator_run.arms` is not a duplicate
        # of a previous arm, if `should_deduplicate is True`
        while should_generate_run:
            generator_run = self._gen(
                n=n,
                pending_observations=pending_observations,
                **model_gen_kwargs,
            )
            should_generate_run = (
                self.should_deduplicate
                and arms_by_signature_for_deduplication
                and any(
                    arm.signature in arms_by_signature_for_deduplication
                    for arm in generator_run.arms
                )
            )
            n_gen_draws += 1
            if should_generate_run:
                if n_gen_draws > max_gen_draws_for_deduplication:
                    raise GenerationStrategyRepeatedPoints(
                        MAX_GEN_DRAWS_EXCEEDED_MESSAGE
                    )
                else:
                    logger.info(
                        "The generator run produced duplicate arms. Re-running the "
                        "generation step in an attempt to deduplicate. Candidates "
                        f"produced in the last generator run: {generator_run.arms}."
                    )
        assert generator_run is not None, (
            "The GeneratorRun is None which is an unexpected state of this"
            " GenerationStrategy. This occured on GenerationNode: {self.node_name}."
        )
        generator_run._generation_node_name = self.node_name
        return generator_run

    def _gen(
        self,
        n: Optional[int] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        **model_gen_kwargs: Any,
    ) -> GeneratorRun:
        """Picks a fitted model, from which to generate candidates (via
        ``self._pick_fitted_model_to_gen_from``) and generates candidates
        from it. Uses the ``model_gen_kwargs`` set on the selected ``ModelSpec``
        alongside any kwargs passed in to this function (with local kwargs)
        taking precedent.

        Args:
            n: Optional integer representing how many arms should be in the generator
                run produced by this method. When this is ``None``, ``n`` will be
                determined by the ``ModelSpec`` that we are generating from.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            model_gen_kwargs: Keyword arguments, passed through to ``ModelSpec.gen``;
                these override any pre-specified in ``ModelSpec.model_gen_kwargs``.

        Returns:
            A ``GeneratorRun`` containing the newly generated candidates.
        """
        model_spec = self.model_spec_to_gen_from
        if n is None and model_spec.model_gen_kwargs:
            # If `n` is not specified, ensure that the `None` value does not
            # override the one set in `model_spec.model_gen_kwargs`.
            n = model_spec.model_gen_kwargs.get("n", None)
        return model_spec.gen(
            n=n,
            # For `pending_observations`, prefer the input to this function, as
            # `pending_observations` are dynamic throughout the experiment and thus
            # unlikely to be specified in `model_spec.model_gen_kwargs`.
            pending_observations=pending_observations,
            **model_gen_kwargs,
        )

    # ------------------------- Model selection logic helpers. -------------------------

    def _pick_fitted_model_to_gen_from(self) -> ModelSpec:
        """Select one model to generate from among the fitted models on this
        generation node.

        NOTE: In base ``GenerationNode`` class, this method does the following:
          1. if this ``GenerationNode`` has an associated ``BestModelSelector``,
             use it to select one model to generate from among the fitted models
             on this generation node.
          2. otherwise, ensure that this ``GenerationNode`` only contains one
             `ModelSpec` and select it.
        """
        if self.best_model_selector is None:
            if len(self.model_specs) != 1:
                raise NotImplementedError(CANNOT_SELECT_ONE_MODEL_MSG)
            return self.model_specs[0]

        for model_spec in self.model_specs:
            model_spec.cross_validate()

        best_model_index = not_none(self.best_model_selector).best_diagnostic(
            diagnostics=[not_none(m.diagnostics) for m in self.model_specs],
        )
        return self.model_specs[best_model_index]

    # ------------------------- Trial logic helpers. -------------------------
    @property
    def trials_from_node(self) -> Set[int]:
        """Returns a dictionary mapping a GenerationNode to the trials it generated.

        Returns:
            Set[int]: A set containing all the trials indices generated by this node.
        """
        trials_from_node = set()
        for _idx, trial in self.experiment.trials.items():
            for gr in trial.generator_runs:
                if (
                    gr._generation_node_name is not None
                    and gr._generation_node_name == self.node_name
                ):
                    trials_from_node.add(trial.index)
        return trials_from_node

    def should_transition_to_next_node(
        self, raise_data_required_error: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Checks whether we should transition to the next node based on this node's
        TransitionCriterion

        Args:
            raise_data_required_error: Whether to raise ``DataRequiredError`` in the
                case detailed above. Not raising the error is useful if just looking to
                check how many generator runs (to be made into trials) can be produced,
                but not actually producing them yet.
        Returns:
            bool: Whether we should transition to the next node.
        """
        # if no transition criteria are defined, this node can generate unlimited trials
        if len(self.transition_criteria) == 0:
            return False, None

        transition_blocking = [
            tc for tc in self.transition_criteria if tc.block_transition_if_unmet
        ]
        transition_blocking_met = all(
            tc.is_met(
                experiment=self.experiment,
                trials_from_node=self.trials_from_node,
            )
            for tc in transition_blocking
        )
        # Raise any necessary generation errors: for any met criterion,
        # call its `block_continued_generation_error` method if not all
        # transition-blocking criteria are met. The method might not raise an
        # error, depending on its implementation on given criterion, so the error
        # from the first met one that does block continued generation, will be raised.
        if not transition_blocking_met:
            for tc in self.transition_criteria:
                if (
                    tc.is_met(self.experiment, trials_from_node=self.trials_from_node)
                    and raise_data_required_error
                ):
                    tc.block_continued_generation_error(
                        node_name=self.node_name,
                        model_name=self.model_to_gen_from_name,
                        experiment=self.experiment,
                        trials_from_node=self.trials_from_node,
                    )

        # Determine transition state
        if len(transition_blocking) > 0 and transition_blocking_met:
            next_nodes = [
                c.transition_to
                for c in transition_blocking
                if c._transition_to is not None
            ]
            if len(set(next_nodes)) > 1:
                # TODO: support intelligent selection between multiple transition nodes
                raise NotImplementedError(
                    "Cannot currently select between multiple nodes to transition to."
                )
            else:
                return True, next_nodes[0]

        return False, None

    def generator_run_limit(self, supress_generation_errors: bool = True) -> int:
        """How many generator runs can this generation strategy generate right now,
        assuming each one of them becomes its own trial. Only considers
        `transition_criteria` that are TrialBasedCriterion.

        Returns:
            The number of generator runs that can currently be produced, with -1
            meaning unlimited generator runs.
        """
        # TODO: @mgarrard Should we consider returning `None` if there is no limit?
        # TODO:@mgarrard Should we instead have `raise_generation_error`? The name
        # of this method doesn't suggest that it would raise errors by default, since
        # it's just finding out the limit according to the name. I know we want the
        # errors in some cases, so we could call the flag `raise_error_if_cannot_gen` or
        # something like that : )
        trial_based_gen_blocking_criteria = [
            criterion
            for criterion in self.transition_criteria
            if criterion.block_gen_if_met and isinstance(criterion, TrialBasedCriterion)
        ]
        gen_blocking_criterion_delta_from_threshold = [
            criterion.num_till_threshold(
                experiment=self.experiment, trials_from_node=self.trials_from_node
            )
            for criterion in trial_based_gen_blocking_criteria
        ]

        # Raise any necessary generation errors: for any met criterion,
        # call its `block_continued_generation_error` method The method might not
        # raise an error, depending on its implementation on given criterion, so the
        # error from the first met one that does block continued generation, will be
        # raised.
        if not supress_generation_errors:
            for criterion in trial_based_gen_blocking_criteria:
                # TODO[mgarrard]: Raise a group of all the errors, from each gen-
                # blocking transition criterion.
                if criterion.is_met(
                    self.experiment, trials_from_node=self.trials_from_node
                ):
                    criterion.block_continued_generation_error(
                        node_name=self.node_name,
                        model_name=self.model_to_gen_from_name,
                        experiment=self.experiment,
                        trials_from_node=self.trials_from_node,
                    )
        if len(gen_blocking_criterion_delta_from_threshold) == 0:
            return -1
        return min(gen_blocking_criterion_delta_from_threshold)

    def __repr__(self) -> str:
        "String representation of this GenerationNode"
        # add model specs
        str_rep = f"{self.__class__.__name__}(model_specs="
        model_spec_str = str(self.model_specs).replace("\n", " ").replace("\t", "")
        str_rep += model_spec_str

        str_rep += f", node_name={self.node_name}"
        str_rep += f", transition_criteria={str(self.transition_criteria)}"

        return f"{str_rep})"


@dataclass
class GenerationStep(GenerationNode, SortableBase):
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
        use_update: DEPRECATED.
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

    Note for developers: by "model" here we really mean an Ax ModelBridge object, which
    contains an Ax Model under the hood. We call it "model" here to simplify and focus
    on explaining the logic of GenerationStep and GenerationStrategy.
    """

    # Required options:
    model: Union[ModelRegistryBase, Callable[..., ModelBridge]]
    num_trials: int

    # Optional model specifications:
    # Kwargs to pass into the Models constructor (or factory function).
    model_kwargs: Optional[Dict[str, Any]] = None
    # Kwargs to pass into the Model's `.gen` function.
    model_gen_kwargs: Optional[Dict[str, Any]] = None

    # Optional specifications for use in generation strategy:
    completion_criteria: Sequence[TransitionCriterion] = field(default_factory=list)
    min_trials_observed: int = 0
    max_parallelism: Optional[int] = None
    use_update: bool = False
    enforce_num_trials: bool = True
    # Whether the generation strategy should deduplicate the suggested arms against
    # the arms already present on the experiment. If this is `True`
    # on a given generation step, during that step the generation
    # strategy will discard a generator run that contains an arm
    # already present on the experiment and produce a new generator
    # run instead before returning it from `gen` or `_gen_multiple`.
    should_deduplicate: bool = False
    index: int = -1  # Index of this step, set internally.

    # Optional model name. Defaults to `model_spec.model_key`.
    model_name: str = field(default_factory=str)

    def __post_init__(self) -> None:
        if self.use_update:
            raise DeprecationWarning("`GenerationStep.use_update` is deprecated.")
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
        if not isinstance(self.model, ModelRegistryBase):
            if not callable(self.model):
                raise UserInputError(
                    "`model` in generation step must be either a `ModelRegistryBase` "
                    "enum subclass entry or a callable factory function returning a "
                    "model bridge instance."
                )
            model_spec = FactoryFunctionModelSpec(
                factory_function=self.model,
                model_kwargs=self.model_kwargs,
                model_gen_kwargs=self.model_gen_kwargs,
            )
        else:
            model_spec = ModelSpec(
                model_enum=self.model,
                model_kwargs=self.model_kwargs,
                model_gen_kwargs=self.model_gen_kwargs,
            )
        if self.model_name == "":
            try:
                self.model_name = model_spec.model_key
            except TypeError:
                # Factory functions may not always have a model key defined.
                self.model_name = f"Unknown {model_spec.__class__.__name__}"

        # Create transition criteria for this step. MaximumTrialsInStatus can be used
        # to ensure that requirements related to num_trials and unlimited trials
        # are met. MinimumTrialsInStatus can be used enforce the min_trials_observed
        # requirement, and override MaxTrials if enforce flag is set to true. We set
        # `transition_to` is set in `GenerationStrategy` constructor,
        # because only then is the order of the generation steps actually known.
        transition_criteria = []
        if self.num_trials != -1:
            transition_criteria.append(
                MaxTrials(
                    threshold=self.num_trials,
                    not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
                    block_gen_if_met=self.enforce_num_trials,
                    block_transition_if_unmet=True,
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
                )
            )
        if self.max_parallelism is not None:
            transition_criteria.append(
                MaxGenerationParallelism(
                    threshold=self.max_parallelism,
                    only_in_statuses=[TrialStatus.RUNNING],
                    block_gen_if_met=True,
                    block_transition_if_unmet=False,
                    transition_to=None,
                )
            )

        transition_criteria += self.completion_criteria
        super().__init__(
            node_name=f"GenerationStep_{str(self.index)}",
            model_specs=[model_spec],
            should_deduplicate=self.should_deduplicate,
            transition_criteria=transition_criteria,
        )

    @property
    def model_spec(self) -> ModelSpec:
        """Returns the first model_spec from the model_specs attribute."""
        return self.model_specs[0]

    @property
    def _unique_id(self) -> str:
        """Returns the unique ID of this generation step, which is the index."""
        return str(self.index)

    def gen(
        self,
        n: Optional[int] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        max_gen_draws_for_deduplication: int = MAX_GEN_DRAWS,
        arms_by_signature_for_deduplication: Optional[Dict[str, Arm]] = None,
        **model_gen_kwargs: Any,
    ) -> GeneratorRun:
        gr = super().gen(
            n=n,
            pending_observations=pending_observations,
            max_gen_draws_for_deduplication=max_gen_draws_for_deduplication,
            arms_by_signature_for_deduplication=arms_by_signature_for_deduplication,
            **model_gen_kwargs,
        )
        gr._generation_step_index = self.index
        return gr

    def __eq__(self, other: Base) -> bool:
        # We need to override `__eq__` to make sure we inherit the one from
        # the base class and not the one from dataclasses library, since we
        # want to be comparing equality of generation steps in the same way
        # as we compare equality of other Ax objects (and we want all the
        # same special-casing to apply).
        return SortableBase.__eq__(self, other=other)
