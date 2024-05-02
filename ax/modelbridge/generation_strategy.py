#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy
from functools import wraps
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.utils import (
    extend_pending_observations,
    get_pending_observation_features_based_on_trial_status,
)
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.exceptions.generation_strategy import (
    GenerationStrategyCompleted,
    GenerationStrategyMisconfiguredException,
)
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_node import GenerationNode, GenerationStep
from ax.modelbridge.model_spec import FactoryFunctionModelSpec
from ax.modelbridge.modelbridge_utils import get_fixed_features_from_experiment
from ax.modelbridge.registry import _extract_model_state_after_gen, ModelRegistryBase
from ax.modelbridge.transition_criterion import TrialBasedCriterion
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.typeutils import checked_cast, checked_cast_list, not_none
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)


MAX_CONDITIONS_GENERATED = 10000
MAX_GEN_DRAWS = 5
MAX_GEN_DRAWS_EXCEEDED_MESSAGE = (
    f"GenerationStrategy exceeded `MAX_GEN_DRAWS` of {MAX_GEN_DRAWS} while trying to "
    "generate a unique parameterization. This indicates that the search space has "
    "likely been fully explored, or that the sweep has converged."
)
T = TypeVar("T")


def step_based_gs_only(f: Callable[..., T]) -> Callable[..., T]:
    """
    For use as a decorator on functions only implemented for ``GenerationStep``-based
    ``GenerationStrategies``. Mainly useful for older ``GenerationStrategies``.
    """

    @wraps(f)
    def impl(
        self: "GenerationStrategy", *args: List[Any], **kwargs: Dict[str, Any]
    ) -> T:
        if self.is_node_based:
            raise UnsupportedError(
                f"{f.__name__} is not supported for GenerationNode based"
                " GenerationStrategies."
            )
        return f(self, *args, **kwargs)

    return impl


class GenerationStrategy(GenerationStrategyInterface):
    """GenerationStrategy describes which model should be used to generate new
    points for which trials, enabling and automating use of different models
    throughout the optimization process. For instance, it allows to use one
    model for the initialization trials, and another one for all subsequent
    trials. In the general case, this allows to automate use of an arbitrary
    number of models to generate an arbitrary numbers of trials
    described in the `trials_per_model` argument.

    Args:
        nodes: A list of `GenerationNode`. Each `GenerationNode` in the list
            represents a single node in a `GenerationStrategy` which, when
            composed of `GenerationNodes`, can be conceptualized as a graph instead
            of a linear list. `TransitionCriterion` defined in each `GenerationNode`
            represent the edges in the `GenerationStrategy` graph. `GenerationNodes`
            are more flexible than `GenerationSteps` and new `GenerationStrategies`
            should use nodes. Notably, either, but not both, of `nodes` and `steps`
            must be provided.
        steps: A list of `GenerationStep` describing steps of this strategy.
        name: An optional name for this generaiton strategy. If not specified,
            strategy's name will be names of its nodes' models joined with '+'.
    """

    _nodes: List[GenerationNode]
    _curr: GenerationNode  # Current node in the strategy.
    # Whether all models in this GS are in Models registry enum.
    _uses_registered_models: bool
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment] = None
    # Trial indices as last seen by the model; updated in `_model` property setter.
    _model: Optional[ModelBridge] = None  # Current model.

    def __init__(
        self,
        steps: Optional[List[GenerationStep]] = None,
        name: Optional[str] = None,
        nodes: Optional[List[GenerationNode]] = None,
    ) -> None:
        # Validate that one and only one of steps or nodes is provided
        if not ((steps is None) ^ (nodes is None)):
            raise GenerationStrategyMisconfiguredException(
                error_info="GenerationStrategy must contain either steps or nodes."
            )

        # pyre-ignore[8]
        self._nodes = none_throws(nodes if steps is None else steps)

        # Validate correctness of steps list or nodes graph
        if isinstance(steps, list) and all(
            isinstance(s, GenerationStep) for s in steps
        ):
            self._validate_and_set_step_sequence(steps=self._nodes)
        elif isinstance(nodes, list) and self.is_node_based:
            self._validate_and_set_node_graph(nodes=nodes)
        else:
            # TODO[mgarrard]: Allow mix of nodes and steps
            raise GenerationStrategyMisconfiguredException(
                "`GenerationStrategy` inputs are:\n"
                "`steps` (list of `GenerationStep`) or\n"
                "`nodes` (list of `GenerationNode`)."
            )

        # Log warning if the GS uses a non-registered (factory function) model.
        self._uses_registered_models = not any(
            isinstance(ms, FactoryFunctionModelSpec)
            for node in self._nodes
            for ms in node.model_specs
        )
        if not self._uses_registered_models:
            logger.info(
                "Using model via callable function, "
                "so optimization is not resumable if interrupted."
            )
        self._generator_runs = []
        # Set name to an explicit value ahead of time to avoid
        # adding properties during equality checks
        super().__init__(name=name or self._make_default_name())

    @property
    def is_node_based(self) -> bool:
        """Whether this strategy consists of GenerationNodes only.
        This is useful for determining initialization properties and
        other logic.
        """
        return not any(isinstance(n, GenerationStep) for n in self._nodes) and all(
            isinstance(n, GenerationNode) for n in self._nodes
        )

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps, set at the time of the
        ``GenerationStrategy`` creation.
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set generation strategy name."""
        self._name = name

    @property
    @step_based_gs_only
    def model_transitions(self) -> List[int]:
        """[DEPRECATED]List of trial indices where a transition happened from one model
        to another.
        """
        raise DeprecationWarning(
            "`model_transitions` is no longer supported. Please refer to `model_key` "
            "field on generator runs for similar information if needed."
        )

    @property
    def current_step(self) -> GenerationStep:
        """Current generation step."""
        if not isinstance(self._curr, GenerationStep):
            raise TypeError(
                "The current object is not a GenerationStep, you may be looking "
                "for the current_node property."
            )
        return self._curr

    @property
    def current_node(self) -> GenerationNode:
        """Current generation node."""
        if not isinstance(self._curr, GenerationNode):
            raise TypeError(
                "The current object is not a GenerationNode, you may be looking for the"
                " current_step property."
            )
        return self._curr

    @property
    def current_node_name(self) -> str:
        """Current generation node name."""
        return self._curr.node_name

    @property
    @step_based_gs_only
    def current_step_index(self) -> int:
        """Returns the index of the current generation step. This attribute
        is replaced by node_name in newer GenerationStrategies but surfaced here
        for backward compatibility.
        """
        node_names_for_all_steps = [step._node_name for step in self._nodes]
        assert (
            self._curr.node_name in node_names_for_all_steps
        ), "The current step is not found in the list of steps"

        return node_names_for_all_steps.index(self._curr.node_name)

    @property
    def model(self) -> Optional[ModelBridge]:
        """Current model in this strategy. Returns None if no model has been set
        yet (i.e., if no generator runs have been produced from this GS).
        """
        return self._curr._fitted_model

    @property
    def experiment(self) -> Experiment:
        """Experiment, currently set on this generation strategy."""
        if self._experiment is None:
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
        else:
            raise ValueError(
                "This generation strategy has been used for experiment "
                f"{self.experiment._name} so far; cannot reset experiment"
                f" to {experiment._name}. If this is a new optimization, "
                "a new generation strategy should be created instead."
            )

    @property
    def last_generator_run(self) -> Optional[GeneratorRun]:
        """Latest generator run produced by this generation strategy.
        Returns None if no generator runs have been produced yet.
        """
        # Used to restore current model when decoding a serialized GS.
        return self._generator_runs[-1] if self._generator_runs else None

    @property
    def uses_non_registered_models(self) -> bool:
        """Whether this generation strategy involves models that are not
        registered and therefore cannot be stored."""
        return not self._uses_registered_models

    @property
    def trials_as_df(self) -> Optional[pd.DataFrame]:
        """Puts information on individual trials into a data frame for easy
        viewing.

        For example for a GenerationStrategy composed of GenerationSteps:
        Gen. Step | Model | Trial Index | Trial Status | Arm Parameterizations
        0         | Sobol | 0           | RUNNING      | {"0_0":{"x":9.17...}}
        """
        logger.info(
            "Note that parameter values in dataframe are rounded to 2 decimal "
            "points; the values in the dataframe are thus not the exact ones "
            "suggested by Ax in trials."
        )

        if self._experiment is None or all(
            len(step.trials_from_node) == 0 for step in self._nodes
        ):
            return None

        step_or_node_col = (
            "Generation Node" if self.is_node_based else "Generation Step"
        )
        records = [
            {
                step_or_node_col: node.node_name,
                "Generation Model": self._nodes[
                    node_idx
                ].model_spec_to_gen_from.model_key,
                "Trial Index": trial_idx,
                "Trial Status": self.experiment.trials[trial_idx].status.name,
                "Arm Parameterizations": {
                    arm.name: _round_floats_for_logging(arm.parameters)
                    for arm in self.experiment.trials[trial_idx].arms
                },
            }
            for node_idx, node in enumerate(self._nodes)
            for trial_idx in node.trials_from_node
        ]

        return pd.DataFrame.from_records(records).reindex(
            columns=[
                step_or_node_col,
                "Generation Model",
                "Trial Index",
                "Trial Status",
                "Arm Parameterizations",
            ]
        )

    @property
    def optimization_complete(self) -> bool:
        """Checks whether all nodes are completed in the generation strategy."""
        return all(node.is_completed for node in self._nodes)

    @property
    @step_based_gs_only
    def _steps(self) -> List[GenerationStep]:
        """List of generation steps."""
        return self._nodes  # pyre-ignore[7]

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
        with the `model_gen_kwargs` set on the current generation node.

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
                generator run. By default, data is all data on the `experiment`.
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

    def gen_for_multiple_trials_with_multiple_models(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Optional[Data] = None,
        n: int = 1,
    ) -> List[List[GeneratorRun]]:
        """Produce GeneratorRuns for multiple trials at once with the possibility of
        ensembling, or using multiple models per trial, getting multiple
        GeneratorRuns per trial.

        NOTE: This method is in development.  Please do not use it yet.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment`.
            n: Integer representing how many trials should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.

        Returns:
            A list of lists of lists generator runs. Each outer list represents
            a trial being suggested and  each inner list represents a generator
            run for that trial.
        """
        grs = self._gen_multiple(
            experiment=experiment,
            num_generator_runs=num_generator_runs,
            data=data,
            n=n,
            pending_observations=get_pending_observation_features_based_on_trial_status(
                experiment=experiment
            ),
            fixed_features=get_fixed_features_from_experiment(experiment=experiment),
        )
        return [[gr] for gr in grs]

    def current_generator_run_limit(
        self,
    ) -> Tuple[int, bool]:
        """First check if we can move the generation strategy to the next node, which
        is safe, as the next call to ``gen`` will just pick up from there. Then
        determine how many generator runs this generation strategy can generate right
        now, assuming each one of them becomes its own trial, and whether optimization
        is completed.

        Returns: a two-item tuple of:
              - the number of generator runs that can currently be produced, with -1
                meaning unlimited generator runs,
              - whether optimization is completed and the generation strategy cannot
                generate any more generator runs at all.
        """
        try:
            self._maybe_move_to_next_step(raise_data_required_error=False)
        except GenerationStrategyCompleted:
            return 0, True

        # if the generation strategy is not complete, optimization is not complete
        return self._curr.generator_run_limit(), False

    def clone_reset(self) -> GenerationStrategy:
        """Copy this generation strategy without it's state."""
        cloned_nodes = deepcopy(self._nodes)
        for n in cloned_nodes:
            # Unset the generation strategy back-pointer, so the nodes are not
            # associated with any generation strategy.
            n._generation_strategy = None
        if self.is_node_based:
            return GenerationStrategy(name=self.name, nodes=cloned_nodes)

        return GenerationStrategy(
            name=self.name, steps=checked_cast_list(GenerationStep, cloned_nodes)
        )

    def _unset_non_persistent_state_fields(self) -> None:
        """Utility for testing convenience: unset fields of generation strategy
        that are set during candidate generation; these fields are not persisted
        during storage. To compare a pre-storage and a reloaded generation
        strategies; call this utility on the pre-storage one first. The rest
        of the fields should be identical.
        """
        self._model = None
        for s in self._nodes:
            s._model_spec_to_gen_from = None

    @step_based_gs_only
    def _validate_and_set_step_sequence(self, steps: List[GenerationStep]) -> None:
        """Initialize and validate the steps provided to this GenerationStrategy.

        Some GenerationStrategies are composed of GenerationStep objects, but we also
        need to initialize the correct GenerationNode representation for these steps.
        This function validates:
            1. That only the last step has num_trials=-1, which indicates unlimited
               trial generation is possible.
            2. That each step's num_trials attrivute is either positive or -1
            3. That each step's max_parallelism attribute is either None or positive
        It then sets the corect TransitionCriterion and node_name attributes on the
        underlying GenerationNode objects.
        """
        for idx, step in enumerate(steps):
            if step.num_trials == -1 and len(step.completion_criteria) < 1:
                if idx < len(self._steps) - 1:
                    raise UserInputError(
                        "Only last step in generation strategy can have "
                        "`num_trials` set to -1 to indicate that the model in "
                        "the step shouldbe used to generate new trials "
                        "indefinitely unless completion critera present."
                    )
            elif step.num_trials < 1 and step.num_trials != -1:
                raise UserInputError(
                    "`num_trials` must be positive or -1 (indicating unlimited) "
                    "for all generation steps."
                )
            if step.max_parallelism is not None and step.max_parallelism < 1:
                raise UserInputError(
                    "Maximum parallelism should be None (if no limit) or "
                    f"a positive number. Got: {step.max_parallelism} for "
                    f"step {step.model_name}."
                )

            step._node_name = f"GenerationStep_{str(idx)}"
            step.index = idx

            # Set transition_to field for all but the last step, which remains
            # null.
            if idx != len(self._steps):
                for transition_criteria in step.transition_criteria:
                    if (
                        transition_criteria.criterion_class
                        != "MaxGenerationParallelism"
                    ):
                        transition_criteria._transition_to = (
                            f"GenerationStep_{str(idx + 1)}"
                        )
            step._generation_strategy = self
        self._curr = steps[0]

    def _validate_and_set_node_graph(self, nodes: List[GenerationNode]) -> None:
        """Initialize and validate the node graph provided to this GenerationStrategy.

        This function validates:
            1. That all nodes have unique names.
            2. That there is at least one node with a transition_to field.
            3. That all `transition_to` attributes on a TransitionCriterion point to
                another node in the same GenerationStrategy.
            4. Warns if no nodes contain a transition criterion
        """
        node_names = []
        for node in self._nodes:
            # validate that all node names are unique
            if node.node_name in node_names:
                raise GenerationStrategyMisconfiguredException(
                    error_info="All node names in a GenerationStrategy "
                    + "must be unique."
                )

            node_names.append(node.node_name)
            node._generation_strategy = self

        # validate `transition_criterion`
        contains_a_transition_to_argument = False
        for node in self._nodes:
            for transition_criteria in node.transition_criteria:
                if transition_criteria.transition_to is not None:
                    contains_a_transition_to_argument = True
                    if transition_criteria.transition_to not in node_names:
                        raise GenerationStrategyMisconfiguredException(
                            error_info=f"`transition_to` argument "
                            f"{transition_criteria.transition_to} does not "
                            "correspond to any node in this GenerationStrategy."
                        )

        # validate that at least one node has transition_to field
        if len(self._nodes) > 1 and not contains_a_transition_to_argument:
            logger.warning(
                "None of the nodes in this GenerationStrategy "
                "contain a `transition_to` argument in their transition_criteria. "
                "Therefore, the GenerationStrategy will not be able to "
                "move from one node to another. Please add a "
                "`transition_to` argument."
            )
        self._curr = nodes[0]

    @step_based_gs_only
    def _step_repr(self, step_str_rep: str) -> str:
        """Return the string representation of the steps in a GenerationStrategy
        composed of GenerationSteps.
        """
        step_str_rep += "steps=["
        remaining_trials = "subsequent" if len(self._nodes) > 1 else "all"
        for step in self._nodes:
            num_trials = remaining_trials
            for criterion in step.transition_criteria:
                if criterion.criterion_class == "MaxTrials" and isinstance(
                    criterion, TrialBasedCriterion
                ):
                    num_trials = criterion.threshold

            try:
                model_name = step.model_spec_to_gen_from.model_key
            except TypeError:
                model_name = "model with unknown name"

            step_str_rep += f"{model_name} for {num_trials} trials, "
        step_str_rep = step_str_rep[:-2]
        step_str_rep += "])"
        return step_str_rep

    def _make_default_name(self) -> str:
        """Make a default name for this generation strategy; used when no name is passed
        to the constructor. For node-based generation strategies, the name is
        constructed by joining together the names of the nodes set on this
        generation strategy. For step-based generation strategies, the model keys
        of the underlying model specs are used.
        Note: This should only be called once the nodes are set.
        """
        if not self._nodes:
            raise UnsupportedError(
                "Cannot make a default name for a generation strategy with no nodes "
                "set yet."
            )
        # TODO: Simplify this after updating GStep names to represent underlying models.
        if self.is_node_based:
            node_names = (node.node_name for node in self._nodes)
        else:
            node_names = (node.model_spec_to_gen_from.model_key for node in self._nodes)
            # Trim the "get_" beginning of the factory function if it's there.
            node_names = (n[4:] if n[:4] == "get_" else n for n in node_names)
        return "+".join(node_names)

    def __repr__(self) -> str:
        """String representation of this generation strategy."""
        gs_str = f"GenerationStrategy(name='{self.name}', "
        if not self.is_node_based:
            return self._step_repr(gs_str)
        gs_str += f"nodes={str(self._nodes)})"
        return gs_str

    # ------------------------- Candidate generation helpers. -------------------------

    def _gen_multiple(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Optional[Data] = None,
        n: int = 1,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        **model_gen_kwargs: Any,
    ) -> List[GeneratorRun]:
        """Produce multiple generator runs at once, to be made into multiple
        trials on the experiment.

        NOTE: This is used to ensure that maximum paralellism and number
        of trials per node are not violated when producing many generator
        runs from this generation strategy in a row. Without this function,
        if one generates multiple generator runs without first making any
        of them into running trials, generation strategy cannot enforce that it only
        produces as many generator runs as are allowed by the paralellism
        limit and the limit on number of trials in current node.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment`.
            n: Integer representing how many arms should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            model_gen_kwargs: Keyword arguments that are passed through to
                ``GenerationNode.gen``, which will pass them through to
                ``ModelSpec.gen``, which will pass them to ``ModelBridge.gen``.
        """
        self.experiment = experiment
        self._maybe_move_to_next_step()
        self._fit_current_model(data=data)

        # Get GeneratorRun limit that respects the node's transition criterion that
        # affect the number of generator runs that can be produced.
        gr_limit = self._curr.generator_run_limit(supress_generation_errors=False)
        if gr_limit == -1:
            num_generator_runs = max(num_generator_runs, 1)
        else:
            num_generator_runs = max(min(num_generator_runs, gr_limit), 1)
        generator_runs = []
        pending_observations = deepcopy(pending_observations) or {}
        for _ in range(num_generator_runs):
            try:
                generator_run = self._curr.gen(
                    n=n,
                    pending_observations=pending_observations,
                    arms_by_signature_for_deduplication=experiment.arms_by_signature,
                    **model_gen_kwargs,
                )

            except DataRequiredError as err:
                # Model needs more data, so we log the error and return
                # as many generator runs as we were able to produce, unless
                # no trials were produced at all (in which case its safe to raise).
                if len(generator_runs) == 0:
                    raise
                logger.debug(f"Model required more data: {err}.")
                break

            self._generator_runs.append(generator_run)
            generator_runs.append(generator_run)

            # Extend the `pending_observation` with newly generated point(s)
            # to avoid repeating them.
            extend_pending_observations(
                experiment=experiment,
                pending_observations=pending_observations,
                generator_run=generator_run,
            )
        return generator_runs

    # ------------------------- Model selection logic helpers. -------------------------

    def _fit_current_model(self, data: Optional[Data]) -> None:
        """Fits or update the model on the current generation node (does not move
        between generation nodes).

        Args:
            data: Optional ``Data`` to fit or update with; if not specified, generation
                strategy will obtain the data via ``experiment.lookup_data``.
        """
        data = self.experiment.lookup_data() if data is None else data
        # If last generator run's index matches the current node, extract
        # model state from last generator run and pass it to the model
        # being instantiated in this function.
        model_state_on_lgr = self._get_model_state_from_last_generator_run()
        if not data.df.empty:
            trial_indices_in_data = sorted(data.df["trial_index"].unique())
            logger.debug(f"Fitting model with data for trials: {trial_indices_in_data}")

        self._curr.fit(experiment=self.experiment, data=data, **model_state_on_lgr)
        self._model = self._curr._fitted_model

    def _maybe_move_to_next_step(self, raise_data_required_error: bool = True) -> bool:
        """Moves this generation strategy to next node if the current node is completed,
        and it is not the last node in this generation strategy. This method is safe to
        use both when generating candidates or simply checking how many generator runs
        (to be made into trials) can currently be produced.

        NOTE: this method raises ``GenerationStrategyCompleted`` error if the current
        generation node is complete, but it is also the last in generation strategy.

        Args:
            raise_data_required_error: Whether to raise ``DataRequiredError`` in the
                maybe_step_completed method in GenerationNode class.

        Returns:
            Whether generation strategy moved to the next node.
        """
        move_to_next_node, next_node = self._curr.should_transition_to_next_node(
            raise_data_required_error=raise_data_required_error
        )
        if move_to_next_node:
            if self.optimization_complete:
                raise GenerationStrategyCompleted(
                    f"Generation strategy {self} generated all the trials as "
                    "specified in its nodes."
                )
            if next_node is None:
                # If the last node did not specify which node to transition to,
                # move to the next node in the list.
                current_node_index = self._nodes.index(self._curr)
                next_node = self._nodes[current_node_index + 1].node_name
            for node in self._nodes:
                if node.node_name == next_node:
                    self._curr = node
                    # Moving to the next node also entails unsetting this GS's model
                    # (since new node's model will be initialized for the first time;
                    # this is done in `self._fit_current_model).
                    self._model = None
        return move_to_next_node

    def _get_model_state_from_last_generator_run(self) -> Dict[str, Any]:
        lgr = self.last_generator_run
        # NOTE: This will not be easily compatible with `GenerationNode`;
        # will likely need to find last generator run per model. Not a problem
        # for now though as GS only allows `GenerationStep`-s for now.
        # Potential solution: store generator runs on `GenerationNode`-s and
        # split them per-model there.
        model_state_on_lgr = {}
        # Need to check if model_spec_to_gen_from is none to account for
        # ExternalGenerationNodes which leverage models from outside Ax.
        model_on_curr = (
            self._curr.model_spec_to_gen_from.model_enum
            if self._curr.model_spec_to_gen_from
            else None
        )
        if lgr is None:
            return model_state_on_lgr

        if not self.is_node_based:
            grs_equal = lgr._generation_step_index == self.current_step_index
        else:
            grs_equal = lgr._generation_node_name == self._curr.node_name

        if grs_equal and lgr._model_state_after_gen:
            if self.model or isinstance(model_on_curr, ModelRegistryBase):
                # TODO[drfreund]: Consider moving this to `GenerationStep` or
                # `GenerationNode`.
                model_cls = (
                    self.model.model.__class__
                    if self.model is not None
                    # NOTE: This checked cast is save per the OR-statement in last line
                    # of the IF-check above.
                    else checked_cast(ModelRegistryBase, model_on_curr).model_class
                )
                model_state_on_lgr = _extract_model_state_after_gen(
                    generator_run=lgr,
                    model_class=model_cls,
                )

        return model_state_on_lgr
