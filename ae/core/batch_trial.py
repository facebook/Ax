#!/usr/bin/env python3

from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    DefaultDict,
    Dict,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Union,
)

import numpy as np
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.trial import immutable_once_run
from ae.lazarus.ae.utils.common.equality import datetime_equals, equality_typechecker
from ae.lazarus.ae.utils.common.typeutils import checked_cast


if TYPE_CHECKING:
    from ae.lazarus.ae.core.experiment import (
        Experiment,
    )  # noqa F401  # pragma: no cover


class AbandonedCondition(NamedTuple):
    """Tuple storing metadata of condition that has been abandoned within
    a BatchTrial.
    """

    name: str
    time: datetime
    reason: Optional[str] = None

    @equality_typechecker
    def __eq__(self, other: "AbandonedCondition") -> bool:
        return (
            self.name == other.name
            and self.reason == other.reason
            and datetime_equals(self.time, other.time)
        )


class GeneratorRunStruct(NamedTuple):
    """Stores GeneratorRun object as well as the weight with which it was added."""

    generator_run: GeneratorRun
    weight: float


class BatchTrial(BaseTrial):
    def __init__(self, experiment: "Experiment") -> None:
        super().__init__(experiment=experiment)
        self._generator_run_structs: List[GeneratorRunStruct] = []
        self._abandoned_conditions_metadata: Dict[str, AbandonedCondition] = {}
        self._status_quo: Optional[Condition] = None
        self._status_quo_weight: float = 0.0
        self.set_status_quo(experiment.status_quo)

    @property
    def experiment(self) -> "Experiment":
        """The experiment this batch belongs to."""
        return self._experiment

    @property
    def index(self) -> int:
        """The index of this batch within the experiment's batch list."""
        return self._index

    @property
    def generator_run_structs(self) -> List[GeneratorRunStruct]:
        """List of generator run structs attached to this trial.

        Struct holds generator_run object and the weight with which it was added.
        """
        return self._generator_run_structs

    @property
    def condition_weights(self) -> Optional[MutableMapping[Condition, float]]:
        """The set of conditions and associated weights for the trial.

        These are constructed by merging the conditions and weights from
        each generator run that is attached to the trial.
        """
        if len(self._generator_run_structs) == 0 and self.status_quo is None:
            return None
        condition_weights = OrderedDict()
        for struct in self._generator_run_structs:
            multiplier = struct.weight
            for condition, weight in struct.generator_run.condition_weights.items():
                scaled_weight = weight * multiplier
                if condition in condition_weights:
                    condition_weights[condition] += scaled_weight
                else:
                    condition_weights[condition] = scaled_weight
        if self.status_quo is not None:
            condition_weights[
                self.status_quo
            ] = self._status_quo_weight + condition_weights.get(self.status_quo, 0.0)
        return condition_weights

    @condition_weights.setter
    def condition_weights(
        self, condition_weights: MutableMapping[Condition, float]
    ) -> None:
        raise NotImplementedError("Use `trial.add_conditions_and_weights`")

    @immutable_once_run
    def add_condition(self, condition: Condition, weight: float = 1.0) -> "BatchTrial":
        """Add a condition to the trial.

        Args:
            condition: The condition to be added.
            weight: The weight with which this condition should be added.

        Returns:
            The trial instance.
        """
        return self.add_conditions_and_weights(conditions=[condition], weights=[weight])

    @immutable_once_run
    def add_conditions_and_weights(
        self,
        conditions: List[Condition],
        weights: Optional[List[float]] = None,
        multiplier: float = 1.0,
    ) -> "BatchTrial":
        """Add conditions and weights to the trial.

        Args:
            conditions: The conditions to be added.
            weights: The weights associated with the conditions.
            multiplier: The multiplier applied to input weights before merging with
                the current set of conditions and weights.

        Returns:
            The trial instance.
        """

        # TODO: Somehow denote on the generator run that it was manually created.
        # Currently no generator info is stored on generator run
        return self.add_generator_run(
            generator_run=GeneratorRun(conditions=conditions, weights=weights),
            multiplier=multiplier,
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> "BatchTrial":
        """Add a generator run to the trial.

        The conditions and weights from the generator run will be merged with
        the existing conditions and weights on the trial, and the generator run
        object will be linked to the trial for tracking.

        Args:
            generator_run: The generator run to be added.
            multiplier: The multiplier applied to input weights before merging with
                the current set of conditions and weights.

        Returns:
            The trial instance.
        """

        # Add names to conditions
        # For those not yet added to this experiment, create a new name
        # Else, use the name of the existing condition
        for condition in generator_run.conditions:
            self._check_existing_and_name_condition(condition)

        # TODO validate that conditions belong to search space
        self._generator_run_structs.append(
            GeneratorRunStruct(generator_run=generator_run, weight=multiplier)
        )
        generator_run.index = len(self._generator_run_structs) - 1

        # Resize status_quo based on new conditions
        self.set_status_quo(self.status_quo)
        return self

    @property
    def status_quo(self) -> Optional[Condition]:
        """The control condition for this batch."""
        return self._status_quo

    @immutable_once_run
    def set_status_quo(
        self, status_quo: Optional[Condition], weight: Optional[float] = None
    ) -> "BatchTrial":
        """Sets status quo condition.

        Defaults weight to average of existing weights or 1.0 if no weights exist.
        """
        # Assign a name to this condition if none exists
        if weight is not None and weight <= 0.0:
            raise ValueError("Status quo weight must be positive.")

        if status_quo is not None:
            # If status_quo is identical to an existing condition, set to that
            # so that the names match.
            if status_quo.signature in self.experiment.conditions_by_signature:
                new_status_quo = self.experiment.conditions_by_signature[
                    status_quo.signature
                ]
                if status_quo.name != new_status_quo.name:
                    raise ValueError(
                        f"Condition already exists with name {new_status_quo.name}."
                    )
                status_quo = new_status_quo
            elif not status_quo.has_name:
                status_quo.name = "status_quo_" + str(self.index)

        # Unset status_quo so avg weight computation works as intended
        self._status_quo = None
        if weight is None:
            weight = (
                1.0
                if len(self.weights) == 0
                else float(sum(self.weights)) / len(self.weights)
            )

        self._status_quo = status_quo
        self._status_quo_weight = weight
        return self

    @property
    def conditions(self) -> List[Condition]:
        """All conditions contained in the trial."""
        condition_weights = self.condition_weights
        return [] if condition_weights is None else list(condition_weights.keys())

    @property
    def weights(self) -> List[float]:
        """Weights corresponding to conditions contained in the trial."""
        condition_weights = self.condition_weights
        return [] if condition_weights is None else list(condition_weights.values())

    @property
    def conditions_by_name(self) -> Dict[str, Condition]:
        """Map from condition name to object for all conditions in trial."""
        conditions_by_name = {}
        for condition in self.conditions:
            if not condition.has_name:
                raise ValueError(  # pragma: no cover
                    "Conditions attached to a trial must have a name."
                )
            conditions_by_name[condition.name] = condition
        return conditions_by_name

    @property
    def abandoned_conditions(self) -> List[Condition]:
        """List of conditions that have been abandoned within this trial"""
        return [
            self.conditions_by_name[condition.name]
            for condition in self._abandoned_conditions_metadata.values()
        ]

    @property
    def abandoned_conditions_metadata(self) -> List[AbandonedCondition]:
        return list(self._abandoned_conditions_metadata.values())

    @property
    def is_factorial(self) -> bool:
        """Return true if the trial's conditions are a factorial design with
        no linked factors.
        """
        # To match the model behavior, this should probably actually be pulled
        # from exp.params. However, that seems rather ugly when this function
        # intuitively should just depend on the conditions.
        sufficient_factors = all(
            len(condition.params or []) >= 2 for condition in self.conditions
        )
        if not sufficient_factors:
            return False
        # pyre: param_levels is declared to have type `DefaultDict[str,
        # pyre: Dict[Union[float, str], int]]` but is used as type
        # pyre-fixme[9]: `defaultdict`.
        param_levels: DefaultDict[str, Dict[Union[str, float], int]] = (
            defaultdict(dict)
        )
        for condition in self.conditions:
            for param_name, param_value in condition.params.items():
                # Expected `Union[float, str]` for 2nd anonymous parameter to call
                # `dict.__setitem__` but got `Optional[Union[bool, float, str]]`.
                # pyre-fixme[6]:
                param_levels[param_name][param_value] = 1
        param_cardinality = 1
        for param_values in param_levels.values():
            param_cardinality *= len(param_values)
        return len(self.conditions) == param_cardinality

    def run(self) -> "BatchTrial":
        return checked_cast(BatchTrial, super().run())

    def normalized_condition_weights(
        self, total: float = 1, trunc_digits: Optional[int] = None
    ) -> MutableMapping[Condition, float]:
        """Returns conditions with a new set of weights normalized
        to the given total.

        This method is useful for many runners where we need to normalize weights
        to a certain total without mutating the weights attached to a trial.

        Args:
            total: The total weight to which to normalize.
                Default is 1, in which case condition weights
                can be interpreted as probabilities.
            trunc_digits: The number of digits to keep. If the
                resulting total weight is not equal to `total`, re-allocate
                weight in such a way to maintain relative weights as best as
                possible.

        Returns:
            Mapping from conditions to the new set of weights.

        """
        weights = np.array(self.weights)
        if trunc_digits is not None:
            atomic_weight = 10 ** -trunc_digits
            int_weights = (
                (total / atomic_weight) * (weights / np.sum(weights))
            ).astype(int)
            n_leftover = int(total / atomic_weight) - np.sum(int_weights)
            int_weights[:n_leftover] += 1
            weights = int_weights * atomic_weight
        else:
            weights = weights * (total / np.sum(weights))
        return OrderedDict(zip(self.conditions, weights))

    def mark_condition_abandoned(
        self, condition: Condition, reason: Optional[str] = None
    ) -> "BatchTrial":
        """Mark a condition abandoned.

        Usually done after deployment when one condition causes issues but
        user wants to continue running other conditions in the batch.

        Args:
            condition: The condition object to abandon.
            reason: The reason for abandoning the condition.

        Returns:
            The batch instance.
        """
        if condition not in self.conditions:
            raise ValueError("Condition must be contained in batch.")
        if not condition.has_name:  # pragma: nocover
            raise ValueError("Condition must have a name.")

        abandoned_condition = AbandonedCondition(
            name=condition.name, time=datetime.now(), reason=reason
        )
        self._abandoned_conditions_metadata[condition.name] = abandoned_condition
        return self

    def clone(self) -> "BatchTrial":
        """Clone the trial.

        Returns:
            A new instance of the trial.
        """
        new_trial = self._experiment.new_batch_trial()
        for struct in self._generator_run_structs:
            new_trial.add_generator_run(struct.generator_run, struct.weight)
        new_trial.trial_type = self._trial_type
        new_trial.runner = self._runner
        return new_trial

    def __repr__(self) -> str:
        return (
            "BatchTrial("
            f"experiment_name='{self._experiment.name}', "
            f"index={self._index}, "
            f"status={self._status})"
        )
