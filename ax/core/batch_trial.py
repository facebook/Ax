#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.trial import immutable_once_run
from ax.utils.common.equality import datetime_equals, equality_typechecker
from ax.utils.common.typeutils import checked_cast


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401  # pragma: no cover


class AbandonedArm(NamedTuple):
    """Tuple storing metadata of arm that has been abandoned within
    a BatchTrial.
    """

    name: str
    time: datetime
    reason: Optional[str] = None

    @equality_typechecker
    def __eq__(self, other: "AbandonedArm") -> bool:
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
    def __init__(
        self,
        experiment: "core.experiment.Experiment",
        generator_run: Optional[GeneratorRun] = None,
        trial_type: Optional[str] = None,
    ) -> None:
        super().__init__(experiment=experiment, trial_type=trial_type)
        self._generator_run_structs: List[GeneratorRunStruct] = []
        self._abandoned_arms_metadata: Dict[str, AbandonedArm] = {}
        self._status_quo: Optional[Arm] = None
        self._status_quo_weight: float = 0.0
        if generator_run is not None:
            self.add_generator_run(generator_run=generator_run)
        self.status_quo = experiment.status_quo

    @property
    def experiment(self) -> "core.experiment.Experiment":
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
    def arm_weights(self) -> Optional[MutableMapping[Arm, float]]:
        """The set of arms and associated weights for the trial.

        These are constructed by merging the arms and weights from
        each generator run that is attached to the trial.
        """
        if len(self._generator_run_structs) == 0 and self.status_quo is None:
            return None
        arm_weights = OrderedDict()
        for struct in self._generator_run_structs:
            multiplier = struct.weight
            for arm, weight in struct.generator_run.arm_weights.items():
                scaled_weight = weight * multiplier
                if arm in arm_weights:
                    arm_weights[arm] += scaled_weight
                else:
                    arm_weights[arm] = scaled_weight
        if self.status_quo is not None:
            arm_weights[self.status_quo] = self._status_quo_weight + arm_weights.get(
                self.status_quo, 0.0
            )
        return arm_weights

    @arm_weights.setter
    def arm_weights(self, arm_weights: MutableMapping[Arm, float]) -> None:
        raise NotImplementedError("Use `trial.add_arms_and_weights`")

    @immutable_once_run
    def add_arm(self, arm: Arm, weight: float = 1.0) -> "BatchTrial":
        """Add a arm to the trial.

        Args:
            arm: The arm to be added.
            weight: The weight with which this arm should be added.

        Returns:
            The trial instance.
        """
        return self.add_arms_and_weights(arms=[arm], weights=[weight])

    @immutable_once_run
    def add_arms_and_weights(
        self,
        arms: List[Arm],
        weights: Optional[List[float]] = None,
        multiplier: float = 1.0,
    ) -> "BatchTrial":
        """Add arms and weights to the trial.

        Args:
            arms: The arms to be added.
            weights: The weights associated with the arms.
            multiplier: The multiplier applied to input weights before merging with
                the current set of arms and weights.

        Returns:
            The trial instance.
        """

        return self.add_generator_run(
            generator_run=GeneratorRun(
                arms=arms, weights=weights, type=GeneratorRunType.MANUAL.name
            ),
            multiplier=multiplier,
        )

    @immutable_once_run
    def add_generator_run(
        self, generator_run: GeneratorRun, multiplier: float = 1.0
    ) -> "BatchTrial":
        """Add a generator run to the trial.

        The arms and weights from the generator run will be merged with
        the existing arms and weights on the trial, and the generator run
        object will be linked to the trial for tracking.

        Args:
            generator_run: The generator run to be added.
            multiplier: The multiplier applied to input weights before merging with
                the current set of arms and weights.

        Returns:
            The trial instance.
        """
        # Copy the generator run, to preserve initial and skip mutations to arms.
        generator_run = generator_run.clone()

        # First validate generator run arms
        for arm in generator_run.arms:
            self.experiment.search_space.check_types(arm.parameters, raise_error=True)

        # Add names to arms
        # For those not yet added to this experiment, create a new name
        # Else, use the name of the existing arm
        for arm in generator_run.arms:
            self._check_existing_and_name_arm(arm)

        self._generator_run_structs.append(
            GeneratorRunStruct(generator_run=generator_run, weight=multiplier)
        )
        generator_run.index = len(self._generator_run_structs) - 1

        # Resetting status quo reweights the status_quo, based on new arms
        self.reweight_status_quo()
        return self

    @property
    def status_quo(self) -> Optional[Arm]:
        """The control arm for this batch."""
        return self._status_quo

    @status_quo.setter
    def status_quo(self, status_quo: Optional[Arm]) -> None:
        """Sets status quo arm."""
        self.set_status_quo_with_weight(status_quo)

    @immutable_once_run
    def set_status_quo_with_weight(
        self, status_quo: Optional[Arm], weight: Optional[float] = None
    ) -> "BatchTrial":
        """Sets status quo arm.

        Defaults weight to average of existing weights or 1.0 if no weights exist.
        """
        # Assign a name to this arm if none exists
        if weight is not None and weight <= 0.0:
            raise ValueError("Status quo weight must be positive.")

        if status_quo is not None:
            self.experiment.search_space.check_types(
                status_quo.parameters, raise_error=True
            )
            self.experiment._name_and_store_arm_if_not_exists(
                arm=status_quo, proposed_name="status_quo_" + str(self.index)
            )
        self._status_quo = status_quo
        self.reweight_status_quo(weight)
        return self

    @immutable_once_run
    def reweight_status_quo(self, weight: Optional[float] = None) -> "BatchTrial":
        """Update status quo weight.

        If arms have been added since the status quo was initially added,
        the optimal weight of the status quo may change.
        """
        status_quo = self._status_quo
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
    def arms(self) -> List[Arm]:
        """All arms contained in the trial."""
        arm_weights = self.arm_weights
        return [] if arm_weights is None else list(arm_weights.keys())

    @property
    def weights(self) -> List[float]:
        """Weights corresponding to arms contained in the trial."""
        arm_weights = self.arm_weights
        return [] if arm_weights is None else list(arm_weights.values())

    @property
    def arms_by_name(self) -> Dict[str, Arm]:
        """Map from arm name to object for all arms in trial."""
        arms_by_name = {}
        for arm in self.arms:
            if not arm.has_name:
                raise ValueError(  # pragma: no cover
                    "Arms attached to a trial must have a name."
                )
            arms_by_name[arm.name] = arm
        return arms_by_name

    @property
    def abandoned_arms(self) -> List[Arm]:
        """List of arms that have been abandoned within this trial"""
        return [
            self.arms_by_name[arm.name]
            for arm in self._abandoned_arms_metadata.values()
        ]

    @property
    def abandoned_arms_metadata(self) -> List[AbandonedArm]:
        return list(self._abandoned_arms_metadata.values())

    @property
    def is_factorial(self) -> bool:
        """Return true if the trial's arms are a factorial design with
        no linked factors.
        """
        # To match the model behavior, this should probably actually be pulled
        # from exp.parameters. However, that seems rather ugly when this function
        # intuitively should just depend on the arms.
        sufficient_factors = all(len(arm.parameters or []) >= 2 for arm in self.arms)
        if not sufficient_factors:
            return False
        param_levels: DefaultDict[str, Dict[Union[str, float], int]] = (
            defaultdict(dict)
        )
        for arm in self.arms:
            for param_name, param_value in arm.parameters.items():
                # Expected `Union[float, str]` for 2nd anonymous parameter to call
                # `dict.__setitem__` but got `Optional[Union[bool, float, str]]`.
                # pyre-fixme[6]: Expected `Union[float, str]` for 1st param but got `...
                param_levels[param_name][param_value] = 1
        param_cardinality = 1
        for param_values in param_levels.values():
            param_cardinality *= len(param_values)
        return len(self.arms) == param_cardinality

    def run(self) -> "BatchTrial":
        return checked_cast(BatchTrial, super().run())

    def normalized_arm_weights(
        self, total: float = 1, trunc_digits: Optional[int] = None
    ) -> MutableMapping[Arm, float]:
        """Returns arms with a new set of weights normalized
        to the given total.

        This method is useful for many runners where we need to normalize weights
        to a certain total without mutating the weights attached to a trial.

        Args:
            total: The total weight to which to normalize.
                Default is 1, in which case arm weights
                can be interpreted as probabilities.
            trunc_digits: The number of digits to keep. If the
                resulting total weight is not equal to `total`, re-allocate
                weight in such a way to maintain relative weights as best as
                possible.

        Returns:
            Mapping from arms to the new set of weights.

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
        return OrderedDict(zip(self.arms, weights))

    def mark_arm_abandoned(
        self, arm_name: str, reason: Optional[str] = None
    ) -> "BatchTrial":
        """Mark a arm abandoned.

        Usually done after deployment when one arm causes issues but
        user wants to continue running other arms in the batch.

        Args:
            arm_name: The name of the arm to abandon.
            reason: The reason for abandoning the arm.

        Returns:
            The batch instance.
        """
        if arm_name not in self.arms_by_name:
            raise ValueError("Arm must be contained in batch.")

        abandoned_arm = AbandonedArm(name=arm_name, time=datetime.now(), reason=reason)
        self._abandoned_arms_metadata[arm_name] = abandoned_arm
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
            f"experiment_name='{self._experiment._name}', "
            f"index={self._index}, "
            f"status={self._status})"
        )
