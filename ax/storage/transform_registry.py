#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.bilog_y import BilogY
from ax.adapter.transforms.choice_encode import (
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
)
from ax.adapter.transforms.derelativize import Derelativize
from ax.adapter.transforms.fill_missing_parameters import FillMissingParameters
from ax.adapter.transforms.fixed_to_tunable import FixedToTunable
from ax.adapter.transforms.int_range_to_choice import IntRangeToChoice
from ax.adapter.transforms.int_to_float import IntToFloat
from ax.adapter.transforms.ivw import IVW
from ax.adapter.transforms.log import Log
from ax.adapter.transforms.log_y import LogY
from ax.adapter.transforms.logit import Logit
from ax.adapter.transforms.map_key_to_float import MapKeyToFloat
from ax.adapter.transforms.merge_repeated_measurements import MergeRepeatedMeasurements
from ax.adapter.transforms.metadata_to_task import MetadataToTask
from ax.adapter.transforms.metrics_as_task import MetricsAsTask
from ax.adapter.transforms.one_hot import OneHot
from ax.adapter.transforms.power_transform_y import PowerTransformY
from ax.adapter.transforms.relativize import (
    Relativize,
    RelativizeWithConstantControl,
    SelectiveRelativizeWithConstantControl,
)
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.adapter.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.adapter.transforms.time_as_feature import TimeAsFeature
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.adapter.transforms.trial_as_task import TrialAsTask
from ax.adapter.transforms.unit_x import UnitX
from ax.adapter.transforms.winsorize import Winsorize


"""
A registry of transform classes for storage.

Transforms are stored in the DB as part of the JSON-encoded
GenerationNode (or Step) objects. When loading the GenerationStrategy,
we will look up the transform class that matches the stringified
transform type in JSON object, and return the matching transform class.

NOTE: If removing a transform, please add it to REMOVED_TRANSFORMS.
These will be discarded while loading experiments from the DB.
If deprecating a transform in favor of a new one (i.e. renaming),
please add it to DEPRECATED_TRANSFORMS. When loading from the DB,
we will return the replacement class.
"""
TRANSFORM_REGISTRY: set[type[Transform]] = {
    Transform,
    # ConvertMetricNames, DEPRECATED
    Derelativize,
    FixedToTunable,
    IntRangeToChoice,
    IntToFloat,
    IVW,
    Log,
    OneHot,
    # OrderedChoiceEncode,  # DEPRECATED
    OrderedChoiceToIntegerRange,
    # This transform was upstreamed into the base adapter.
    # DEPRECATED: OutOfDesign
    RemoveFixed,
    SearchSpaceToChoice,
    StandardizeY,
    StratifiedStandardizeY,
    # TaskEncode,  # DEPRECATED
    TaskChoiceToIntTaskChoice,
    TrialAsTask,
    UnitX,
    Winsorize,
    # CapParameter,  DEPRECATED
    PowerTransformY,
    # ChoiceEncode,  # DEPRECATED
    ChoiceToNumericChoice,
    Logit,
    # MapUnitX, DEPRECATED
    # MetadataToFloat, DEPRECATED
    MetadataToTask,
    MetricsAsTask,
    LogY,
    Relativize,
    RelativizeWithConstantControl,
    SelectiveRelativizeWithConstantControl,
    MergeRepeatedMeasurements,
    TimeAsFeature,
    TransformToNewSQ,
    FillMissingParameters,
    # LogIntToFloat, DEPRECATED
    MapKeyToFloat,
    BilogY,
}

REMOVED_TRANSFORMS: set[str] = {
    "OutOfDesign",
    "CapParameter",
    "MapUnitX",
    "ConvertMetricNames",
    "LogIntToFloat",
    "MetadataToFloat",
}

"""
Dict mapping transforms which are being deprecated to the transforms replacing them.
These will be present in TRANSFORM_REGISTRY so that old call sites
can still store properly, but when loading back the new class will be used.
"""
DEPRECATED_TRANSFORMS: dict[str, type[Transform]] = {
    "OrderedChoiceEncode": OrderedChoiceToIntegerRange,
    "ChoiceEncode": ChoiceToNumericChoice,
    "TaskEncode": TaskChoiceToIntTaskChoice,
}

REVERSE_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {
    k.__name__: k for k in TRANSFORM_REGISTRY if k not in DEPRECATED_TRANSFORMS
}


def register_transform(transform: type[Transform]) -> None:
    """Register a transform class for storage.

    This is intended for external additions. Transforms that are available in
    the core library should be added to the TRANSFORM_REGISTRY directly.
    """
    TRANSFORM_REGISTRY.add(transform)
    REVERSE_TRANSFORM_REGISTRY[transform.__name__] = transform
