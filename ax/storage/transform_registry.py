#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.bilog_y import BilogY
from ax.modelbridge.transforms.choice_encode import (
    ChoiceEncode,
    ChoiceToNumericChoice,
    OrderedChoiceEncode,
    OrderedChoiceToIntegerRange,
)
from ax.modelbridge.transforms.convert_metric_names import ConvertMetricNames
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.int_to_float import IntToFloat, LogIntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.log_y import LogY
from ax.modelbridge.transforms.logit import Logit
from ax.modelbridge.transforms.map_key_to_float import MapKeyToFloat
from ax.modelbridge.transforms.merge_repeated_measurements import (
    MergeRepeatedMeasurements,
)
from ax.modelbridge.transforms.metrics_as_task import MetricsAsTask
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.power_transform_y import PowerTransformY
from ax.modelbridge.transforms.relativize import (
    Relativize,
    RelativizeWithConstantControl,
)
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice, TaskEncode
from ax.modelbridge.transforms.time_as_feature import TimeAsFeature
from ax.modelbridge.transforms.transform_to_new_sq import TransformToNewSQ
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.winsorize import Winsorize


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
    ConvertMetricNames,
    Derelativize,
    IntRangeToChoice,
    IntToFloat,
    IVW,
    Log,
    OneHot,
    OrderedChoiceEncode,  # TO BE DEPRECATED
    OrderedChoiceToIntegerRange,
    # This transform was upstreamed into the base modelbridge.
    # DEPRECATED: OutOfDesign
    RemoveFixed,
    SearchSpaceToChoice,
    StandardizeY,
    StratifiedStandardizeY,
    TaskEncode,  # TO BE DEPRECATED
    TaskChoiceToIntTaskChoice,
    TrialAsTask,
    UnitX,
    Winsorize,
    # CapParameter,  DEPRECATED
    PowerTransformY,
    ChoiceEncode,  # TO BE DEPRECATED
    ChoiceToNumericChoice,
    Logit,
    # MapUnitX, DEPRECATED
    MetricsAsTask,
    LogY,
    Relativize,
    RelativizeWithConstantControl,
    MergeRepeatedMeasurements,
    TimeAsFeature,
    TransformToNewSQ,
    FillMissingParameters,
    LogIntToFloat,
    MapKeyToFloat,
    BilogY,
}

REMOVED_TRANSFORMS: set[str] = {
    "OutOfDesign",
    "CapParameter",
    "MapUnitX",
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
