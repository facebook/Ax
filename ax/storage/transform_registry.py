#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.modelbridge.transforms.base import Transform
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
from ax.modelbridge.transforms.map_unit_x import MapUnitX
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


# TODO: Annotate and add `register_transform`

"""
Mapping of Transform classes to ints.

All transforms will be stored in the same table in the database. When
saving, we look up the transform subclass in TRANSFORM_REGISTRY, and store
the corresponding type field in the database. When loading, we look
up the type field in REVERSE_TRANSFORM_REGISTRY, and initialize the
corresponding transform subclass.
"""
TRANSFORM_REGISTRY: dict[type[Transform], int] = {
    ConvertMetricNames: 0,
    Derelativize: 1,
    IntRangeToChoice: 2,
    IntToFloat: 3,
    IVW: 4,
    Log: 5,
    OneHot: 6,
    OrderedChoiceEncode: 7,  # TO BE DEPRECATED
    OrderedChoiceToIntegerRange: 7,
    # This transform was upstreamed into the base modelbridge.
    # Old transforms serialized with this will have the OutOfDesign transform
    # replaced with a no-op, the base transform.
    # DEPRECATED: OutOfDesign: 8
    Transform: 8,
    RemoveFixed: 9,
    SearchSpaceToChoice: 10,
    StandardizeY: 11,
    StratifiedStandardizeY: 12,
    TaskEncode: 13,  # TO BE DEPRECATED
    TaskChoiceToIntTaskChoice: 13,
    TrialAsTask: 14,
    UnitX: 15,
    Winsorize: 16,
    # CapParameter: 17,  DEPRECATED
    PowerTransformY: 18,
    ChoiceEncode: 19,  # TO BE DEPRECATED
    ChoiceToNumericChoice: 19,
    Logit: 20,
    MapUnitX: 21,
    MetricsAsTask: 22,
    LogY: 23,
    Relativize: 24,
    RelativizeWithConstantControl: 25,
    MergeRepeatedMeasurements: 26,
    TimeAsFeature: 27,
    TransformToNewSQ: 28,
    FillMissingParameters: 29,
    LogIntToFloat: 30,
}

"""
List transforms which are be deprecated.
These will be present in TRANSFORM_REGISTRY so that old call sites
can still store properly, but when loading back the new class will
be used.
"""
DEPRECATED_TRANSFORMS: list[type[Transform]] = [
    OrderedChoiceEncode,  # replaced by OrderedChoiceToIntegerRange
    ChoiceEncode,  # replaced by ChoiceToNumericChoice
    TaskEncode,  # replaced by TaskChoiceToIntTaskChoice
]

REVERSE_TRANSFORM_REGISTRY: dict[int, type[Transform]] = {
    v: k for k, v in TRANSFORM_REGISTRY.items() if k not in DEPRECATED_TRANSFORMS
}
