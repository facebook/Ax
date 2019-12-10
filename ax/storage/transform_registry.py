#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Type

from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cap_parameter import CapParameter
from ax.modelbridge.transforms.convert_metric_names import ConvertMetricNames
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.ordered_choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskEncode
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
TRANSFORM_REGISTRY: Dict[Type[Transform], int] = {
    ConvertMetricNames: 0,
    Derelativize: 1,
    IntRangeToChoice: 2,
    IntToFloat: 3,
    IVW: 4,
    Log: 5,
    OneHot: 6,
    OrderedChoiceEncode: 7,
    # This transform was upstreamed into the base modelbridge.
    # Old transforms serialized with this will have the OutOfDesign transform
    # replaced with a no-op, the base transform.
    # DEPRECATED: OutOfDesign: 8
    Transform: 8,
    RemoveFixed: 9,
    SearchSpaceToChoice: 10,
    StandardizeY: 11,
    StratifiedStandardizeY: 12,
    TaskEncode: 13,
    TrialAsTask: 14,
    UnitX: 15,
    Winsorize: 16,
    CapParameter: 17,
}


REVERSE_TRANSFORM_REGISTRY: Dict[int, Type[Transform]] = {
    v: k for k, v in TRANSFORM_REGISTRY.items()
}
