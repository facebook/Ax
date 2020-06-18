#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, unique


# -------------------------- Error messages --------------------------


UNEXPECTED_METRIC_COMBINATION = """
Unexpected combination of dummy base `Metric` class metrics and `Metric`
subclasses with defined fetching logic.
"""


# --------------------------- Reserved keys ---------------------------


@unique
class Keys(str, Enum):
    """Enum of reserved keys in options dicts etc, alphabetized.


    NOTE: Useful for keys in dicts that correspond to kwargs to
    classes or functions and/or are used in multiple places.
    """

    ACQF_KWARGS = "acquisition_function_kwargs"
    BATCH_INIT_CONDITIONS = "batch_initial_conditions"
    CURRENT_VALUE = "current_value"
    EXPECTED_ACQF_VAL = "expected_acquisition_value"
    FRAC_RANDOM = "frac_random"
    NUM_INNER_RESTARTS = "num_inner_restarts"
    NUM_RESTARTS = "num_restarts"
    OPTIMIZER_KWARGS = "optimizer_kwargs"
    RAW_INNER_SAMPLES = "raw_inner_samples"
    RAW_SAMPLES = "raw_samples"
    SUBSET_MODEL = "subset_model"
