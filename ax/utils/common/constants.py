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
    CANDIDATE_SET = "candidate_set"
    CANDIDATE_SIZE = "candidate_size"
    COST_AWARE_UTILITY = "cost_aware_utility"
    COST_INTERCEPT = "cost_intercept"
    CURRENT_VALUE = "current_value"
    EXPAND = "expand"
    EXPECTED_ACQF_VAL = "expected_acquisition_value"
    FIDELITY_WEIGHTS = "fidelity_weights"
    FRAC_RANDOM = "frac_random"
    MAXIMIZE = "maximize"
    NUM_INNER_RESTARTS = "num_inner_restarts"
    NUM_RESTARTS = "num_restarts"
    NUM_TRACE_OBSERVATIONS = "num_trace_observations"
    OPTIMIZER_KWARGS = "optimizer_kwargs"
    PROJECT = "project"
    QMC = "qmc"
    RAW_INNER_SAMPLES = "raw_inner_samples"
    RAW_SAMPLES = "raw_samples"
    SAMPLER = "sampler"
    SEED_INNER = "seed_inner"
    SEQUENTIAL = "sequential"
    SUBCLASS = "subclass"
    SUBSET_MODEL = "subset_model"
