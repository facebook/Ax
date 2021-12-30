#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from ax.storage.sqa_store.sqa_classes import SQAGeneratorRun
from sqlalchemy.orm import defaultload, lazyload, strategy_options
from sqlalchemy.orm.attributes import InstrumentedAttribute


GR_LARGE_MODEL_ATTRS: List[InstrumentedAttribute] = [  # pyre-ignore[9]
    SQAGeneratorRun.model_kwargs,
    SQAGeneratorRun.bridge_kwargs,
    SQAGeneratorRun.model_state_after_gen,
    SQAGeneratorRun.gen_metadata,
]


GR_PARAMS_METRICS_COLS = [
    "parameters",
    "parameter_constraints",
    "metrics",
]


def get_query_options_to_defer_immutable_duplicates() -> List[strategy_options.Load]:
    """Returns the query options that defer loading of attributes that are duplicated
    on each trial (like search space attributes and metrics). These attributes do not
    need to be loaded for experiments with immutable search space and optimization
    configuration.
    """
    options = [lazyload(f"generator_runs.{col}") for col in GR_PARAMS_METRICS_COLS]
    return options


def get_query_options_to_defer_large_model_cols() -> List[strategy_options.Load]:
    """Returns the query options that defer loading of model-state-related columns
    of generator runs, which can be large and are not needed on every generator run
    when loading experiment and generation strategy in reduced state.
    """
    return [
        defaultload("generator_runs").defer(col.key) for col in GR_LARGE_MODEL_ATTRS
    ]
