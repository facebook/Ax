#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Any, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.metadata_to_parameter import MetadataToParameterMixin
from ax.core import ParameterType
from ax.core.parameter import ChoiceParameter, TParamValue
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


logger: Logger = get_logger(__name__)


class MetadataToTask(MetadataToParameterMixin, Transform):
    """
    This transform converts metadata from observation features into a task parameter.

    This is used in transfer learning to specify the task for each observation.

    It allows the user to specify the `config` with `task_values` as the key, which
    contains the unique task values for the search space.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        Transform.__init__(
            self,
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        config = config or {}
        task_values: list[TParamValue] = [
            assert_is_instance(v, TParamValue)
            for v in assert_is_instance(config["task_values"], list)
        ]
        self.parameters: dict[str, dict[str, Any]] = {Keys.TASK_FEATURE_NAME.value: {}}
        self._parameter_list = [
            ChoiceParameter(
                name=Keys.TASK_FEATURE_NAME.value,
                parameter_type=ParameterType.INT,
                values=task_values,
                is_task=True,
                target_value=0,
                is_ordered=False,
                sort_values=True,
            )
        ]
