#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TYPE_CHECKING

from ax.core.observation import Observation
from ax.core.parameter import RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.typeutils import checked_cast

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class CapParameter(Transform):
    """Cap parameter range(s) to given values. Expects a configuration of form
    { parameter_name -> new_upper_range_value }.

    This transform only transforms the search space.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.config = config or {}
        assert search_space is not None, "CapParameter requires search space"
        # pyre-fixme[4]: Attribute must be annotated.
        self.transform_parameters = {  # Only transform parameters in config.
            p_name for p_name in search_space.parameters if p_name in self.config
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters:
                if not isinstance(p, RangeParameter):
                    raise NotImplementedError(
                        "Can only cap range parameters currently."
                    )
                checked_cast(RangeParameter, p).update_range(
                    upper=self.config.get(p_name)
                )
        return search_space
