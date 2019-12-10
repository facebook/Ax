#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Tuple

import numpy as np
from ax.core.types import TConfig, TGenMetadata, TParamValue, TParamValueList
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class FullFactorialGenerator(DiscreteModel):
    """Generator for full factorial designs.

    Generates arms for all possible combinations of parameter values,
    each with weight 1.

    The value of n supplied to `gen` will be ignored, as the number
    of arms generated is determined by the list of parameter values.
    To suppress this warning, use n = -1.
    """

    def __init__(
        self, max_cardinality: int = 100, check_cardinality: bool = True
    ) -> None:
        """
        Args:
            max_cardinality: maximum number of arms allowed if
                check_cardinality == True. Default is 100.
            check_cardinality: if True, throw if number of arms
                exceeds max_cardinality.
        """
        super().__init__()
        self.max_cardinality = max_cardinality
        self.check_cardinality = check_cardinality

    @copy_doc(DiscreteModel.gen)
    def gen(
        self,
        n: int,
        parameter_values: List[TParamValueList],
        objective_weights: Optional[np.ndarray],
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, TParamValue]] = None,
        pending_observations: Optional[List[List[TParamValueList]]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Tuple[List[TParamValueList], List[float], TGenMetadata]:
        if n != -1:
            raise ValueError(
                "FullFactorialGenerator will ignore the specified value of n. "
                "The generator automatically determines how many arms to "
                "generate."
            )

        if fixed_features:
            for fixed_feature_index, fixed_feature_value in fixed_features.items():
                parameter_values[fixed_feature_index] = [fixed_feature_value]

        num_arms = reduce(mul, [len(values) for values in parameter_values], 1)
        if self.check_cardinality and num_arms > self.max_cardinality:
            raise ValueError(
                f"FullFactorialGenerator generated {num_arms} arms, "
                f"but the maximum number of arms allowed is "
                f"{self.max_cardinality}."
            )

        points = [list(x) for x in itertools.product(*parameter_values)]
        return (points, [1.0 for _ in range(len(points))], {})
