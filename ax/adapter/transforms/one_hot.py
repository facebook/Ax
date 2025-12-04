#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.rounding import randomized_onehot_round, strict_onehot_round
from ax.adapter.transforms.utils import construct_new_search_space
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization, TParamValue
from ax.generators.types import TConfig
from pyre_extensions import assert_is_instance

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


OH_PARAM_INFIX = "_OH_PARAM"


class OneHotEncoder:
    """OneHot encodes a list of labels."""

    def __init__(self, values: list[TParamValue]) -> None:
        assert len(values) >= 2
        self.values: list[TParamValue] = values
        self.encoded_len: int = 1 if len(values) == 2 else len(values)

    def transform(self, label: TParamValue) -> list[int]:
        """One hot encode a given label."""
        effective_index = self.values.index(label)
        if self.encoded_len == 1:
            return [effective_index]
        else:
            encoding = [0 for _ in range(self.encoded_len)]
            encoding[effective_index] = 1
            return encoding

    def inverse_transform(self, encoded_label: list[int]) -> TParamValue:
        """Inverse transorm a one hot encoded label."""
        if self.encoded_len == 1:
            return self.values[encoded_label[0]]
        else:
            return self.values[encoded_label.index(1)]


class OneHot(Transform):
    """Convert categorical parameters (unordered ChoiceParameters) to
    one-hot-encoded parameters.

    Does not convert task parameters.

    Parameters will be one-hot-encoded, yielding a set of RangeParameters,
    of type float, on [0, 1]. If there are two values, one single RangeParameter
    will be yielded, otherwise there will be a new RangeParameter for each
    ChoiceParameter value.

    In the reverse transform, floats can be converted to a one-hot encoded vector
    using one of two methods:

    Strict rounding: Choose the maximum value. With levels ['a', 'b', 'c'] and
        float values [0.2, 0.4, 0.3], the restored parameter would be set to 'b'.
        Ties are broken randomly, so values [0.2, 0.4, 0.4] is randomly set to 'b'
        or 'c'.

    Randomized rounding: Sample from the distribution. Float values
        [0.2, 0.4, 0.3] are transformed to 'a' w.p.
        0.2/0.9, 'b' w.p. 0.4/0.9, or 'c' w.p. 0.3/0.9.

    Type of rounding can be set using transform_config['rounding'] to either
    'strict' or 'randomized'. Defaults to strict.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "OneHot requires search space"
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        # pyre-fixme[4]: Attribute must be annotated.
        self.rounding = "strict"
        if self.config is not None:
            self.rounding = self.config.get("rounding", "strict")
        self.encoder: dict[str, OneHotEncoder] = {}
        self.encoded_parameters: dict[str, list[str]] = {}
        self.encoded_values: dict[str, list[TParamValue]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and not p.is_ordered and not p.is_task:
                values = deepcopy(p.values)
                self.encoded_values[p.name] = values
                self.encoder[p.name] = OneHotEncoder(values)
                encoded_len = self.encoder[p.name].encoded_len
                if encoded_len == 1:
                    # Two levels handled in one parameter
                    self.encoded_parameters[p.name] = [p.name + OH_PARAM_INFIX]
                else:
                    self.encoded_parameters[p.name] = [
                        f"{p.name}{OH_PARAM_INFIX}_{i}" for i in range(encoded_len)
                    ]

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, encoder in self.encoder.items():
                if p_name in obsf.parameters:
                    vals = encoder.transform(label=obsf.parameters.pop(p_name))
                    updated_parameters: TParameterization = {
                        self.encoded_parameters[p_name][i]: v
                        for i, v in enumerate(vals)
                    }
                    obsf.parameters.update(updated_parameters)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.encoded_parameters:
                p = assert_is_instance(p, ChoiceParameter)
                if p.is_fidelity:
                    raise ValueError(
                        f"Cannot one-hot-encode fidelity parameter {p_name}"
                    )
                if not set(p.values).issubset(self.encoded_values[p_name]):
                    raise ValueError(
                        f"{p_name} has values {p.values} which are not a subset of "
                        f"the original values {self.encoded_values[p_name]} used to "
                        "initialize the transform."
                    )
                encoded_p = self.encoded_parameters[p_name]
                if len(encoded_p) > 1:
                    # Remove any parameters that are not in the search space being
                    # transformed. This is necessary if the search space used to
                    # initialize the transform is larger than the search space
                    # being transformed, to ensure that the missing parameters
                    # do not get selected.
                    encoded_p = [
                        encoded_p[self.encoded_values[p_name].index(v)]
                        for v in p.values
                    ]
                for new_p_name in encoded_p:
                    transformed_parameters[new_p_name] = RangeParameter(
                        name=new_p_name,
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=1,
                    )
            else:
                transformed_parameters[p_name] = p
        return construct_new_search_space(
            search_space=search_space,
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone_with_transformed_parameters(
                    transformed_parameters=transformed_parameters
                )
                for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoder.keys():
                has_params = [
                    p in obsf.parameters for p in self.encoded_parameters[p_name]
                ]
                if not any(has_params):
                    continue
                x = np.array(
                    [
                        # If the parameter isn't present, default to -1 ensure it
                        # does not get selected after rounding.
                        obsf.parameters.pop(p, -1.0)
                        for p in self.encoded_parameters[p_name]
                    ]
                )
                if self.rounding == "strict":
                    x = strict_onehot_round(x)
                else:
                    x = randomized_onehot_round(x)
                val = self.encoder[p_name].inverse_transform(
                    encoded_label=x.astype(int).tolist()
                )
                obsf.parameters[p_name] = val
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        for p_name, values in self.encoded_values.items():
            # First, replace values with 0, 1, 2, so that column names are as expected.
            arm_data[p_name] = arm_data[p_name].map(
                {v: i for i, v in enumerate(values)}
            )

            if len(values) == 2:
                # Handle the special case. Only need to rename the column.
                arm_data = arm_data.rename(columns={p_name: p_name + OH_PARAM_INFIX})
            else:
                # Use get_dummies to one-hot encode the column.
                arm_data = pd.get_dummies(
                    arm_data,
                    columns=[p_name],
                    prefix=p_name + OH_PARAM_INFIX,
                    # Could be int, but using float to match the parameter type.
                    dtype=float,
                )
                # Make sure all expected columns are present, even if there is no
                # corresponding value in the data.
                for i in range(len(values)):
                    if f"{p_name}{OH_PARAM_INFIX}_{i}" not in arm_data:
                        arm_data[f"{p_name}{OH_PARAM_INFIX}_{i}"] = 0.0

        return ExperimentData(
            arm_data=arm_data, observation_data=experiment_data.observation_data
        )
