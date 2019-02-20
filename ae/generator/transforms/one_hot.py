#!/usr/bin/env python3

from typing import Any, Dict, List, Optional

import numpy as np
from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.parameter import (
    ChoiceParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig, TParameterization
from ae.lazarus.ae.generator.transforms.base import Transform
from ae.lazarus.ae.generator.transforms.rounding import (
    randomized_onehot_round,
    strict_onehot_round,
)
from sklearn.preprocessing import LabelBinarizer


OH_PARAM_INFIX = "_OH_PARAM_"


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
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        # Identify parameters that should be transformed
        self.rounding = "strict"
        if config is not None:
            self.rounding = config.get("rounding", "strict")
        self.encoder: Dict[str, Any] = {}
        self.encoded_params: Dict[str, List[str]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and not p.is_ordered and not p.is_task:
                self.encoder[p.name] = LabelBinarizer().fit(p.values)
                nc = len(self.encoder[p.name].classes_)
                if nc == 2:
                    # Two levels handled in one parameter
                    self.encoded_params[p.name] = [p.name + OH_PARAM_INFIX]
                else:
                    self.encoded_params[p.name] = [
                        "{}{}_{}".format(p.name, OH_PARAM_INFIX, i) for i in range(nc)
                    ]

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, encoder in self.encoder.items():
                if p_name in obsf.parameters:
                    vals = encoder.transform([obsf.parameters.pop(p_name)])[0]
                    updated_params: TParameterization = {
                        self.encoded_params[p_name][i]: v for i, v in enumerate(vals)
                    }
                    obsf.parameters.update(updated_params)
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p in search_space.parameters.values():
            if p.name in self.encoded_params:
                for new_p_name in self.encoded_params[p.name]:
                    transformed_parameters[new_p_name] = RangeParameter(
                        name=new_p_name,
                        parameter_type=ParameterType.FLOAT,
                        lower=0,
                        upper=1,
                    )
            else:
                transformed_parameters[p.name] = p
        return SearchSpace(
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone() for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoder.keys():
                x = np.array(
                    [obsf.parameters.pop(p) for p in self.encoded_params[p_name]]
                )
                if self.rounding == "strict":
                    x = strict_onehot_round(x)
                else:
                    x = randomized_onehot_round(x)
                val = self.encoder[p_name].inverse_transform(x[None, :])[0]
                if isinstance(val, np.bool_):
                    val = bool(val)  # Numpy bools don't serialize
                obsf.parameters[p_name] = val
        return observation_features
