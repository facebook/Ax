#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterator
from copy import deepcopy
from unittest.mock import ANY

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.metadata_to_float import MetadataToFloat
from ax.core.observation import ObservationFeatures, observations_from_data
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


WIDTHS = [2.0, 4.0, 8.0]
HEIGHTS = [4.0, 2.0, 8.0]
STEPS_ENDS = [1, 5, 3]


def _enumerate() -> Iterator[tuple[float, float, float]]:
    yield from (
        (width, height, float(i + 1))
        for (width, height, steps_end) in zip(WIDTHS, HEIGHTS, STEPS_ENDS)
        for i in range(steps_end)
    )


class MetadataToFloatTransformTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="width",
                    parameter_type=ParameterType.FLOAT,
                    lower=1,
                    upper=20,
                ),
                RangeParameter(
                    name="height",
                    parameter_type=ParameterType.FLOAT,
                    lower=1,
                    upper=20,
                ),
            ]
        )
        self.experiment = get_experiment_with_observations(
            observations=[[0.0] for _ in range(sum(STEPS_ENDS))],
            search_space=self.search_space,
            parameterizations=[
                {"width": w, "height": h}
                for steps_end, w, h in zip(STEPS_ENDS, WIDTHS, HEIGHTS)
                for _ in range(steps_end)
            ],
            candidate_metadata=[
                {
                    "foo": 42,
                    "bar": 3.0 * steps,
                }
                for steps_end, _w, _h in zip(STEPS_ENDS, WIDTHS, HEIGHTS)
                for steps in range(1, steps_end + 1)
            ],
        )
        self.observations = observations_from_data(
            experiment=self.experiment, data=self.experiment.lookup_data()
        )
        self.experiment_data = extract_experiment_data(
            experiment=self.experiment,
            data_loader_config=DataLoaderConfig(),
        )

        self.t = MetadataToFloat(
            experiment_data=self.experiment_data,
            config={
                "parameters": {"bar": {"digits": 4}},
            },
        )

    def test_Init(self) -> None:
        self.assertEqual(len(self.t._parameter_list), 1)
        p = self.t._parameter_list[0]
        # check that the parameter options are specified in a sensible manner
        # by default if the user does not specify them explicitly
        self.assertEqual(p.name, "bar")
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 3.0)
        self.assertEqual(p.upper, 15.0)
        self.assertFalse(p.log_scale)
        self.assertFalse(p.logit_scale)
        self.assertEqual(p.digits, 4)
        self.assertFalse(p.is_fidelity)
        self.assertIsNone(p.target_value)

        with self.assertRaisesRegex(DataRequiredError, "requires non-empty data"):
            MetadataToFloat(search_space=None)

    def test_TransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        self.assertSetEqual(
            set(ss2.parameters.keys()),
            {"height", "width", "bar"},
        )

        p = assert_is_instance(ss2.parameters["bar"], RangeParameter)

        self.assertEqual(p.name, "bar")
        self.assertEqual(p.parameter_type, ParameterType.FLOAT)
        self.assertEqual(p.lower, 3.0)
        self.assertEqual(p.upper, 15.0)
        self.assertFalse(p.log_scale)
        self.assertFalse(p.logit_scale)
        self.assertEqual(p.digits, 4)
        self.assertFalse(p.is_fidelity)
        self.assertIsNone(p.target_value)

    def test_TransformObservationFeatures(self) -> None:
        observation_features = [obs.features for obs in self.observations]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)

        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    trial_index=trial_index,
                    parameters={
                        "width": width,
                        "height": height,
                        "bar": 3.0 * steps,
                    },
                    metadata={"foo": 42, Keys.TRIAL_COMPLETION_TIMESTAMP: ANY},
                )
                for trial_index, (width, height, steps) in enumerate(_enumerate())
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)

    def test_transform_experiment_data(self) -> None:
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(self.experiment_data)
        )
        # Check that arm data now has a new column for the transformed parameter.
        expected_bar_values = [
            3.0 * s for steps in STEPS_ENDS for s in range(1, steps + 1)
        ]
        self.assertEqual(transformed_data.arm_data["bar"].tolist(), expected_bar_values)
        # Remaining columns are unchanged.
        assert_frame_equal(
            transformed_data.arm_data.drop(columns=["bar"]),
            self.experiment_data.arm_data,
        )
        # Observation data is not changed.
        assert_frame_equal(
            transformed_data.observation_data,
            self.experiment_data.observation_data,
        )
