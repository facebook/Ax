#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
from ax.adapter.adapter_utils import extract_search_space_digest
from ax.adapter.random import RandomAdapter
from ax.adapter.registry import Cont_X_trans
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.generators.random.base import RandomGenerator
from ax.generators.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_data
from ax.utils.testing.modeling_stubs import get_experiment_for_value


class RandomAdapterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        x = RangeParameter("x", ParameterType.FLOAT, lower=0, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=1, upper=2)
        z = RangeParameter("z", ParameterType.FLOAT, lower=0, upper=5)
        self.parameters = [x, y, z]
        parameter_constraints: list[ParameterConstraint] = [
            ParameterConstraint(inequality="x <= y"),
            ParameterConstraint(inequality="x + z >= 3.5"),
        ]
        self.search_space = SearchSpace(self.parameters, parameter_constraints)
        self.experiment = Experiment(search_space=self.search_space)
        self.model_gen_options = {"option": "yes"}

    def test_fit(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, generator=RandomGenerator())
        self.assertEqual(adapter.parameters, ["x", "y", "z"])
        self.assertTrue(isinstance(adapter.generator, RandomGenerator))

    def test_predict(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, generator=RandomGenerator())
        with self.assertRaises(NotImplementedError):
            adapter._predict([])

    def test_cross_validate(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, generator=RandomGenerator())
        with self.assertRaises(NotImplementedError):
            # pyre-ignore[6]: None input for testing.
            adapter._cross_validate(self.search_space, None, None)

    def test_gen_w_constraints(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, generator=RandomGenerator())
        with mock.patch.object(
            adapter.generator,
            "gen",
            return_value=(
                np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]),
                np.array([1.0, 2.0]),
            ),
        ) as mock_gen:
            gen_results = adapter._gen(
                n=3,
                search_space=self.search_space,
                pending_observations={},
                fixed_features=ObservationFeatures({"z": 3.0}),
                optimization_config=None,
                # pyre-fixme[6]: For 6th param expected `Optional[Dict[str,
                # Union[None, Dict[str, typing.Any], OptimizationConfig,
                # AcquisitionFunction, float, int, str]]]` but got `Dict[str,
                # str]`.
                model_gen_options=self.model_gen_options,
            )
        gen_args = mock_gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        ssd = gen_args["search_space_digest"]
        self.assertEqual(
            ssd,
            extract_search_space_digest(
                self.search_space, list(self.search_space.parameters.keys())
            ),
        )
        self.assertEqual(ssd.bounds, [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
        self.assertTrue(
            np.array_equal(
                gen_args["linear_constraints"][0],
                np.array([[1.0, -1, 0.0], [-1.0, 0.0, -1.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(gen_args["linear_constraints"][1], np.array([[0.0], [-3.5]]))
        )
        self.assertEqual(gen_args["fixed_features"], {2: 3.0})
        self.assertEqual(gen_args["model_gen_options"], {"option": "yes"})
        obsf = gen_results.observation_features
        self.assertEqual(obsf[0].parameters, {"x": 1.0, "y": 2.0, "z": 3.0})
        self.assertEqual(obsf[1].parameters, {"x": 3.0, "y": 4.0, "z": 3.0})
        self.assertTrue(np.array_equal(gen_results.weights, np.array([1.0, 2.0])))

    def test_gen_simple(self) -> None:
        # Test with no constraints, no fixed feature, no pending observations
        search_space = SearchSpace(self.parameters[:2])
        adapter = RandomAdapter(
            experiment=Experiment(search_space=search_space),
            generator=RandomGenerator(),
        )
        with mock.patch.object(
            adapter.generator,
            "gen",
            return_value=(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0])),
        ) as mock_gen:
            adapter._gen(
                n=3,
                search_space=search_space,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
                optimization_config=None,
                # pyre-fixme[6]: For 6th param expected `Optional[Dict[str,
                # Union[None, Dict[str, typing.Any], OptimizationConfig,
                # AcquisitionFunction, float, int, str]]]` but got `Dict[str,
                # str]`.
                model_gen_options=self.model_gen_options,
            )
        gen_args = mock_gen.mock_calls[0][2]
        ssd = gen_args["search_space_digest"]
        self.assertEqual(
            ssd,
            extract_search_space_digest(
                search_space, list(search_space.parameters.keys())
            ),
        )
        self.assertEqual(ssd.bounds, [(0.0, 1.0), (1.0, 2.0)])
        self.assertIsNone(gen_args["linear_constraints"])
        self.assertIsNone(gen_args["fixed_features"])

    def test_search_space_not_expanded(self) -> None:
        data = get_data(num_non_sq_arms=0)
        sq_arm = Arm(name="status_quo", parameters={"x": 10.0, "y": 1.0, "z": 1.0})
        experiment = Experiment(
            search_space=self.search_space,
            status_quo=sq_arm,
        )
        trial = experiment.new_trial()
        trial.add_arm(sq_arm)
        trial.mark_running(no_runner_required=True)
        trial.mark_completed()
        experiment.add_tracking_metric(metric=Metric("ax_test_metric"))
        sobol = RandomAdapter(
            search_space=self.search_space,
            generator=SobolGenerator(),
            experiment=experiment,
            data=data,
            transforms=Cont_X_trans,
        )
        # test that search space is not expanded
        sobol.gen(1)
        self.assertEqual(sobol._model_space, sobol._search_space)

    def test_generation_with_all_fixed(self) -> None:
        # Make sure candidate generation succeeds and returns correct parameters
        # when all parameters are fixed.
        exp = get_experiment_for_value()
        adapter = RandomAdapter(
            experiment=exp, generator=SobolGenerator(), transforms=Cont_X_trans
        )
        gr = adapter.gen(n=1)
        self.assertEqual(gr.arms[0].parameters, {"x": 3.0})
