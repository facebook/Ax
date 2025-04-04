#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.exceptions.core import SearchSpaceExhausted
from ax.modelbridge.random import RandomAdapter
from ax.modelbridge.registry import Cont_X_trans
from ax.models.random.base import RandomGenerator
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_data,
    get_search_space_for_range_values,
    get_small_discrete_search_space,
)


class RandomAdapterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        x = RangeParameter("x", ParameterType.FLOAT, lower=0, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=1, upper=2)
        z = RangeParameter("z", ParameterType.FLOAT, lower=0, upper=5)
        self.parameters = [x, y, z]
        parameter_constraints: list[ParameterConstraint] = [
            OrderConstraint(x, y),
            SumConstraint([x, z], False, 3.5),
        ]
        self.search_space = SearchSpace(self.parameters, parameter_constraints)
        self.experiment = Experiment(search_space=self.search_space)
        self.model_gen_options = {"option": "yes"}

    def test_fit(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, model=RandomGenerator())
        self.assertEqual(adapter.parameters, ["x", "y", "z"])
        self.assertTrue(isinstance(adapter.model, RandomGenerator))

    def test_predict(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, model=RandomGenerator())
        with self.assertRaises(NotImplementedError):
            adapter._predict([])

    def test_cross_validate(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, model=RandomGenerator())
        with self.assertRaises(NotImplementedError):
            adapter._cross_validate(self.search_space, [], [])

    def test_gen_w_constraints(self) -> None:
        adapter = RandomAdapter(experiment=self.experiment, model=RandomGenerator())
        with mock.patch.object(
            adapter.model,
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
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
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
            experiment=Experiment(search_space=search_space), model=RandomGenerator()
        )
        with mock.patch.object(
            adapter.model,
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
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0)])
        self.assertIsNone(gen_args["linear_constraints"])
        self.assertIsNone(gen_args["fixed_features"])

    def test_deduplicate(self) -> None:
        exp = Experiment(search_space=get_small_discrete_search_space())
        sobol = RandomAdapter(
            experiment=exp,
            model=SobolGenerator(deduplicate=True),
            transforms=Cont_X_trans,
        )
        for _ in range(4):  # Search space is {[0, 1], {"red", "panda"}}
            # Generate & attach trials to the experiment so that the
            # generated points are used for deduplication.
            gr = sobol.gen(1)
            exp.new_trial(generator_run=gr).mark_running(no_runner_required=True)
            self.assertEqual(len(gr.arms), 1)
        with self.assertRaises(SearchSpaceExhausted):
            sobol.gen(1)

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
            model=SobolGenerator(),
            experiment=experiment,
            data=data,
            transforms=Cont_X_trans,
        )
        # test that search space is not expanded
        sobol.gen(1)
        self.assertEqual(sobol._model_space, sobol._search_space)

    def test_generated_points(self) -> None:
        # Checks for generated points argument passed to Generator.gen.
        # Search space has two range parameters in [0, 5].
        exp = Experiment(
            search_space=get_search_space_for_range_values(min=0.0, max=5.0)
        )
        generator = SobolGenerator(deduplicate=True)
        gen_res = generator.gen(
            n=1, bounds=[(0.0, 1.0), (0.0, 1.0)], rounding_func=lambda x: x
        )
        # Using Cont_X_trans, particularly UnitX here to test transform application.
        adapter = RandomAdapter(
            experiment=exp, model=generator, transforms=Cont_X_trans
        )

        # No pending points or previous trials on the experiment.
        with mock.patch.object(generator, "gen", return_value=gen_res) as mock_gen:
            adapter.gen(n=1)
        self.assertIsNone(mock_gen.call_args.kwargs["generated_points"])

        # Attach two trials to the experiment.
        exp.new_trial().add_arm(Arm(parameters={"x": 0.0, "y": 0.0})).mark_running(
            no_runner_required=True
        )
        exp.new_trial().add_arm(Arm(parameters={"x": 2.0, "y": 2.0})).mark_running(
            no_runner_required=True
        )
        with mock.patch.object(generator, "gen", return_value=gen_res) as mock_gen:
            adapter.gen(n=1)
        self.assertEqual(
            mock_gen.call_args.kwargs["generated_points"].tolist(),
            [[0.0, 0.0], [0.4, 0.4]],
        )

        # Add pending points -- only unique ones should be passed down.
        pending_observations = {
            m: [ObservationFeatures(parameters={"x": 3.0, "y": 3.0})]
            for m in ("m1", "m2")
        }
        with mock.patch.object(generator, "gen", return_value=gen_res) as mock_gen:
            adapter.gen(n=1, pending_observations=pending_observations)
        self.assertEqual(
            mock_gen.call_args.kwargs["generated_points"].tolist(),
            [[0.0, 0.0], [0.4, 0.4], [0.6, 0.6]],
        )

        # Turn off deduplicate, nothing should be passed down.
        generator.deduplicate = False
        with mock.patch.object(generator, "gen", return_value=gen_res) as mock_gen:
            adapter.gen(n=1, pending_observations=pending_observations)
        self.assertIsNone(mock_gen.call_args.kwargs["generated_points"])
