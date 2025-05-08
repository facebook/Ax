#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.transition_criterion import AutoTransitionAfterGen
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from pyre_extensions import none_throws


class TestCenterGenerationNode(TestCase):
    def test_center_generation(self) -> None:
        ss = SearchSpace(
            parameters=[
                RangeParameter(  # Simple float.
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=-5.0,
                    upper=10.0,
                ),
                RangeParameter(  # Integer with log transform.
                    name="x2",
                    parameter_type=ParameterType.INT,
                    lower=10.0,
                    upper=100.0,
                    log_scale=True,
                ),
                ChoiceParameter(  # Ordered choice.
                    name="x3",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c", "d"],
                    is_ordered=True,
                ),
                FixedParameter(  # Fixed parameter.
                    name="x4",
                    parameter_type=ParameterType.BOOL,
                    value=True,
                ),
            ]
        )
        node = CenterGenerationNode(next_node_name="test")
        self.assertEqual(node.next_node_name, "test")
        self.assertEqual(
            node.transition_criteria,
            [
                AutoTransitionAfterGen(
                    transition_to="test", continue_trial_generation=False
                )
            ],
        )
        experiment = Experiment(search_space=ss)
        params = (
            none_throws(node.gen(experiment=experiment, pending_observations=None))
            .arms[0]
            .parameters
        )
        self.assertEqual(node.search_space, ss)
        self.assertEqual(params, {"x1": 2.5, "x2": 31, "x3": "c", "x4": True})

    def test_deduplication(self) -> None:
        exp = get_branin_experiment()
        exp.new_trial().add_arm(arm=Arm({"x1": 2.5, "x2": 7.5})).run()
        node = CenterGenerationNode(next_node_name="test")
        gr = none_throws(node.gen(experiment=exp, pending_observations=None))
        self.assertEqual(gr._model_key, "Sobol")

    def test_repr(self) -> None:
        node = CenterGenerationNode(next_node_name="test")
        self.assertEqual(
            repr(node),
            "CenterGenerationNode(next_node_name='test')",
        )

    def test_equality(self) -> None:
        node = CenterGenerationNode(next_node_name="test")
        node2 = CenterGenerationNode(next_node_name="test")
        self.assertEqual(node, node2)
        other_node = CenterGenerationNode(next_node_name="test2")
        self.assertNotEqual(node, other_node)
        # Still equal after generation, despite the search spaces being different.
        # The two nodes will function the same if we call gen with a different
        # experiment after this, so I think this is fine.
        exp = get_branin_experiment()
        node.gen(experiment=exp, pending_observations=None)
        self.assertEqual(node, node2)
