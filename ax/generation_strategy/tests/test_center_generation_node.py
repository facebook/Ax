#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGenOrExhaustion,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from pyre_extensions import none_throws


class TestCenterGenerationNode(TestCase):
    def _create_center_to_sobol_gs(self, exp: Experiment) -> GenerationStrategy:
        """Helper to create a GenerationStrategy with CenterGenerationNode -> Sobol."""
        gs = GenerationStrategy(
            name="test",
            nodes=[
                CenterGenerationNode(next_node_name="sobol"),
                GenerationNode(
                    name="sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                ),
            ],
        )
        gs.experiment = exp
        return gs

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
                DerivedParameter(
                    name="x5",
                    parameter_type=ParameterType.FLOAT,
                    expression_str="x1 + x2",
                ),
            ]
        )
        node = CenterGenerationNode(next_node_name="test")
        self.assertEqual(node.next_node_name, "test")
        self.assertEqual(
            node.transition_criteria,
            [
                AutoTransitionAfterGenOrExhaustion(
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
        self.assertEqual(
            params, {"x1": 2.5, "x2": 31, "x3": "c", "x4": True, "x5": 33.5}
        )

    def test_center_generation_with_logit_scale(self) -> None:
        """Test that center computation works correctly for logit-scale parameters."""
        ss = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=0.9,
                    logit_scale=True,
                ),
                RangeParameter(
                    name="x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )
        node = CenterGenerationNode(next_node_name="test")
        experiment = Experiment(search_space=ss)
        params = (
            none_throws(node.gen(experiment=experiment, pending_observations=None))
            .arms[0]
            .parameters
        )
        # For logit-scale parameter with bounds [0.1, 0.9]:
        # logit(0.1) = log(0.1 / 0.9) ≈ -2.197
        # logit(0.9) = log(0.9 / 0.1) ≈ 2.197
        # center in logit space = 0
        # inverse_logit(0) = 1 / (1 + exp(0)) = 0.5
        self.assertAlmostEqual(float(params["x1"]), 0.5, places=5)
        self.assertEqual(params["x2"], 0.5)

    def test_center_generation_with_logit_scale_extreme_bounds(self) -> None:
        """Test logit-scale with asymmetric extreme bounds from near 0 to near 1."""
        ss = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0001,
                    upper=0.999,
                    logit_scale=True,
                ),
            ]
        )
        node = CenterGenerationNode(next_node_name="test")
        experiment = Experiment(search_space=ss)
        params = (
            none_throws(node.gen(experiment=experiment, pending_observations=None))
            .arms[0]
            .parameters
        )
        # For asymmetric extreme bounds, center is NOT at the linear midpoint
        # logit(0.0001) = log(0.0001) - log(0.9999) ≈ -9.210
        # logit(0.999) = log(0.999) - log(0.001) ≈ 6.906
        # center in logit space = (-9.210 + 6.906) / 2 ≈ -1.152
        # expit(-1.152) ≈ 0.240
        center = float(params["x1"])
        self.assertGreater(center, 0.0001)  # Above lower bound
        self.assertLess(center, 0.999)  # Below upper bound
        # Verify it's at the asymmetric logit-space center (~0.24)
        # NOT at the linear midpoint (0.4995) or symmetric logit center (0.5)
        self.assertAlmostEqual(center, 0.240, places=2)

    def test_deduplication(self) -> None:
        """Test that CenterGenerationNode skips generation and transitions to the next
        node when center already exists.
        """
        exp = get_branin_experiment()
        exp.new_trial().add_arm(arm=Arm({"x1": 2.5, "x2": 7.5})).run()
        node = CenterGenerationNode(next_node_name="test")
        gr = node.gen(experiment=exp, pending_observations=None)
        # The existing arm is the center of search space, so we skip generation
        self.assertIsNone(gr)
        # Verify that the transition criterion is met after skipping
        self.assertTrue(
            node.transition_criteria[0].is_met(experiment=exp, curr_node=node)
        )

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

    def test_successful_center_generation(self) -> None:
        """Test that center is successfully generated when valid and not present."""
        exp = get_branin_experiment()
        gs = self._create_center_to_sobol_gs(exp)
        gr = gs.gen(experiment=exp, n=1)[0]

        # Center should be generated by CenterGenerationNode
        self.assertEqual(len(gr), 1)
        self.assertEqual(gr[0]._generation_node_name, "CenterOfSearchSpace")
        # Check that the generated parameters are the center
        params = gr[0].arms[0].parameters
        self.assertEqual(params["x1"], 2.5)
        self.assertEqual(params["x2"], 7.5)
        # After generating center, should still be on center node
        self.assertEqual(gs.current_node_name, "CenterOfSearchSpace")

        # Next generation should transition to Sobol
        gr2 = gs.gen(experiment=exp, n=1)[0]
        self.assertEqual(gr2[0]._generation_node_name, "sobol")
        self.assertEqual(gs.current_node_name, "sobol")

    def test_with_custom_trials_containing_center(self) -> None:
        """Test CenterGenerationNode transitions when custom trials contain the center.

        This test validates the fix for the issue where CenterGenerationNode would
        fail with a fallback to Sobol when custom trials already contained the center
        point, instead of transitioning to the next node.
        """
        exp = get_branin_experiment()

        # Add a custom trial with the center point of the search space
        center_arm = Arm(parameters={"x1": 2.5, "x2": 7.5})
        exp.new_trial().add_arm(arm=center_arm)
        gs = self._create_center_to_sobol_gs(exp)
        gr = gs.gen(experiment=exp, n=1)[0]

        # CenterGenerationNode should skip generation and transition to Sobol
        # (not use the Fallback_Sobol)
        self.assertEqual(len(gr), 1)
        self.assertEqual(gr[0]._generation_node_name, "sobol")
        self.assertEqual(gr[0]._generator_key, "Sobol")  # Regular Sobol, not Fallback
        self.assertEqual(gs.current_node_name, "sobol")

    def test_with_infeasible_center(self) -> None:
        """Test CenterGenerationNode transitions when the center is infeasible.

        This test validates that CenterGenerationNode properly transitions to the next
        node when the center point violates parameter constraints, instead of using
        a Sobol fallback.
        """
        # Create a search space where the center violates constraints
        ss = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=-5.0,
                    upper=10.0,
                ),
                RangeParameter(
                    name="x2",
                    parameter_type=ParameterType.INT,
                    lower=10.0,
                    upper=100.0,
                    log_scale=True,
                ),
            ],
            parameter_constraints=[  # x1 <= 0
                ParameterConstraint(inequality="x1 <= 0")
            ],
        )
        exp = Experiment(search_space=ss)
        gs = self._create_center_to_sobol_gs(exp)
        gr = gs.gen(experiment=exp, n=1)[0]

        # CenterGenerationNode should skip and transition to Sobol
        # Since the center is infeasible, it should skip and go straight to sobol
        self.assertEqual(len(gr), 1)
        self.assertEqual(gr[0]._generation_node_name, "sobol")
        self.assertEqual(gr[0]._generator_key, "Sobol")
        self.assertEqual(gs.current_node_name, "sobol")
