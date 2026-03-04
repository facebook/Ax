#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest.mock import patch

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
from ax.exceptions.generation_strategy import (
    AxGenerationException,
    GenerationStrategyRepeatedPoints,
)
from ax.generation_strategy.center_generation_node import CenterGenerationNode
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import AutoTransitionAfterGen
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from pyre_extensions import none_throws


class TestCenterGenerationNode(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.complex_ss = SearchSpace(
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
        self.node = CenterGenerationNode(next_node_name="test")
        self.branin_exp = get_branin_experiment()
        self.complex_exp = Experiment(search_space=self.complex_ss)
        self.center_to_sobol_gs = GenerationStrategy(
            name="test",
            nodes=[
                CenterGenerationNode(next_node_name="sobol"),
                GenerationNode(
                    name="sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                ),
            ],
        )
        self.center_to_sobol_gs.experiment = self.branin_exp
        self.constrained_ss = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=-5.0,
                    upper=10.0,
                ),
                RangeParameter(
                    name="x2",
                    parameter_type=ParameterType.FLOAT,
                    lower=10.0,
                    upper=100.0,
                ),
            ],
            parameter_constraints=[ParameterConstraint(inequality="x1 <= 0")],
        )
        self.constrained_exp = Experiment(search_space=self.constrained_ss)
        self.constrained_gs = GenerationStrategy(
            name="test",
            nodes=[
                CenterGenerationNode(next_node_name="sobol"),
                GenerationNode(
                    name="sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                ),
            ],
        )
        self.constrained_gs.experiment = self.constrained_exp

    def test_center_generation(self) -> None:
        self.assertEqual(self.node.next_node_name, "test")
        self.assertEqual(
            self.node.transition_criteria,
            [
                AutoTransitionAfterGen(
                    transition_to="test", continue_trial_generation=False
                )
            ],
        )
        params = (
            none_throws(
                self.node.gen(experiment=self.complex_exp, pending_observations=None)
            )
            .arms[0]
            .parameters
        )
        self.assertEqual(self.node.search_space, self.complex_ss)
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
        experiment = Experiment(search_space=ss)
        params = (
            none_throws(self.node.gen(experiment=experiment, pending_observations=None))
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
        experiment = Experiment(search_space=ss)
        params = (
            none_throws(self.node.gen(experiment=experiment, pending_observations=None))
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
        self.branin_exp.new_trial().add_arm(arm=Arm({"x1": 2.5, "x2": 7.5})).run()
        gr = self.node.gen(experiment=self.branin_exp, pending_observations=None)
        # The existing arm is the center of search space, so we skip generation
        self.assertIsNone(gr)
        # Verify that the transition criterion is met after skipping
        self.assertTrue(
            self.node.transition_criteria[0].is_met(
                experiment=self.branin_exp, curr_node=self.node
            )
        )

    def test_repr(self) -> None:
        self.assertEqual(
            repr(self.node),
            "CenterGenerationNode(next_node_name='test',"
            " use_existing_trials_for_initialization=False)",
        )

    def test_equality(self) -> None:
        node2 = CenterGenerationNode(next_node_name="test")
        self.assertEqual(self.node, node2)
        other_node = CenterGenerationNode(next_node_name="test2")
        self.assertNotEqual(self.node, other_node)
        # Still equal after generation, despite the search spaces being different.
        # The two nodes will function the same if we call gen with a different
        # experiment after this, so I think this is fine.
        self.node.gen(experiment=self.branin_exp, pending_observations=None)
        self.assertEqual(self.node, node2)

    def test_successful_center_generation(self) -> None:
        """Test that center is successfully generated when valid and not present."""
        gr = self.center_to_sobol_gs.gen(experiment=self.branin_exp, n=1)[0]

        # Center should be generated by CenterGenerationNode
        self.assertEqual(len(gr), 1)
        self.assertEqual(gr[0]._generation_node_name, "CenterOfSearchSpace")
        # Check that the generated parameters are the center
        params = gr[0].arms[0].parameters
        self.assertEqual(params["x1"], 2.5)
        self.assertEqual(params["x2"], 7.5)
        # After generating center, should still be on center node
        self.assertEqual(
            self.center_to_sobol_gs.current_node_name, "CenterOfSearchSpace"
        )

        # Next generation should transition to Sobol
        gr2 = self.center_to_sobol_gs.gen(experiment=self.branin_exp, n=1)[0]
        self.assertEqual(gr2[0]._generation_node_name, "sobol")
        self.assertEqual(self.center_to_sobol_gs.current_node_name, "sobol")

    def test_with_custom_trials_containing_center(self) -> None:
        """Test CenterGenerationNode transitions when custom trials contain the center.

        This test validates the fix for the issue where CenterGenerationNode would
        fail with a fallback to Sobol when custom trials already contained the center
        point, instead of transitioning to the next node.
        """
        # Add a custom trial with the center point of the search space
        center_arm = Arm(parameters={"x1": 2.5, "x2": 7.5})
        self.branin_exp.new_trial().add_arm(arm=center_arm)
        gr = self.center_to_sobol_gs.gen(experiment=self.branin_exp, n=1)[0]

        # CenterGenerationNode should skip generation and transition to Sobol
        # (not use the Fallback_Sobol)
        self.assertEqual(len(gr), 1)
        self.assertEqual(gr[0]._generation_node_name, "sobol")
        self.assertEqual(gr[0]._generator_key, "Sobol")  # Regular Sobol, not Fallback
        self.assertEqual(self.center_to_sobol_gs.current_node_name, "sobol")

    def test_with_naive_infeasible_constraints_and_multitype_ss(self) -> None:
        self.complex_ss.add_parameter_constraints(
            # x1 <= 0
            parameter_constraints=[ParameterConstraint(inequality="x1 <= 0")]
        )
        self.assertEqual(
            set(self.node.fallback_specs.keys()),
            {AxGenerationException, GenerationStrategyRepeatedPoints},
        )
        exp = Experiment(search_space=self.complex_ss)
        params = (
            none_throws(self.node.gen(experiment=exp, pending_observations=None))
            .arms[0]
            .parameters
        )
        self.assertEqual(self.node.search_space, self.complex_ss)
        # -2.5 is chebyshev center of [-5, 0]
        # x2, x3, x4 are all types of params which will not use chebyshev
        # (logscale, choice, fixed)
        # x5 is x1+x2, validates incorporates updated value of x1 w/ chebyshev
        self.assertEqual(
            params, {"x1": -2.5, "x2": 31, "x3": "c", "x4": True, "x5": 28.5}
        )

    def test_chebyshev_center_returns_none_skips_generation(self) -> None:
        """Test that when chebyshev center returns None, the node skips generation
        and sets _should_skip to True, allowing transition to next node.
        """
        # Mock chebyshev center on the search space to return None
        with patch.object(
            self.constrained_ss, "compute_chebyshev_center", return_value=None
        ):
            gr = self.node.gen(
                experiment=self.constrained_exp, pending_observations=None
            )

        # Should return None and set _should_skip to True
        self.assertIsNone(gr)
        self.assertTrue(self.node._should_skip)

    def test_chebyshev_returns_none_gs_uses_sobol_not_fallback(self) -> None:
        """Test that when chebyshev center returns None in a GS, it transitions to
        the regular Sobol node rather than using Fallback_Sobol.

        This verifies that the CenterGenerationNode properly skips and transitions
        to the next node when center computation fails.
        """
        # Mock chebyshev center on the search space to return None
        with patch.object(
            self.constrained_ss, "compute_chebyshev_center", return_value=None
        ):
            gr = self.constrained_gs.gen(experiment=self.constrained_exp, n=1)[0]

        # Should transition to regular Sobol node, not use Fallback_Sobol
        self.assertEqual(gr[0]._generation_node_name, "sobol")
        self.assertEqual(gr[0]._generator_key, "Sobol")
