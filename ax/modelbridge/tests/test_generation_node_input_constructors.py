# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_node_input_constructors import NodeInputConstructors
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase


class TestGenerationNodeInputConstructors(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sobol_model_spec = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        self.sobol_generation_node = GenerationNode(
            node_name="test", model_specs=[self.sobol_model_spec]
        )

    def test_consume_all_n_constructor(self) -> None:
        """Test that the consume_all_n_constructor returns full n."""
        num_to_gen = NodeInputConstructors.ALL_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 5},
        )
        self.assertEqual(num_to_gen, 5)

    def test_repeat_arm_n_constructor(self) -> None:
        """Test that the repeat_arm_n_constructor returns a small percentage of n."""
        small_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 5},
        )
        medium_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 8},
        )
        large_n = NodeInputConstructors.REPEAT_N(
            previous_node=None,
            next_node=self.sobol_generation_node,
            gs_gen_call_kwargs={"n": 11},
        )
        self.assertEqual(small_n, 0)
        self.assertEqual(medium_n, 1)
        self.assertEqual(large_n, 2)

    def test_no_n_provided_error(self) -> None:
        """Test raise error if n is not specified."""
        with self.assertRaisesRegex(
            NotImplementedError,
            "`consume_all_n` only supports cases where n is specified",
        ):
            _ = NodeInputConstructors.ALL_N(
                previous_node=None,
                next_node=self.sobol_generation_node,
                gs_gen_call_kwargs={},
            )
        with self.assertRaisesRegex(
            NotImplementedError,
            " `repeat_arm_n` only supports cases where n is specified",
        ):
            _ = NodeInputConstructors.REPEAT_N(
                previous_node=None,
                next_node=self.sobol_generation_node,
                gs_gen_call_kwargs={},
            )
