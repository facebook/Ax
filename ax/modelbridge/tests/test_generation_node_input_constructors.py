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

    def test_consume_all_n_constructor_no_n(self) -> None:
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
