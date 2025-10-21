# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Self

from ax.storage.future.thrift.generation_strategy.thrift_types import (
    GenerationNode as ThriftGenerationNode,
    GenerationStrategy as ThriftGenerationStrategy,
)

from ax.storage.future.universal import ThriftSerializable, UniversalStruct
from pyre_extensions import assert_is_instance, override


@dataclass
class GenerationNode(ThriftSerializable):
    name: str

    @classmethod
    @override
    def thrift_type(cls) -> type[UniversalStruct]:
        return ThriftGenerationNode

    @override
    def serialize(self) -> ThriftGenerationNode:
        return ThriftGenerationNode(name=self.name)

    @classmethod
    @override
    def deserialize(cls, struct: UniversalStruct) -> Self:
        node_struct = assert_is_instance(struct, ThriftGenerationNode)

        return cls(name=node_struct.name)


@dataclass
class GenerationStrategy(ThriftSerializable):
    name: str
    nodes: list[GenerationNode]
    current_node_index: int = 0

    @classmethod
    @override
    def thrift_type(cls) -> type[UniversalStruct]:
        return ThriftGenerationStrategy

    @override
    def serialize(self) -> ThriftGenerationStrategy:
        return ThriftGenerationStrategy(
            name=self.name,
            nodes=[node.serialize() for node in self.nodes],
            current_node_index=self.current_node_index,
        )

    @classmethod
    @override
    def deserialize(cls, struct: UniversalStruct) -> Self:
        gs_struct = assert_is_instance(struct, ThriftGenerationStrategy)

        return cls(
            name=gs_struct.name,
            nodes=[GenerationNode.deserialize(node) for node in gs_struct.nodes],
            current_node_index=gs_struct.current_node_index,
        )


sobol_node = GenerationNode(name="sobol")
mbg_node = GenerationNode(name="modular_botorch_generator")

gs = GenerationStrategy(
    name="gpei",
    nodes=[sobol_node, mbg_node],
    current_node_index=1,
)
