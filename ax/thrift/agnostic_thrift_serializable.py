# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from typing import Self

try:
    from thrift.python.serializer import deserialize, Protocol, serialize
    from thrift.python.types import Struct

    AgnosticStruct = Struct
    AgnosticJSONProtocol = Protocol.JSON
except ImportError:  # Use Apache Thrift if thrift-python is not available.
    from typing import Any

    from thrift.protocol import TJSONProtocol
    from thrift.transport import TTransport

    def serialize(struct, protocol):
        transport = TTransport.TMemoryBuffer()
        proto = protocol.getProtocol(transport)
        struct.write(proto)

        return transport.getvalue()

    def deserialize(klass, buf, protocol):
        transport = TTransport.TMemoryBuffer(buf)
        proto = protocol.getProtocol(transport)
        obj = klass()
        obj.read(proto)
        return obj

    AgnosticStruct = Any
    AgnosticJSONProtocol = TJSONProtocol.TJSONProtocolFactory()


class AgnosticThriftSerializable(ABC):
    @classmethod
    @abstractmethod
    def thrift_type(cls) -> type[AgnosticStruct]: ...

    @abstractmethod
    def serialize(self) -> AgnosticStruct: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, struct: AgnosticStruct) -> Self: ...

    def to_json(self) -> str:
        struct = self.serialize()

        return serialize(struct, protocol=AgnosticJSONProtocol).decode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        struct = deserialize(
            cls.thrift_type(), json_str.encode("utf-8"), protocol=AgnosticJSONProtocol
        )

        return cls.deserialize(struct=struct)
