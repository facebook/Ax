# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Self


try:
    from thrift.python.serializer import deserialize, Protocol, serialize
    from thrift.python.types import Struct

    UniversalStruct = Struct
    JSONProtocol = Protocol.JSON
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

    UniversalStruct = Any
    JSONProtocol = TJSONProtocol.TJSONProtocolFactory()


class ThriftSerializable(ABC):
    @classmethod
    @abstractmethod
    def thrift_type(cls) -> type[UniversalStruct]: ...

    @abstractmethod
    def serialize(self) -> UniversalStruct: ...

    @classmethod
    @abstractmethod
    def deserialize(cls, struct: UniversalStruct) -> Self: ...

    def to_json(self) -> str:
        struct = self.serialize()

        return serialize(struct, protocol=JSONProtocol).decode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        struct = deserialize(
            cls.thrift_type(), json_str.encode("utf-8"), protocol=JSONProtocol
        )

        return cls.deserialize(struct=struct)
