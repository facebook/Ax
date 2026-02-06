#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
LLM Provider protocol and message types for Ax.

This module defines the core abstractions for LLM integration in Ax:
- LLMMessage: Unified message type for conversations and responses
- LLMProvider: Protocol defining the interface for LLM providers
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    """Represents a single message in a conversation.

    This unified class handles both input messages and LLM responses.
    For assistant responses (role="assistant"), the metadata field
    captures information about the generation.

    Attributes:
        role: Message role - "system", "user", or "assistant"
        content: Message content/text
        metadata: Additional metadata.
            For assistant responses, this may include:
            - "usage": Token usage statistics
            - "finish_reason": Reason for generation completion (e.g., "stop")
    """

    role: Literal["system", "user", "assistant"]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers.

    Any class implementing this protocol can be used as an LLM provider in Ax.
    This enables easy integration of custom LLM backends without requiring
    inheritance from a base class.

    Implementations must provide:
        - generate(): method to generate responses from messages

    Example:
        >>> class MyCustomProvider:
        ...     def generate(
        ...         self,
        ...         messages: list[LLMMessage],
        ...         **kwargs: Any,
        ...     ) -> LLMMessage:
        ...         # Custom implementation
        ...         return LLMMessage(role="assistant", content="response")
        ...
        >>> # Type checker will accept this as LLMProvider
        >>> provider: LLMProvider = MyCustomProvider()
    """

    def generate(
        self,
        messages: list[LLMMessage],
        **kwargs: Any,
    ) -> LLMMessage:
        """Generate a response from a sequence of messages.

        Args:
            messages: List of conversation messages with roles and content
            **kwargs: Provider-specific parameters (e.g., temperature, max_tokens)

        Returns:
            LLMMessage with role="assistant" containing the generated response
        """
        ...
