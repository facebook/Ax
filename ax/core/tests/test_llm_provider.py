#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Literal

from ax.core.llm_provider import LLMMessage, LLMProvider
from ax.utils.common.testutils import TestCase


class LLMMessageTest(TestCase):
    def test_llm_message(self) -> None:
        """Test LLMMessage creation and validation."""
        test_cases: list[tuple[Literal["user", "system", "assistant"], str]] = [
            ("user", "Hello"),
            ("system", "You are helpful"),
            ("assistant", "Hi there"),
        ]
        for role, content in test_cases:
            with self.subTest(role=role):
                msg = LLMMessage(role=role, content=content)
                self.assertEqual(msg.role, role)
                self.assertEqual(msg.content, content)
                self.assertEqual(msg.metadata, {})

    def test_llm_message_with_metadata(self) -> None:
        """Test LLMMessage with assistant metadata (response case)."""
        msg = LLMMessage(
            role="assistant",
            content="Hello world",
            metadata={
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
        )
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "Hello world")
        self.assertEqual(msg.metadata["finish_reason"], "stop")
        self.assertEqual(msg.metadata["usage"]["total_tokens"], 30)

        # With minimal fields
        msg_minimal = LLMMessage(role="assistant", content="Hello")
        self.assertEqual(msg_minimal.metadata, {})


class LLMProviderProtocolTest(TestCase):
    def test_protocol_compliance(self) -> None:
        """Test that custom classes can implement the LLMProvider protocol."""

        class MockProvider:
            """A mock provider that implements the LLMProvider protocol."""

            def generate(
                self,
                messages: list[LLMMessage],
                **kwargs: Any,
            ) -> LLMMessage:
                return LLMMessage(
                    role="assistant",
                    content=f"Mock response to: {messages[-1].content}",
                )

        provider = MockProvider()

        # Test that it's recognized as implementing the protocol
        self.assertIsInstance(provider, LLMProvider)

        # Test that it works
        response = provider.generate(
            messages=[LLMMessage(role="user", content="Hello")]
        )
        self.assertEqual(response.role, "assistant")
        self.assertIn("Hello", response.content)
