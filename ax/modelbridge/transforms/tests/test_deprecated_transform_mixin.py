#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

from typing import Any
from unittest.mock import MagicMock

from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.deprecated_transform_mixin import (
    DeprecatedTransformMixin,
)
from ax.utils.common.testutils import TestCase


class DeprecatedTransformTest(TestCase):
    class DeprecatedTransform(DeprecatedTransformMixin, Transform):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)

    class DummyTransform(Transform):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)

    class DeprecatedDummyTransform(DeprecatedTransformMixin, DummyTransform):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)

    def setUp(self) -> None:
        super().setUp()
        self.deprecated_t = self.DeprecatedTransform(MagicMock(), MagicMock())
        self.t = Transform(MagicMock(), MagicMock())

    def test_isinstance(self) -> None:
        self.assertTrue(isinstance(self.deprecated_t, type(self.t)))
        self.assertTrue(isinstance(self.deprecated_t, Transform))
        self.assertTrue(isinstance(self.deprecated_t, self.DeprecatedTransform))
        self.assertTrue(isinstance(self.deprecated_t, DeprecatedTransformMixin))

    def test_deprecated_transform_equality(self) -> None:
        class DeprecatedTransform(DeprecatedTransformMixin, Transform):
            def __init__(self, *args):
                super().__init__(*args)

        t = Transform(MagicMock(), MagicMock())
        t2 = Transform(MagicMock(), MagicMock())
        self.assertEqual(t.__dict__, t2.__dict__)

        dt = DeprecatedTransform(MagicMock(), MagicMock())
        self.assertEqual(t.__dict__, dt.__dict__)

    def test_logging(self) -> None:
        with self.assertLogs(
            "ax.modelbridge.transforms.deprecated_transform_mixin",
            level=logging.WARNING,
        ) as logger:
            _ = self.DeprecatedTransform(MagicMock(), MagicMock())
            message = DeprecatedTransformMixin.warn_deprecated_message(
                self.DeprecatedTransform.__name__,
                Transform.__name__,
            )
            self.assertTrue(any(message in s for s in logger.output))
