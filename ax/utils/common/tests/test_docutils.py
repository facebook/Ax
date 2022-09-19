#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.utils.common.docutils import copy_doc
from ax.utils.common.testutils import TestCase


def has_doc() -> None:
    """I have a docstring"""


def has_no_doc() -> None:
    pass


class TestDocUtils(TestCase):
    def test_transfer_doc(self) -> None:
        @copy_doc(has_doc)
        # pyre-fixme[3]: Return type must be annotated.
        def inherits_doc():
            pass

        self.assertEqual(inherits_doc.__doc__, "I have a docstring")

    def test_fail_when_already_has_doc(self) -> None:
        with self.assertRaises(ValueError):

            @copy_doc(has_doc)
            # pyre-fixme[3]: Return type must be annotated.
            def inherits_doc():
                """I already have a doc string"""
                pass

    def test_fail_when_no_doc_to_copy(self) -> None:
        with self.assertRaises(ValueError):

            @copy_doc(has_no_doc)
            # pyre-fixme[3]: Return type must be annotated.
            def f():
                pass
