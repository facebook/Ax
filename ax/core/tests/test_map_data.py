# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.map_data import MapData
from ax.exceptions.core import AxError
from ax.utils.common.testutils import TestCase


class TestMapDataGone(TestCase):
    def test_map_data_is_gone(self) -> None:
        with self.assertRaisesRegex(AxError, "MapData no longer exists"):
            MapData(df="foo", bar="baz")
