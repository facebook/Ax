# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from ax.core.data import Data
from ax.utils.common.testutils import TestCase


class AbstractDataTest(TestCase):
    def test_data_column_data_types_default(self):
        self.assertEqual(Data.column_data_types(), Data.COLUMN_DATA_TYPES)

    def test_data_column_data_types_with_extra_columns(self):
        bartype = random.choice([str, int, float])
        columns = Data.column_data_types(extra_column_types={"foo": bartype})
        for c, t in Data.COLUMN_DATA_TYPES.items():
            self.assertEqual(columns[c], t)
        self.assertEqual(columns["foo"], bartype)

    def test_data_column_data_types_with_removed_columns(self):
        columns = Data.column_data_types(excluded_columns=["fidelities"])
        self.assertNotIn("fidelities", columns)
        for c, t in Data.COLUMN_DATA_TYPES.items():
            if c != "fidelities":
                self.assertEqual(columns[c], t)

    # there isn't really a point in doing this
    # this test just documents expected behavior
    # that excluded_columns wins out
    def test_data_column_data_types_with_extra_columns_also_deleted(self):
        bartype = random.choice([str, int, float])
        excluded_columns = ["fidelities", "foo"]
        columns = Data.column_data_types(
            extra_column_types={"foo": bartype},
            excluded_columns=excluded_columns,
        )
        self.assertNotIn("fidelities", columns)
        self.assertNotIn("foo", columns)
        for c, t in Data.COLUMN_DATA_TYPES.items():
            if c not in excluded_columns:
                self.assertEqual(columns[c], t)
