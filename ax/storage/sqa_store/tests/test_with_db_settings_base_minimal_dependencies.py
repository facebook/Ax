# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.utils.common.testutils import TestCase


class TestWithDbSetingsBaseMinimalDependencies(TestCase):
    @patch.dict("sys.modules", {"sqlalchemy": None})
    def test_with_db_settings_base_no_sql_alchemy(self) -> None:
        from ax.storage.sqa_store.with_db_settings_base import (  # noqa
            WithDBSettingsBase,
        )
