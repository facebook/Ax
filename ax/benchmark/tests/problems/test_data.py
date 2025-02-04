# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from io import BufferedReader
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ax.benchmark.problems.data import AbstractParquetDataLoader

from ax.utils.common.testutils import TestCase


class ConcreteParquetDataLoader(AbstractParquetDataLoader):
    @property
    def url(self) -> str | None:
        return (
            f"https://example.com/{self.benchmark_name}"
            "/main/{self.dataset_name}/{self.filename}"
        )


class TestParquetDataLoader(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.test_data = ConcreteParquetDataLoader(
            benchmark_name="test_benchmark",
            dataset_name="test_dataset",
            stem="test_stem",
            cache_dir=Path("/tmp/test_cache"),
        )

    def tearDown(self) -> None:
        # Delete the cached file if it exists
        self.test_data.cache_path.unlink(missing_ok=True)

    def test_read_cached(self) -> None:
        # Create a mock cached file
        self.test_data.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.test_data.cache_path.touch()

        with patch(
            "pandas.read_parquet",
            return_value=pd.DataFrame(),
        ) as mock_read_parquet:
            result = self.test_data.load()

        # Assert that the cached file was read
        mock_read_parquet.assert_called_once()
        self.assertIsInstance(mock_read_parquet.call_args.args[0], BufferedReader)
        self.assertEqual(
            mock_read_parquet.call_args.args[0].name, str(self.test_data.cache_path)
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_read_not_cached_download_true(self) -> None:
        with patch(
            "pandas.read_parquet",
            return_value=pd.DataFrame(),
        ) as mock_read_parquet:
            # Call the load method with download=True
            result = self.test_data.load(download=True)

        # Assert that the data was downloaded and cached
        mock_read_parquet.assert_called_once_with(self.test_data.url, engine="pyarrow")

        # Assert that the cached file now exists
        self.assertTrue(self.test_data.is_cached())
        self.assertIsInstance(result, pd.DataFrame)

    def test_read_not_cached_download_false(self) -> None:
        # Call the load method with download=False
        with self.assertRaisesRegex(
            ValueError, "File .* does not exist and `download` is False"
        ):
            self.test_data.load(download=False)
