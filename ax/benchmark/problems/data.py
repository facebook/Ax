# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class AbstractParquetDataLoader(ABC):
    def __init__(
        self,
        benchmark_name: str,
        dataset_name: str,
        stem: str,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialize the ParquetDataLoader.

        This class provides a way to load Parquet data from an external URL,
        caching it locally to avoid repeated downloads.
        It downloads the file from the external URL and saves it to the cache
        if it's not already cached, and reads from the cache otherwise.

        Args:
            dataset_name (str): The name of the dataset to load.
            stem (str): The stem of the parquet file.
            cache_dir (Path): The directory where cached data will be stored.
                Defaults to '~/.cache/ax_benchmark_data'.
        """
        self.cache_dir: Path = (
            cache_dir
            if cache_dir is not None
            else Path("~/.cache").expanduser().joinpath("ax_benchmark_data")
        )
        self.benchmark_name = benchmark_name
        self.dataset_name = dataset_name
        self.stem = stem

    @property
    def filename(self) -> str:
        """
        Get the filename of the cached file.

        This method returns the filename of the cached file, which is the stem
        followed by the extension '.parquet.gzip'.

        Returns:
            str: The filename of the cached file.
        """
        return f"{self.stem}.parquet.gzip"

    @property
    def cache_path(self) -> Path:
        """
        Get the path to the cached file.

        This method returns the path where the cached file should be stored.

        Returns:
            Path: The path to the cached file.
        """
        return self.cache_dir.joinpath(
            self.benchmark_name,
            self.dataset_name,
            self.filename,
        )

    def is_cached(self) -> bool:
        """
        Check if the data is already cached (whether the file simply exists).

        Returns:
            bool: True if the data is cached, False otherwise.
        """
        return self.cache_path.exists()

    def load(self, download: bool = True) -> pd.DataFrame:
        """
        Read the parquet data from the cache or download it from the URL.

        If the data is cached, this method reads the data from the cache.
        If the data is not cached and download is True, this method downloads
        the data from the URL, caches it, and then returns the data.
        If the data is not cached and download is False, this method raises an OSError.

        Args:
            download (bool): Whether to download the data if it's not available
                locally. If False, this method raises an OSError. Defaults to True.

        Returns:
            pd.DataFrame: The loaded parquet data.
        """
        if self.is_cached():
            with self.cache_path.open("rb") as infile:
                return pd.read_parquet(infile, engine="pyarrow")
        if download:
            if self.url is None:
                raise ValueError(
                    f"File {self.cache_path} does not exist, "
                    "`download` is True, but URL is not specified."
                )
            return self._fetch_and_cache()
        raise ValueError(
            f"File {self.cache_path} does not exist and `download` is False"
        )

    def _fetch_and_cache(self) -> pd.DataFrame:
        """
        Download the data from the URL and cache it.

        This method downloads the data from the URL, creates the cache directory
        if needed, and saves the data to the cache.

        Returns:
            pd.DataFrame: The downloaded parquet data.
        """
        # Download the data from the URL
        data = pd.read_parquet(self.url, engine="pyarrow")
        # Create the cache directory if needed
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as outfile:
            data.to_parquet(outfile, engine="pyarrow", compression="gzip")
        return data

    @property
    @abstractmethod
    def url(self) -> str | None:
        """
        Get the URL of the parquet file.

        This method should return the URL of the parquet file to download.
        None is allowed to support cases where the user manually populates the
        download cache beforehand.

        Returns:
            str | None: The URL of the parquet file or None.
        """
        pass
