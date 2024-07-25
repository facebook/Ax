# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.core.experiment import Experiment


class BaseAnalysis:
    """
    Abstract Analysis class for ax.
    This is an interface that defines the methods to be implemented by all analyses.
    Computes an output dataframe for each analysis
    """

    def __init__(
        self,
        experiment: Experiment,
        df_input: Optional[pd.DataFrame] = None,
        # TODO: add support for passing in experiment name, and markdown message
    ) -> None:
        """
        Initialize the analysis with the experiment object.
        For scenarios where an analysis output is already available,
        we can pass the dataframe as an input.
        """
        self._experiment = experiment
        self._df: Optional[pd.DataFrame] = df_input

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the output of the analysis of this class.
        """
        if self._df is None:
            self._df = self.get_df()
        return self._df

    def get_df(self) -> pd.DataFrame:
        """
        Return the output of the analysis of this class.
        Subclasses should overwrite this.
        """
        raise NotImplementedError("get_df must be implemented by subclass")
