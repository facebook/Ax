# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod

import pandas as pd
from ax.core.experiment import Experiment


class BaseAnalysis(ABC):
    """
    Abstract Analysis class for ax.
    This is an interface that defines the methods to be implemented by all analyses.
    Computes an output dataframe for each analysis
    """

    def __init__(self, experiment: Experiment) -> None:
        """
        Initialize the analysis with the experiment object.
        """
        self._experiment = experiment

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @abstractmethod
    def get_df(self) -> pd.DataFrame:
        """
        Return the output of the analysis of this class.
        """
