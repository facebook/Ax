#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Classes for optimizing yields from chemical reactions.

References

.. [Perera2018]
    D. Perera, J. W. Tucker, S. Brahmbhatt, C. Helal, A. Chong, W. Farrell,
    P. Richardson, N. W. Sach. A platform for automated nanomole-scale
    reaction screening and micromole-scale synthesis in flow. Science, 26.
    2018.

.. [Shields2021]
   B. J. Shields, J. Stevens, J. Li, et al. Bayesian reaction optimization
   as a tool for chemical synthesis. Nature 590, 89–96 (2021).

"SUZUKI" involves optimization solvent, ligand, and base combinations
in a Suzuki-Miyaura coupling to optimize carbon-carbon bond formation.
See _[Perera2018] for details.

"DIRECT_ARYLATION" involves optimizing the solvent, base, and ligand chemicals
as well as the temperature and concentration for a direct arylation reaction.
See _[Shields2021] for details.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import ZipFile

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.types import TParameterization, TParamValue


class ChemistryProblemType(Enum):
    SUZUKI: str = "suzuki"
    DIRECT_ARYLATION: str = "direct_arylation"


@dataclass(frozen=True)
class ChemistryData:
    param_names: List[str]
    objective_dict: Dict[Tuple[TParamValue, ...], float]

    def evaluate(self, params: TParameterization) -> float:
        k = tuple(params[pname] for pname in self.param_names)
        return self.objective_dict[k]


@lru_cache(maxsize=8)
def _get_data(problem_type: ChemistryProblemType) -> ChemistryData:
    file_path = Path(__file__).parent.joinpath("chemistry_data.zip").absolute()
    with ZipFile(file_path) as zf:
        with zf.open(f"{problem_type.value}.csv") as f:
            df = pd.read_csv(f, index_col=0)
    param_names = sorted(col for col in df.columns if col != "yield")
    return ChemistryData(
        param_names=param_names,
        objective_dict=df.set_index(param_names)["yield"].to_dict(),
    )


class ChemistryMetric(Metric):
    def __init__(
        self,
        name: str,
        noiseless: bool = False,
        problem_type: ChemistryProblemType = ChemistryProblemType.SUZUKI,
    ) -> None:
        self.noiseless = noiseless
        self.problem_type = problem_type
        super().__init__(name=name, lower_is_better=False)

    def clone(self) -> ChemistryMetric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            problem_type=self.problem_type,
        )

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> Data:
        noise_sd = 0.0 if self.noiseless else float("nan")
        data = _get_data(self.problem_type)
        arm_names = []
        mean = []
        for name, arm in trial.arms_by_name.items():
            arm_names.append(name)
            val = data.evaluate(params=arm.parameters)
            mean.append(val)
        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": self.name,
                "mean": mean,
                "sem": noise_sd,
                "trial_index": trial.index,
            }
        )
        return Data(df=df)
