#!/usr/bin/env python3

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.batch_trial import BatchTrial
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.types.types import TParameterization, TParamValue
from ae.lazarus.ae.utils.stats.statstools import agresti_coull_sem


class FactorialMetric(Metric):
    """Metric for testing factorial designs assuming a main effects only
    logit model.
    """

    def __init__(
        self,
        name: str,
        coefficients: Dict[str, Dict[TParamValue, float]],
        batch_size: int = 10000,
        noise_var: float = 0.0,
    ) -> None:
        """
        Args:
            name: name of the metric.
            coefficients: a dictionary mapping
                factors to levels to main effects.
            batch_size: the sample size for one batch, distributed
                between arms proportionally to the design.
            noise_var: used in calculating the probability of
                each arm.
        """
        super(FactorialMetric, self).__init__(name)

        self.coefficients = coefficients
        self.batch_size = batch_size
        self.noise_var = noise_var

    def clone(self) -> "FactorialMetric":
        return FactorialMetric(
            self.name, self.coefficients, self.batch_size, self.noise_var
        )

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> Data:
        if not isinstance(trial, BatchTrial):
            raise ValueError("Factorial metric can only fetch data for batch trials.")
        if not trial.status.expecting_data:
            raise ValueError("Can only fetch data if trial is expecting data.")

        data = []
        normalized_arm_weights = trial.normalized_arm_weights()
        for name, arm in trial.arms_by_name.items():
            weight = normalized_arm_weights[arm]
            mean, sem = evaluation_function(
                parameterization=arm.params,
                weight=weight,
                coefficients=self.coefficients,
                batch_size=self.batch_size,
                noise_var=self.noise_var,
            )
            data.append(
                {
                    "arm_name": name,
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": sem,
                    "trial_index": trial.index,
                }
            )
        return Data(df=pd.DataFrame(data))


def evaluation_function(
    parameterization: TParameterization,
    coefficients: Dict[str, Dict[TParamValue, float]],
    weight: float = 1.0,
    batch_size: int = 10000,
    noise_var: float = 0.0,
) -> Tuple[float, float]:
    probability = _parameterization_probability(
        parameterization=parameterization,
        coefficients=coefficients,
        noise_var=noise_var,
    )
    plays = np.random.binomial(batch_size, weight)
    successes = np.random.binomial(plays, probability)
    mean = float(successes) / plays
    sem = agresti_coull_sem(successes, plays)
    assert isinstance(sem, float)
    return mean, sem


def _parameterization_probability(
    parameterization: TParameterization,
    coefficients: Dict[str, Dict[TParamValue, float]],
    noise_var: float = 0.0,
) -> float:
    z = 0.0
    for factor, level in parameterization.items():
        if factor not in coefficients.keys():
            raise ValueError("{} not in supplied coefficients".format(factor))
        if level not in coefficients[factor].keys():
            raise ValueError("{} not a valid level of {}".format(level, factor))
        z += coefficients[factor][level]
    z += np.sqrt(noise_var) * np.random.randn()
    return np.exp(z) / (1 + np.exp(z))
