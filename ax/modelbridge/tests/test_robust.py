#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.core import Objective, OptimizationConfig
from ax.core.objective import MultiObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)
from ax.core.risk_measures import RiskMeasure
from ax.core.types import ComparisonOp
from ax.exceptions.core import UnsupportedError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models.gp_regression import FixedNoiseGP


class TestRobust(TestCase):
    @fast_botorch_optimize
    def test_robust(
        self,
        risk_measure: Optional[RiskMeasure] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        acqf_class: Optional[str] = None,
    ) -> None:
        exp = get_robust_branin_experiment(
            risk_measure=risk_measure,
            optimization_config=optimization_config,
        )

        for _ in range(5):
            modelbridge = Models.BOTORCH_MODULAR(
                experiment=exp,
                data=exp.fetch_data(),
                surrogate=Surrogate(botorch_model_class=FixedNoiseGP),
                botorch_acqf_class=acqf_class or qNoisyExpectedImprovement,
            )
            trial = (
                exp.new_trial(generator_run=modelbridge.gen(1)).run().mark_completed()
            )

        obs = ObservationFeatures(parameters=trial.arm.parameters)
        modelbridge.predict([obs])

    def test_robust_multi_objective(self) -> None:
        risk_measure = RiskMeasure(
            risk_measure="MultiOutputExpectation",
            options={"n_w": 16},
        )
        metrics = [
            BraninMetric(
                name=f"branin_{i}", param_names=["x1", "x2"], lower_is_better=True
            )
            for i in range(2)
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                [
                    Objective(
                        metric=m,
                        minimize=True,
                    )
                    for m in metrics
                ]
            ),
            objective_thresholds=[
                ObjectiveThreshold(metric=m, bound=10.0, relative=False)
                for m in metrics
            ],
            risk_measure=risk_measure,
        )
        self.test_robust(
            risk_measure,
            optimization_config,
            acqf_class=qNoisyExpectedHypervolumeImprovement,
        )

    def test_mars(self) -> None:
        risk_measure = RiskMeasure(
            risk_measure="MARS",
            options={"n_w": 16, "alpha": 0.8},
        )
        metrics = [
            BraninMetric(
                name=f"branin_{i}", param_names=["x1", "x2"], lower_is_better=False
            )
            for i in range(2)
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                [
                    Objective(
                        metric=m,
                        minimize=False,
                    )
                    for m in metrics
                ]
            ),
            objective_thresholds=[
                ObjectiveThreshold(
                    metric=m, bound=10.0, relative=False, op=ComparisonOp.GEQ
                )
                for m in metrics
            ],
            risk_measure=risk_measure,
        )
        self.test_robust(
            risk_measure,
            optimization_config,
            acqf_class=qNoisyExpectedImprovement,
        )

    def test_unsupported_model(self) -> None:
        exp = get_robust_branin_experiment()
        with self.assertRaisesRegex(UnsupportedError, "support robust"):
            Models.GPEI(
                experiment=exp,
                data=exp.fetch_data(),
            ).gen(n=1)
