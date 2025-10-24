# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json

import torch
from ax.adapter.registry import Generators

from ax.analysis.plotly.objective_p_feasible_frontier import (
    ObjectivePFeasibleFrontierPlot,
)
from ax.core.arm import Arm
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment_with_multi_objective,
    get_branin_metric,
)
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.utils.testing import skip_if_import_error
from pyre_extensions import none_throws


class TestObjectivePFeasibleFrontierPlot(TestCase):
    def setUp(self) -> None:
        super().setUp()

        torch.manual_seed(0)
        self.experiment = get_branin_experiment_with_multi_objective(
            with_completed_batch=True, with_absolute_constraint=True, num_objectives=3
        )
        self.experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=self.experiment.metrics["branin_a"]),
            outcome_constraints=self.experiment.optimization_config.outcome_constraints,
        )
        oc = none_throws(self.experiment.optimization_config).outcome_constraints[0]
        oc.bound = 10.0
        oc.op = ComparisonOp.LEQ

    @skip_if_import_error
    @mock_botorch_optimize
    def test_compute(self) -> None:
        # import pymoo here to raise an import error if pymoo is not installed.
        # The import error is already handled in
        # ax/generators/torch/botorch_modular/acquisition.py so we need to
        # reimport it here
        import pymoo  # noqa: F401

        for pruning in (False, True):
            target = Arm(parameters={"x1": 0.0, "x2": 0.0}) if pruning else None
            self.experiment.optimization_config.pruning_target_parameterization = target
            adapter = Generators.BOTORCH_MODULAR(
                experiment=self.experiment,
                acquisition_options={"prune_irrelevant_parameters": pruning},
            )
            card = ObjectivePFeasibleFrontierPlot().compute(
                experiment=self.experiment, adapter=adapter
            )
            layout = json.loads(card.blob)["layout"]
            self.assertEqual(layout["xaxis"]["title"]["text"], "branin_a")
            self.assertEqual(
                layout["yaxis"]["title"]["text"],
                "% Chance of Satisfying the Constraints",
            )
            self.assertFalse(card.df["branin_a_sem"].isna().any())
            self.assertTrue(card.df["p_feasible_sem"].isna().all())

    def test_validate_applicable_state(self) -> None:
        self.assertIn(
            "Requires an Experiment.",
            none_throws(ObjectivePFeasibleFrontierPlot().validate_applicable_state()),
        )

        opt_config = self.experiment.optimization_config
        self.experiment._optimization_config = None
        self.assertIn(
            "Optimization_config must be set to compute frontier",
            none_throws(
                ObjectivePFeasibleFrontierPlot().validate_applicable_state(
                    experiment=self.experiment
                )
            ),
        )

        self.experiment.optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=m) for m in self.experiment.metrics.values()
                ]
            )
        )

        self.assertIn(
            "Multi-objective optimization is not supported.",
            none_throws(
                ObjectivePFeasibleFrontierPlot().validate_applicable_state(
                    experiment=self.experiment
                )
            ),
        )

        self.experiment.optimization_config = opt_config
        opt_config.outcome_constraints = []
        self.assertIn(
            "requires at least one outcome constraint.",
            none_throws(
                ObjectivePFeasibleFrontierPlot().validate_applicable_state(
                    experiment=self.experiment
                )
            ),
        )
        self.experiment.add_tracking_metric(get_branin_metric("branin2"))
        # Get only tracking metrics, excluding the objective metric to avoid
        # "Cannot constrain on objective metric" error
        constraint_metrics = [
            self.experiment.metrics["branin_b"],
            self.experiment.metrics["branin_c"],
        ]
        opt_config.outcome_constraints = [
            ScalarizedOutcomeConstraint(
                metrics=constraint_metrics,
                weights=[1.0, 1.0],
                relative=False,
                bound=10.0,
                op=ComparisonOp.LEQ,
            )
        ]
        self.assertIn(
            "Scalarized outcome constraints are not supported yet.",
            none_throws(
                ObjectivePFeasibleFrontierPlot().validate_applicable_state(
                    experiment=self.experiment
                )
            ),
        )
