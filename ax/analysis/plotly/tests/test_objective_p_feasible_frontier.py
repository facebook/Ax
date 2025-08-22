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
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UnsupportedError, UserInputError
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

        adapter = Generators.BOTORCH_MODULAR(experiment=self.experiment)
        card = ObjectivePFeasibleFrontierPlot().compute(
            experiment=self.experiment, adapter=adapter
        )
        self.assertEqual(
            json.loads(card.blob)["layout"]["xaxis"]["title"]["text"], "branin_a"
        )
        self.assertEqual(
            json.loads(card.blob)["layout"]["yaxis"]["title"]["text"],
            "% Chance of Satisfying the Constraints",
        )
        self.assertFalse(card.df["branin_a_sem"].isna().any())
        self.assertTrue(card.df["p_feasible_sem"].isna().all())

    def test_no_exceptions(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "ObjectivePFeasibleFrontierPlot requires an Experiment."
        ):
            ObjectivePFeasibleFrontierPlot().compute(experiment=None)
        opt_config = self.experiment.optimization_config
        self.experiment._optimization_config = None
        with self.assertRaisesRegex(
            UserInputError, "Optimization_config must be set to compute frontier."
        ):
            ObjectivePFeasibleFrontierPlot().compute(experiment=self.experiment)
        self.experiment.optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=m) for m in self.experiment.metrics.values()
                ]
            )
        )
        with self.assertRaisesRegex(
            UnsupportedError, "Multi-objective optimization is not supported."
        ):
            ObjectivePFeasibleFrontierPlot().compute(experiment=self.experiment)
        self.experiment.optimization_config = opt_config
        opt_config.outcome_constraints = []
        with self.assertRaisesRegex(
            UserInputError,
            r"Plotting the objective-p\(feasible\) frontier requires at least one "
            "outcome constraint.",
        ):
            ObjectivePFeasibleFrontierPlot().compute(experiment=self.experiment)
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
        with self.assertRaisesRegex(
            UnsupportedError,
            "Scalarized outcome constraints are not supported yet.",
        ):
            ObjectivePFeasibleFrontierPlot().compute(experiment=self.experiment)
