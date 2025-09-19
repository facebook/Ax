# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.adapter.factory import get_thompson
from ax.analysis.plotly.marginal_effects import (
    compute_marginal_effects_adhoc,
    MarginalEffectsPlot,
)
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_discrete_search_space, get_sobol


class TestMarginalEffectsPlot(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.experiment = Experiment(
            search_space=get_discrete_search_space(),
            name="test_experiment",
            is_test=True,
            optimization_config=OptimizationConfig(
                objective=Objective(Metric("metric_1", lower_is_better=False))
            ),
        )
        num_arms = 3
        num_trials = 3
        sobol = get_sobol(search_space=self.experiment.search_space)
        gr = sobol.gen(n=num_arms)

        for i in range(num_trials):
            self.experiment.new_batch_trial(generator_run=gr)
            self.experiment.trials[i].mark_running(no_runner_required=True)
            self.experiment.attach_data(
                Data(
                    pd.DataFrame(
                        {
                            "trial_index": [i] * num_arms,
                            "arm_name": [f"0_{j}" for j in range(num_arms)],
                            "metric_name": ["metric_1"] * num_arms,
                            "mean": list(range(num_arms)),
                            "sem": [0.5] * num_arms,
                            "metric_signature": ["metric_1"] * num_arms,
                        }
                    )
                )
            )

        self.analysis_all_variables = MarginalEffectsPlot(
            metric_name="metric_1", parameters=["x", "y", "z"]
        )
        self.analysis_default_params = MarginalEffectsPlot(metric_name="metric_1")

    def test_compute(self) -> None:
        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            self.analysis_default_params.compute()

        adapter = get_thompson(
            experiment=self.experiment,
            data=self.experiment.lookup_data(trial_indices=[0]),
        )

        with self.assertRaisesRegex(
            UserInputError, "MarginalEffectsPlot is only for `ChoiceParameter`s"
        ):
            self.analysis_all_variables.compute(
                experiment=self.experiment, adapter=adapter
            )

        cards = self.analysis_default_params.compute(
            experiment=self.experiment, adapter=adapter
        ).flatten()

        self.assertEqual(cards[0].name, "MarginalEffectsPlot")
        self.assertEqual(
            cards[0].title, "Marginal Effects for z"
        )  # z is the only choice parameter
        self.assertEqual(
            cards[0].subtitle,
            (
                "This plot visualizes the predicted relative changes in "
                "metric_1 for each factor level of z, offering insights "
                "into their impact on the metric. By comparing the "
                "effects of different levels, this plot helps identify which "
                "factor levels have the most significant influence, providing "
                "a detailed understanding of the experimental results and guiding "
                "future decisions on factor selection."
            ),
        )
        self.assertIsNotNone(cards[0].df)
        self.assertIsNotNone(cards[0].blob)
        self.assertEqual(len(cards), 1)
        self.assertEqual(
            set(cards[0].df.columns),
            {"Name", "Level", "Beta", "SE"},
        )

    def test_compute_adhoc(self) -> None:
        adapter = get_thompson(
            experiment=self.experiment,
            data=self.experiment.lookup_data(trial_indices=[0]),
        )

        card = compute_marginal_effects_adhoc(
            metric_name="metric_1",
            experiment=self.experiment,
            adapter=adapter,
        ).flatten()

        self.assertEqual(card[0].name, "MarginalEffectsPlot")
        self.assertEqual(
            card[0].title, "Marginal Effects for z"
        )  # Assuming 'z' is the only choice parameter
        self.assertEqual(
            card[0].subtitle,
            (
                "This plot visualizes the predicted relative changes in "
                "metric_1 for each factor level of z, offering insights "
                "into their impact on the metric. By comparing the "
                "effects of different levels, this plot helps identify which "
                "factor levels have the most significant influence, providing "
                "a detailed understanding of the experimental results and guiding "
                "future decisions on factor selection."
            ),
        )
        self.assertIsNotNone(card[0].df)
        self.assertIsNotNone(card[0].blob)
        self.assertEqual(len(card), 1)
        self.assertEqual(
            set(card[0].df.columns),
            {"Name", "Level", "Beta", "SE"},
        )
