# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.bandit_rollout import BanditRollout
from ax.core.experiment import Experiment
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_discrete_search_space,
    get_online_experiments,
    get_sobol,
)


class TestBanditRollout(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self.experiment = Experiment(
            search_space=get_discrete_search_space(), name="test_experiment"
        )

        generator = get_sobol(search_space=self.experiment.search_space)
        for _ in range(2):
            gr = generator.gen(n=2)
            self.experiment.new_batch_trial(generator_run=gr)
            self.experiment.new_batch_trial(generator_run=gr)

    def test_compute(self) -> None:
        analysis = BanditRollout()

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        card = analysis.compute(experiment=self.experiment)

        self.assertEqual(card.name, "BanditRollout")
        self.assertEqual(
            card.title, f"Bandit Rollout Weights by Trial for {self.experiment.name}"
        )
        self.assertEqual(
            card.subtitle,
            (
                "The Bandit Rollout visualization provides a comprehensive "
                "overview of the allocation of weights across different trials "
                "and arms. By representing each trial as a distinct axis, this "
                "plot allows for the examination of exploration and exploitation "
                "dynamics over time. It aids in identifying trends and patterns in "
                "arm performance, offering insights into the effectiveness of the "
                "bandit algorithm. Observing the distribution of weights can "
                "reveal correlations and interactions that contribute to the "
                "success or failure of various strategies, enhancing the "
                "understanding of experimental results."
            ),
        )
        self.assertEqual(
            {*card.df.columns},
            {"trial_index", "arm_name", "arm_weight", "normalized_weight"},
        )
        self.assertIsNotNone(card.blob)

    def test_online(self) -> None:
        # Test ParallelCoordinatesPlot can be computed for a variety of experiments
        # which resemble those we see in an online setting.
        for experiment in get_online_experiments():
            analysis = BanditRollout()
            _ = analysis.compute(experiment=experiment)
