# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.plotly.bandit_rollout import BanditRollout
from ax.analysis.plotly.utils import STALE_FAIL_REASON
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.optimization_config import Objective, OptimizationConfig
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
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

    def test_stale_failed_trial_filtering(self) -> None:
        """
        Test that bandit rollout excludes stale failed trials and stale trials but
        includes regular failed trials.
        """
        experiment = Experiment(
            name="bandit_test_stale_filtering",
            search_space=SearchSpace(
                parameters=[
                    ChoiceParameter(
                        name="x", parameter_type=ParameterType.FLOAT, values=[0.0, 1.0]
                    )
                ]
            ),
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric(name="foo"), minimize=False)
            ),
        )

        # Create 4 batch trials with different outcomes
        trials = []
        for i in range(4):
            trial = experiment.new_batch_trial()
            trial.add_arms_and_weights(arms=[Arm(parameters={"x": float(i % 2)})])
            trials.append(trial)

        # Set trial outcomes: success, regular failure, stale failure
        trials[0].mark_running(no_runner_required=True)
        trials[0].mark_completed()
        trials[1].mark_running(no_runner_required=True)
        trials[1].mark_failed(reason="Regular failure")
        trials[2].mark_running(no_runner_required=True)
        trials[2].mark_failed(reason=STALE_FAIL_REASON)
        trials[3].mark_stale()

        card = BanditRollout().compute(experiment=experiment)

        # Verify filtering: include successful (0) + regular failed (1),
        # exclude stale failed (2) and stale (3)
        trial_indices = set(card.df["trial_index"].unique())
        self.assertIn(0, trial_indices)
        self.assertIn(1, trial_indices)
        self.assertNotIn(2, trial_indices)
        self.assertNotIn(3, trial_indices)
