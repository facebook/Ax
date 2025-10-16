# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from ax.analysis.plotly.top_surfaces import TopSurfacesAnalysis
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_offline_experiments_subset,
    get_online_experiments_subset,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


class TestTopSurfacesAnalysis(TestCase):
    def test_validate_applicable_state(self) -> None:
        self.assertIn(
            "Requires an Experiment",
            none_throws(TopSurfacesAnalysis().validate_applicable_state()),
        )

    @mock_botorch_optimize
    def test_compute(self) -> None:
        client = Client()
        client.configure_experiment(
            name="foo",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        client.configure_optimization(objective="bar")

        for _ in range(6):
            for trial_index, parameterization in client.get_next_trials(
                max_trials=1
            ).items():
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "bar": assert_is_instance(parameterization["x1"], float)
                        - 2 * assert_is_instance(parameterization["x2"], float)
                    },
                )

        analysis = TopSurfacesAnalysis(metric_name="bar", order="first")

        cards = analysis.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        ).flatten()

        # First card should be the sensitivity analysis.
        self.assertEqual(cards[0].title, "Sensitivity Analysis for bar")

        # Other cards should be slices.
        self.assertIn("vs. bar", cards[1].title)
        self.assertIn("vs. bar", cards[2].title)

        second = TopSurfacesAnalysis(metric_name="bar", order="second")

        second_card_group = second.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        )

        self.assertEqual(
            {card.name for card in second_card_group.children},
            {
                "TopSurfaceAnalysisSlicePlots",
                "TopSurfaceAnalysisContourPlots",
                "SensitivityAnalysisPlot",
            },
        )

        with_surfaces = second_card_group.flatten()

        self.assertEqual(len(with_surfaces), 4)

        # First card should be the sensitivity analysis.
        self.assertEqual(with_surfaces[0].title, "Sensitivity Analysis for bar")

        # Other cards should be slices or contours.
        self.assertIn("vs. bar", with_surfaces[1].title)
        self.assertIn("vs. bar", with_surfaces[2].title)
        self.assertIn("vs. bar", with_surfaces[3].title)

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_compute_categorical_parameters(self) -> None:
        client = Client()
        client.configure_experiment(
            name="foo",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                ChoiceParameterConfig(
                    name="x2",
                    parameter_type="int",
                    values=[*range(10)],
                    is_ordered=False,
                ),
            ],
        )
        client.configure_optimization(objective="bar")

        for _ in range(6):
            for trial_index, parameterization in client.get_next_trials(
                max_trials=1
            ).items():
                client.complete_trial(
                    trial_index=trial_index,
                    raw_data={
                        "bar": assert_is_instance(parameterization["x1"], float)
                        - 2 * assert_is_instance(parameterization["x2"], int)
                    },
                )

        analysis = TopSurfacesAnalysis(metric_name="bar")

        cards = analysis.compute(
            experiment=client._experiment,
            generation_strategy=client._generation_strategy,
        ).flatten()

        # Only plot x1 vs bar since x2 is categorical.
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].title, "Sensitivity Analysis for bar")
        self.assertEqual(cards[1].title, "x1 vs. bar")

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_online(self) -> None:
        # Test TopSurfacesAnalysis can be computed for a variety of experiments
        # which resemble those we see in an online setting.
        order = "total"  # most common

        for experiment in get_online_experiments_subset():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            analysis = TopSurfacesAnalysis(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0],
                order=order,
                top_k=2,
            )

            _ = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

        # test with default top k
        for experiment in get_online_experiments_subset():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            analysis = TopSurfacesAnalysis(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0],
                order=order,
            )

            _ = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_offline(self) -> None:
        # Test TopSurfacesAnalysis can be computed for a variety of experiments
        # which resemble those we see in an offline setting, in analogous tests we
        # run all experiments with modifications to settings, however, this test
        # is slow and so we limit the number of permutations we validate.
        order = "total"  # most common

        for experiment in get_offline_experiments_subset():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            analysis = TopSurfacesAnalysis(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0],
                order=order,
                top_k=2,
            )

            _ = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )

        # run the test with the default top k
        for experiment in get_offline_experiments_subset():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            analysis = TopSurfacesAnalysis(
                # Select and arbitrary metric from the optimization config
                metric_name=none_throws(
                    experiment.optimization_config
                ).objective.metric_names[0],
                order=order,
            )

            _ = analysis.compute(
                experiment=experiment, generation_strategy=generation_strategy
            )
