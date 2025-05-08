# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import (
    AnalysisBlobAnnotation,
    AnalysisCardCategory,
    AnalysisCardLevel,
)
from ax.analysis.plotly.sensitivity import (
    compute_sensitivity_adhoc,
    SensitivityAnalysisPlot,
)
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.modelbridge.registry import Generators
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


class TestSensitivityAnalysisPlot(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.client = AxClient()
        self.client.create_experiment(
            is_test=True,
            name="foo",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [-1.0, 1.0],
                }
            ],
            objectives={"bar": ObjectiveProperties(minimize=True)},
        )

        for _ in range(10):
            parameterization, trial_index = self.client.get_next_trial()
            self.client.complete_trial(
                trial_index=trial_index, raw_data={"bar": parameterization["x"] ** 2}
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

        analysis = SensitivityAnalysisPlot(metric_names=["bar"], order="first")

        with self.assertRaisesRegex(
            UserInputError, "Must provide either a GenerationStrategy or an Adapter"
        ):
            analysis.compute()

        (card,) = analysis.compute(generation_strategy=client._generation_strategy)
        self.assertEqual(
            card.name,
            "SensitivityAnalysisPlot",
        )
        self.assertEqual(card.title, "Sensitivity Analysis for bar")
        self.assertEqual(
            card.subtitle,
            "Understand how each parameter affects bar according to a first-order "
            "sensitivity analysis.",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(card.category, AnalysisCardCategory.INSIGHT)
        self.assertEqual(
            {*card.df.columns},
            {"parameter_name", "sensitivity"},
        )
        self.assertEqual(len(card.df), 2)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, AnalysisBlobAnnotation.PLOTLY)

        second_order = SensitivityAnalysisPlot(metric_names=["bar"], order="second")
        (card,) = second_order.compute(generation_strategy=client._generation_strategy)
        self.assertEqual(len(card.df), 3)  # 2 first order + 1 second order

    @mock_botorch_optimize
    def test_compute_adhoc(self) -> None:
        metric_mapping = {"bar": "spunky"}
        data = self.client.experiment.lookup_data()
        adapter = Generators.BOTORCH_MODULAR(
            experiment=self.client.experiment, data=data
        )
        cards = compute_sensitivity_adhoc(adapter=adapter, labels=metric_mapping)
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.name, "SensitivityAnalysisPlot")
        self.assertEqual(card.title, "Sensitivity Analysis for spunky")

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_online(self) -> None:
        # Test SensitivityAnalysisPlot can be computed for a variety of experiments
        # which resemble those we see in an online setting.

        for experiment in get_online_experiments():
            for order in ["first", "second", "total"]:
                for top_k in [None, 1]:
                    generation_strategy = get_default_generation_strategy_at_MBM_node(
                        experiment=experiment
                    )
                    analysis = SensitivityAnalysisPlot(
                        # Select and arbitrary metric from the optimization config
                        metric_names=[
                            none_throws(
                                experiment.optimization_config
                            ).objective.metric_names[0]
                        ],
                        order=order,  # pyre-ignore[6] Valid Literal
                        top_k=top_k,
                    )

                    _ = analysis.compute(
                        experiment=experiment, generation_strategy=generation_strategy
                    )

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_offline(self) -> None:
        # Test SensitivityAnalysisPlot can be computed for a variety of experiments
        # which resemble those we see in an offline setting.

        for experiment in get_offline_experiments():
            for order in ["first", "second", "total"]:
                for top_k in [None, 1]:
                    generation_strategy = get_default_generation_strategy_at_MBM_node(
                        experiment=experiment
                    )
                    analysis = SensitivityAnalysisPlot(
                        # Select and arbitrary metric from the optimization config
                        metric_names=[
                            none_throws(
                                experiment.optimization_config
                            ).objective.metric_names[0]
                        ],
                        order=order,  # pyre-ignore[6] Valid Literal
                        top_k=top_k,
                    )

                    _ = analysis.compute(
                        experiment=experiment, generation_strategy=generation_strategy
                    )
