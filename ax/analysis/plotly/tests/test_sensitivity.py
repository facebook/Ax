# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from itertools import product

from ax.adapter.registry import Generators
from ax.analysis.plotly.sensitivity import (
    compute_sensitivity_adhoc,
    SensitivityAnalysisPlot,
)
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.exceptions.core import UserInputError
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_default_generation_strategy_at_MBM_node
from pyre_extensions import assert_is_instance, none_throws


@mock_botorch_optimize
def get_test_client() -> AxClient:
    client = AxClient()
    client.create_experiment(
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
        parameterization, trial_index = client.get_next_trial()
        client.complete_trial(
            trial_index=trial_index, raw_data={"bar": parameterization["x"] ** 2}
        )
    return client


class TestSensitivityAnalysisPlot(TestCase):
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

        (card,) = analysis.compute(
            generation_strategy=client._generation_strategy
        ).flatten()
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
        self.assertEqual(
            {*card.df.columns},
            {"parameter_name", "sensitivity"},
        )
        self.assertEqual(len(card.df), 2)
        self.assertIsNotNone(card.blob)

        second_order = SensitivityAnalysisPlot(metric_names=["bar"], order="second")
        (card,) = second_order.compute(
            generation_strategy=client._generation_strategy
        ).flatten()
        self.assertEqual(len(card.df), 3)  # 2 first order + 1 second order

    @mock_botorch_optimize
    def test_compute_adhoc(self) -> None:
        metric_mapping = {"bar": "spunky"}
        client = get_test_client()
        data = client.experiment.lookup_data()
        adapter = Generators.BOTORCH_MODULAR(experiment=client.experiment, data=data)
        cards = compute_sensitivity_adhoc(
            adapter=adapter, labels=metric_mapping
        ).flatten()
        self.assertEqual(len(cards), 1)
        card = cards[0]
        self.assertEqual(card.name, "SensitivityAnalysisPlot")
        self.assertEqual(card.title, "Sensitivity Analysis for spunky")

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_online(self) -> None:
        for experiment in get_online_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            # Select an arbitrary metric from the optimization config
            metric_names = [
                none_throws(experiment.optimization_config).objective.metric_names[0]
            ]
            for order, top_k in product(["first", "second", "total"], [None, 1]):
                analysis = SensitivityAnalysisPlot(
                    metric_names=metric_names,
                    # pyre-fixme: Incompatible parameter type [6]: It isn't sure
                    # if "order" has one of the values specified by the Literal
                    order=order,
                    top_k=top_k,
                )

                _ = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )

    @mock_botorch_optimize
    @TestCase.ax_long_test(reason="Expensive to compute Sobol indicies")
    def test_offline(self) -> None:
        for experiment in get_offline_experiments():
            generation_strategy = get_default_generation_strategy_at_MBM_node(
                experiment=experiment
            )
            # Select an arbitrary metric from the optimization config
            metric_names = [
                none_throws(experiment.optimization_config).objective.metric_names[0]
            ]
            for order, top_k in product(["first", "second", "total"], [None, 1]):
                analysis = SensitivityAnalysisPlot(
                    metric_names=metric_names,
                    # pyre-fixme: Incompatible parameter type [6]: It isn't sure
                    # if "order" has one of the values specified by the Literal
                    order=order,
                    top_k=top_k,
                )

                # This prints a lot of warnings about y being constant
                # because the first MOO experiment produces constant data
                _ = analysis.compute(
                    experiment=experiment, generation_strategy=generation_strategy
                )
