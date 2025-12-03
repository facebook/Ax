# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.graphviz.hierarchical_search_space_graph import (
    HierarchicalSearchSpaceGraph,
    SUBTITLE,
)
from ax.analysis.search_space_summary import SearchSpaceSummary
from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_offline_experiments, get_online_experiments


class TestSearchSpaceSummary(TestCase):
    def test_compute(self) -> None:
        client = Client()

        client.configure_experiment(
            name="foo",
            parameters=[
                RangeParameterConfig(
                    name="learning_rate",
                    bounds=(1e-10, 1.0),
                    parameter_type="float",
                    scaling="log",
                ),
                ChoiceParameterConfig(
                    name="activation_fn",
                    values=["tanh", "sigmoid", "relu"],
                    parameter_type="str",
                ),
                ChoiceParameterConfig(
                    name="optimizer",
                    values=["sgd", "adam"],
                    parameter_type="str",
                    dependent_parameters={
                        "sgd": [
                            "momentum",
                            "dampening",
                            "use_gradient_clipping",
                            "use_weight_decay",
                        ],
                        "adam": ["beta1", "beta2", "epsilon"],
                    },
                ),
                # SGD-specific hyperparameters
                RangeParameterConfig(
                    name="momentum",
                    bounds=(0.0, 0.99),
                    parameter_type="float",
                ),
                RangeParameterConfig(
                    name="dampening",
                    bounds=(0.0, 1.0),
                    parameter_type="float",
                ),
                # Adam-specific hyperparameters
                RangeParameterConfig(
                    name="beta1",
                    bounds=(0.0, 0.99),
                    parameter_type="float",
                ),
                RangeParameterConfig(
                    name="beta2",
                    bounds=(0.0, 0.999),
                    parameter_type="float",
                ),
                RangeParameterConfig(
                    name="epsilon",
                    bounds=(1e-10, 1e-4),
                    parameter_type="float",
                ),
                ChoiceParameterConfig(
                    name="use_gradient_clipping",
                    values=[True, False],
                    parameter_type="bool",
                    dependent_parameters={
                        True: ["clip_norm_type", "clip_norm_max"],
                    },
                ),
                ChoiceParameterConfig(
                    name="clip_norm_type",
                    values=["l1", "l2", "inf"],
                    parameter_type="str",
                ),
                RangeParameterConfig(
                    name="clip_norm_max",
                    bounds=(1e-10, 1.0),
                    parameter_type="float",
                ),
                ChoiceParameterConfig(
                    name="use_weight_decay",
                    values=[True, False],
                    parameter_type="bool",
                    dependent_parameters={
                        True: ["weight_decay"],
                        False: [],
                    },
                ),
                RangeParameterConfig(
                    name="weight_decay",
                    bounds=(1e-10, 1.0),
                    parameter_type="float",
                    scaling="log",
                ),
            ],
        )

        analysis = HierarchicalSearchSpaceGraph()
        card = analysis.compute(experiment=client._experiment)

        # Test metadata
        self.assertEqual(card.name, "HierarchicalSearchSpaceGraph")
        self.assertEqual(card.title, "Hierarchical Search Space Graph")
        self.assertEqual(card.subtitle, SUBTITLE)
        self.assertIsNotNone(card.blob)

        # Test dataframe for accuracy
        search_space_summary_card = SearchSpaceSummary().compute(
            experiment=client._experiment
        )
        pd.testing.assert_frame_equal(card.df, search_space_summary_card.df)

        # Test digraph
        dot = card.get_digraph()
        source = dot.source

        for parameter in client._experiment.search_space.top_level_parameters.values():
            self.assertIn(parameter.name, source)

            if parameter.is_hierarchical:
                for (
                    parameter_value,
                    dependent_parameter_names,
                ) in parameter.dependents.items():
                    self.assertIn(f"cluster_{parameter.name}_{parameter_value}", source)
                    self.assertIn(
                        f"{parameter.name} -> cluster_{parameter.name}_{parameter_value}_anchor",  # noqa[E501]
                        source,
                    )

                    for dependent_parameter_name in dependent_parameter_names:
                        self.assertIn(dependent_parameter_name, source)

    def test_online(self) -> None:
        analysis = HierarchicalSearchSpaceGraph()
        for experiment in get_online_experiments():
            # If validation fails (i.e. this Experiment is not applicable to HSSGraph)
            # then skip it in the tests
            if analysis.validate_applicable_state(experiment=experiment) is not None:
                continue

            _ = analysis.compute(experiment=experiment)

    def test_offline(self) -> None:
        analysis = HierarchicalSearchSpaceGraph()
        for experiment in get_offline_experiments():
            # If validation fails (i.e. this Experiment is not applicable to HSSGraph)
            # then skip it in the tests
            if analysis.validate_applicable_state(experiment=experiment) is not None:
                continue

            _ = analysis.compute(experiment=experiment)
